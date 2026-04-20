"""
rag_server.py  —  Self-contained RAG micro-service (port 8002)

Everything runs in ONE process — no external search URL needed.

Embedded pipeline:
    ✅ LaBSE baseline embeddings
    ✅ BGE-M3 dense + sparse embeddings
    ✅ BGE reranker v2-m3
    ✅ Hybrid MRR ensemble (7 ranking methods)
    ✅ Top-5 unique document retrieval
    ✅ Structured chunk extraction & context building

Start:
    python rag_server.py
"""

import os
import json
import logging
import time

import torch
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sklearn.preprocessing import MinMaxScaler

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RAG] %(levelname)s  %(message)s"
)
logger = logging.getLogger("rag_server")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA          = 0.7          # dense/sparse hybrid weight
CORPUS_PATH    = "gov_corpus.json"
TOP_K_DEFAULT  = 3
MAX_RESULTS    = 5            # unique docs returned by search pipeline

EXPECTED_SECTIONS = [
    "Details", "Benefits", "Eligibility",
    "Application Process", "Documents Required",
]

# ── Global state ──────────────────────────────────────────────────────────────
models: dict             = {}
corpus_documents: dict   = {}   # doc_id → full JSON object
corpus_data: dict        = {    # flat chunk arrays for embedding lookup
    "ids":      [],
    "texts":    [],
    "metadata": [],             # [{doc_id, chunk_id, section_matched}, …]
}
corpus_embeddings: dict  = {}   # pre-computed tensors / sparse dicts

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load everything once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⏳ Loading corpus and models — please wait…")

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    logger.info("📂 Reading %s", CORPUS_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        hierarchical_corpus = json.load(f)

    global corpus_documents
    corpus_documents = hierarchical_corpus

    for doc_id, doc_data in hierarchical_corpus.items():
        ministry = doc_data.get("Ministry", "Unknown")
        title    = doc_data.get("Title",    "Unknown")
        tags     = ", ".join(doc_data.get("Tags", []))

        for section in EXPECTED_SECTIONS:
            if section not in doc_data:
                continue
            chunk_id   = f"{doc_id}_{section.lower().replace(' ', '_')}"
            content    = doc_data[section]
            chunk_text = (
                f"Ministry: {ministry}\n"
                f"Scheme Title: {title}\n"
                f"Tags: {tags}\n"
                f"Section: {section}\n"
                f"Information: {content}"
            )
            corpus_data["ids"].append(chunk_id)
            corpus_data["texts"].append(chunk_text)
            corpus_data["metadata"].append({
                "doc_id":           doc_id,
                "chunk_id":         chunk_id,
                "section_matched":  section,
            })

    logger.info("✅ Corpus: %d searchable chunks from %d documents",
                len(corpus_data["texts"]), len(hierarchical_corpus))

    # ── 2. Load models ────────────────────────────────────────────────────────
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("🤖 Loading LaBSE…")
    models["baseline"] = SentenceTransformer(
        "sentence-transformers/LaBSE", device=DEVICE
    )

    logger.info("🤖 Loading BGE-M3…")
    models["bge"] = BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True, device=DEVICE
    )

    logger.info("🤖 Loading BGE reranker…")
    models["reranker"] = FlagReranker(
        "BAAI/bge-reranker-v2-m3", use_fp16=True, device=DEVICE
    )

    models["scaler"] = MinMaxScaler()

    # ── 3. Encode corpus once ─────────────────────────────────────────────────
    logger.info("🔢 Encoding corpus with LaBSE…")
    corpus_embeddings["d_emb_A"] = models["baseline"].encode(
        corpus_data["texts"], batch_size=64, convert_to_tensor=True
    )

    logger.info("🔢 Encoding corpus with BGE-M3 (dense + sparse)…")
    d_out = models["bge"].encode(
        corpus_data["texts"], batch_size=64,
        return_dense=True, return_sparse=True,
    )
    corpus_embeddings["d_dense"]  = torch.tensor(d_out["dense_vecs"], device=DEVICE)
    corpus_embeddings["d_sparse"] = d_out["lexical_weights"]

    logger.info("🚀 RAG server ready!")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("🛑 Clearing models from memory…")
    models.clear()
    corpus_embeddings.clear()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Server", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:  str
    top_k:  int = TOP_K_DEFAULT


class RetrieveResponse(BaseModel):
    context:    str
    chunks:     list[str]
    latency_ms: float

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fast_sparse_dot(q_sparse: dict, d_sparse_list: list) -> np.ndarray:
    """Dot product between a query sparse dict and a list of doc sparse dicts."""
    return np.array([
        sum(w * d.get(t, 0.0) for t, w in q_sparse.items())
        for d in d_sparse_list
    ])


def _rerank(query: str, idxs: list[int]) -> list[int]:
    """Re-rank a shortlist of chunk indices using the cross-encoder reranker."""
    pairs  = [[query, corpus_data["texts"][i]] for i in idxs]
    scores = models["reranker"].compute_score(pairs, batch_size=32)
    return [i for i, _ in sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)]


def _search(query: str) -> list[dict]:
    """
    Full 7-method MRR ensemble search.
    Returns up to MAX_RESULTS unique-document result dicts.
    """
    scaler = MinMaxScaler()   # fresh per query — thread-safe

    # ── Encode query ──────────────────────────────────────────────────────────
    q_emb_A = models["baseline"].encode([query], convert_to_tensor=True)[0]
    q_out   = models["bge"].encode([query], return_dense=True, return_sparse=True)
    q_dense  = torch.tensor(q_out["dense_vecs"][0], device=DEVICE)
    q_sparse = q_out["lexical_weights"][0]

    # ── Score all chunks ──────────────────────────────────────────────────────
    scores_A      = torch.matmul(corpus_embeddings["d_emb_A"], q_emb_A).cpu().numpy()
    scores_C      = torch.matmul(corpus_embeddings["d_dense"],  q_dense).cpu().numpy()
    sparse_scores = _fast_sparse_dot(q_sparse, corpus_embeddings["d_sparse"])

    # Normalised hybrid
    dense_norm  = scaler.fit_transform(scores_C.reshape(-1, 1)).flatten()
    sparse_norm = scaler.fit_transform(sparse_scores.reshape(-1, 1)).flatten()
    hybrid_scores = ALPHA * dense_norm + (1 - ALPHA) * sparse_norm

    # ── Initial top-50 rankings ───────────────────────────────────────────────
    rankA = np.argsort(scores_A)[::-1][:50].tolist()
    rankB = np.argsort(sparse_scores)[::-1][:50].tolist()
    rankC = np.argsort(scores_C)[::-1][:50].tolist()
    rankD = np.argsort(hybrid_scores)[::-1][:50].tolist()

    # ── Rerank top-20 of sparse / dense / hybrid ──────────────────────────────
    rankE = _rerank(query, rankB[:20])
    rankF = _rerank(query, rankC[:20])
    rankG = _rerank(query, rankD[:20])

    # ── MRR ensemble across all 7 methods ─────────────────────────────────────
    mrr: dict[int, float] = {}
    for ranked_list in [rankA, rankB, rankC, rankD, rankE, rankF, rankG]:
        for rank, chunk_idx in enumerate(ranked_list):
            mrr[chunk_idx] = mrr.get(chunk_idx, 0.0) + 1.0 / (rank + 1)

    final_order = sorted(mrr, key=mrr.get, reverse=True)

    # ── Deduplicate to unique documents ───────────────────────────────────────
    results: list[dict] = []
    seen_docs: set[str] = set()

    for chunk_idx in final_order:
        meta   = corpus_data["metadata"][chunk_idx]
        doc_id = meta["doc_id"]

        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        results.append({
            "rank":             len(results) + 1,
            "doc_id":           doc_id,
            "matched_on_section": meta["section_matched"],
            "mrr_score":        round(mrr[chunk_idx], 4),
            "document_content": corpus_documents[doc_id],
        })

        if len(results) == MAX_RESULTS:
            break

    return results


def _build_chunks(results: list[dict], top_k: int) -> list[str]:
    """
    Extract readable text chunks from the top-k search results.
    Sections included: Title, Details, Benefits, Eligibility, Application Process.
    """
    chunks: list[str] = []
    for r in results[:top_k]:
        doc   = r.get("document_content", {})
        parts = []
        for section in ["Title", "Details", "Benefits", "Eligibility", "Application Process"]:
            value = doc.get(section)
            if value:
                parts.append(f"{section}: {value}")
        chunk = "\n".join(parts)
        if chunk:
            chunks.append(chunk)
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "device":       DEVICE,
        "chunks_loaded": len(corpus_data["texts"]),
        "docs_loaded":  len(corpus_documents),
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.time()

    try:
        import asyncio
        results = await asyncio.to_thread(_search, req.query)
        chunks  = _build_chunks(results, req.top_k)
    except Exception as e:
        logger.exception("❌ Retrieval error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    context    = "\n\n---\n\n".join(chunks)
    latency_ms = (time.time() - t0) * 1000

    logger.info("📚 Retrieved %d chunks in %.0f ms | query: %.50s…",
                len(chunks), latency_ms, req.query)

    return RetrieveResponse(
        context=context,
        chunks=chunks,
        latency_ms=latency_ms,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("RAG_PORT", "8002"))
    logger.info("📚 Starting self-contained RAG server on port %d  device=%s",
                port, DEVICE)
    uvicorn.run(
        "rag_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,   # MUST be 1 — models live in global memory
    )