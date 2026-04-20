# Telegram RAG Bot

A multilingual, profile-aware Retrieval-Augmented Generation (RAG) chatbot delivered over Telegram. The system answers user questions strictly from a government-scheme knowledge base (`gov_corpus.json`) and supports two LLM backends: a local HuggingFace model and the Google Gemini API.

---

## Architecture

The system is composed of **three independent FastAPI/Uvicorn services** that communicate over HTTP on localhost:

| Service | File | Default Port |
|---|---|---|
| Bot Webhook Server | `main.py` | 8000 |
| LLM Inference Server | `llm_server.py` | 8001 |
| RAG Retrieval Server | `rag_server.py` | 8002 |

Persistent storage is handled by a single SQLite file (`bot.db`) via `aiosqlite`. All database logic lives in `database.py`. Configuration for all three services is centralised in `config.py`.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI webhook server (port 8000). Handles Telegram updates, onboarding, rate limiting, queue management, and the end-to-end RAG pipeline. |
| `llm_server.py` | LLM inference server (port 8001). Supports HuggingFace (local GPU) and Google Gemini backends with automatic model-fallback on 503 errors. |
| `rag_server.py` | RAG retrieval server (port 8002). Loads the corpus at startup, encodes it with LaBSE + BGE-M3, and serves a 7-method MRR ensemble search with BGE reranker. |
| `database.py` | Async SQLite layer (aiosqlite). Manages `users`, `user_profiles`, `messages`, and `query_cache` tables. |
| `config.py` | Central configuration; all values are read from environment variables with defaults. |
| `gov_corpus.json` | Hierarchical knowledge base of government schemes. Each document has Ministry, Title, Tags, Details, Benefits, Eligibility, Application Process, and Documents Required fields. |
| `requirements.txt` | Pinned Python dependencies for all three services. |
| `start.sh` | Bash script: creates the Python 3.11 `uv` virtual environment, installs dependencies, kills old processes, starts all three services as background jobs, and registers the Telegram webhook. |
| `stop.sh` | Bash script: gracefully stops all three services using the saved `.pids` file, then force-kills any processes still bound to the ports. |
| `notes.txt` | Developer scratch notes (run commands, webhook registration commands, API key references). |

---

## Database Schema

Four SQLite tables (defined in `database.py`):

```
users          — chat_id, username, first_name, created_at, last_active,
                 message_count, is_blocked, onboarding_done, onboarding_step
user_profiles  — chat_id (FK→users), age, gender, occupation, education, updated_at
messages       — chat_id (FK→users), role ('user'|'assistant'), content,
                 rag_context, latency_ms, created_at
query_cache    — query_hash (MD5), query_text, answer, hits, created_at, expires_at
```

WAL journal mode and foreign key enforcement are enabled on every connection.

---

## LLM Backends (`llm_server.py`)

Selected via the `BACKEND` environment variable (default: `gemini`).

### Gemini backend (`BACKEND=gemini`)
- Requires `GEMINI_API_KEY`.
- Primary model: `gemini-3-flash-preview` (configurable via `GEMINI_MODEL`).
- Automatic fallback list: `gemini-3.1-pro-preview` → `gemini-3.1-flash-lite-preview` → `gemini-2.5-pro` → `gemini-pro-latest` → `gemini-flash-latest`.
- Falls through to the next model on 503 / 429 / resource-exhausted errors.
- Inference runs via `asyncio.to_thread` (non-blocking).

### HuggingFace backend (`BACKEND=hf`)
- Default model: `bharatgenai/Param2-17B-A2.4B-Thinking` (configurable via `MODEL_ID`).
- Loaded in `bfloat16` with `device_map="auto"`.
- Uses `torch.inference_mode()` and greedy decoding (`do_sample=False`).
- Runs two warmup inferences at startup (lengths 10 and 50 tokens) to eliminate cold-start latency.
- Batch endpoint (`/infer/batch`) uses left-padded batching on CUDA.

### LLM endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"/"loading", "model_loaded": bool, "model_id": str, "backend": str}` |
| `/infer` | POST | Single inference. Body: `{prompt, max_new_tokens, temperature, top_p}` |
| `/infer/batch` | POST | Batch inference. Body: list of the same. Max batch size: `MAX_BATCH_SIZE` (default 4). |

---

## RAG Pipeline (`rag_server.py`)

All models are loaded once at startup and held in global memory (the server must run with `workers=1`).

### Models loaded
| Variable | Model | Use |
|---|---|---|
| `models["baseline"]` | `sentence-transformers/LaBSE` | Dense baseline embeddings |
| `models["bge"]` | `BAAI/bge-m3` | Dense + sparse (lexical) embeddings |
| `models["reranker"]` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking |

### Corpus indexing
- Each document section (Details, Benefits, Eligibility, Application Process, Documents Required) is extracted into a chunk with the text: `Ministry / Scheme Title / Tags / Section / Information`.
- All chunks are embedded on startup.

### Search (7-method MRR ensemble)
1. LaBSE cosine similarity (rankA)
2. BGE-M3 sparse (lexical) scores (rankB)
3. BGE-M3 dense cosine scores (rankC)
4. Hybrid = 0.7 × dense_norm + 0.3 × sparse_norm (rankD)
5. BGE reranker on top-20 of rankB (rankE)
6. BGE reranker on top-20 of rankC (rankF)
7. BGE reranker on top-20 of rankD (rankG)

MRR (1/rank) is summed across all 7 lists. Results are deduplicated to unique documents; up to `MAX_RESULTS` (5) documents are returned.

### RAG endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns status, device, chunk count, doc count |
| `/retrieve` | POST | Body: `{query, top_k}`. Returns `{context, chunks, latency_ms}` |

---

## Bot Features (`main.py`)

### User onboarding (multi-step inline keyboard)
New users are taken through a 4-step onboarding flow before they can ask questions:

| Step | Field | Options |
|---|---|---|
| 1/4 | Age | Under 18, 18–24, 25–34, 35–44, 45–54, 55–64, 65+ |
| 2/4 | Gender | Male, Female, Non-binary, Prefer not to say |
| 3/4 | Occupation | Student, Employed (Private), Employed (Government), Self-employed / Business, Unemployed, Retired, Other |
| 4/4 | Education | No formal education, Primary school, Secondary school / High school, Diploma / Vocational, Bachelor's degree, Master's degree, Doctorate (PhD), Other |

Profile data is saved to the `user_profiles` table and injected into every LLM prompt.

### Bot commands
| Command | Aliases | Action |
|---|---|---|
| `/start` | — | Welcome returning user or begin onboarding for new user |
| `/profile` | — | Display current profile |
| `/editprofile` | `/edit` | Re-run onboarding to update profile |
| `/clear` | — | Delete all conversation history for the user |
| `/stats` | — | Show message count, question count, last-active time |
| `/deleteaccount` | `/delete` | Prompt for confirmation, then erase all user data |
| `/help` | `/h`, `/menu` | Show help text with all commands and rate-limit info |

### Reply-keyboard menu buttons
After onboarding a persistent reply keyboard is shown with five buttons:
- 📊 My Stats
- 👤 My Profile
- 🗑️ Clear History
- ✏️ Edit Profile
- ❓ Help

### Concurrency and queuing
- `asyncio.Queue` with max size `MAX_QUEUE_SIZE` (default 50).
- `NUM_WORKERS` worker coroutines (default 4) drain the queue.
- `asyncio.Semaphore(MAX_CONCURRENT_TASKS)` (default 4) limits parallel pipeline runs.
- Duplicate in-flight requests (same chat_id + query) are deduplicated via `in_flight` dict.

### Rate limiting
- In-memory sliding window: `RATE_LIMIT_MESSAGES` (default 5) messages per `RATE_LIMIT_WINDOW` (default 10) seconds per user.

### Context-aware RAG query building
- A follow-up detection heuristic checks whether all content tokens in the current message are in a fixed set of function/filler words.
- If the query is a follow-up, the earliest topic-bearing prior user turn is prepended as a topic anchor before sending to the RAG server.

### Caching
- Answers are cached in the `query_cache` SQLite table with a TTL of `CACHE_TTL` (default 300 s).
- Answers that start with ⚠️ or contain "I don't know based on the available information" are not cached.
- A background coroutine purges expired cache entries every 300 s.

### LLM prompt structure
The prompt sent to the LLM contains (in order):
1. System instruction (multilingual, strict context-only answering rules).
2. User profile block (age, gender, occupation, education).
3. RAG context retrieved from the corpus.
4. Last 6 conversation turns.
5. The current user question.
6. Output instruction (respond in same language as the question).

### Message limits
- Maximum message length: `MAX_MESSAGE_LENGTH` (default 500 characters).
- Outgoing Telegram messages are split into 4096-character chunks automatically.

### Bot webhook endpoint
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{status, queue_size, in_flight, workers}` |
| `/webhook` | POST | Receives all Telegram updates |

---

## Configuration (`config.py`)

All settings are read from environment variables. Defaults are shown below.

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `gemini` | LLM backend: `gemini` or `hf` |
| `TELEGRAM_BOT_TOKEN` | *(hardcoded fallback)* | Telegram bot token |
| `GEMINI_API_KEY` | *(hardcoded fallback)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Primary Gemini model |
| `GEMINI_THINK_LEVEL` | `MINIMAL` | Thinking level: `MINIMAL`, `MEDIUM`, `HIGH` |
| `BOT_PORT` | `8000` | Port for `main.py` |
| `LLM_PORT` | `8001` | Port for `llm_server.py` |
| `RAG_PORT` | `8002` | Port for `rag_server.py` |
| `LLM_SERVICE_URL` | `http://localhost:8001` | URL `main.py` uses to call LLM server |
| `RAG_SERVICE_URL` | `http://localhost:8002` | URL `main.py` uses to call RAG server |
| `MAX_QUEUE_SIZE` | `50` | Max tasks in the bot's async queue |
| `MAX_CONCURRENT_TASKS` | `4` | Semaphore limit for pipeline runs |
| `NUM_WORKERS` | `4` | Number of worker coroutines |
| `MAX_MEMORY` | `10` | Conversation turns kept per user |
| `CACHE_TTL` | `300` | Cache time-to-live in seconds |
| `RATE_LIMIT_MESSAGES` | `5` | Max messages per window |
| `RATE_LIMIT_WINDOW` | `10` | Rate-limit window in seconds |
| `MAX_MESSAGE_LENGTH` | `500` | Max incoming message length |
| `RAG_TIMEOUT` | `5.0` | HTTP timeout for RAG calls (seconds) |
| `LLM_TIMEOUT` | `180.0` | HTTP timeout for LLM calls (seconds) |
| `DB_PATH` | `bot.db` | SQLite file path |
| `MODEL_ID` | `bharatgenai/Param2-17B-A2.4B-Thinking` | HuggingFace model ID |
| `MODEL_MAX_TOKENS` | `512` | Max new tokens for HF backend |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature |
| `MODEL_TOP_P` | `0.9` | Top-p sampling |
| `INFER_TIMEOUT` | `300.0` | Per-request inference timeout for LLM server |
| `MAX_BATCH_SIZE` | `4` | Max batch size for `/infer/batch` |

---

## Dependencies (`requirements.txt`)

| Package | Version | Role |
|---|---|---|
| fastapi | 0.111.0 | Web framework for all three servers |
| uvicorn[standard] | 0.29.0 | ASGI server |
| pydantic | 2.7.1 | Request/response validation |
| httpx | 0.27.0 | Async HTTP client (inter-service calls) |
| aiosqlite | 0.20.0 | Async SQLite |
| numpy | <2 | Numerical operations in RAG |
| torch | 2.2.1 | ML tensor operations |
| transformers | 4.42.4 | HuggingFace model loading |
| tokenizers | 0.19.1 | Fast tokenisation |
| accelerate | 0.34.2 | Device-map and mixed-precision support |
| bitsandbytes | 0.43.1 | Quantisation utilities |
| peft | 0.11.1 | Parameter-efficient fine-tuning utilities |
| sentencepiece | latest | Tokenisation for multilingual models |
| safetensors | latest | Safe model weight serialisation |
| einops | latest | Tensor reshaping utilities |
| setuptools | latest | Build tools |
| datasets | 3.6.0 | HuggingFace datasets |
| google-genai | latest | Google Gemini SDK |
| sentence-transformers | 3.0.1 | LaBSE embedding model |
| FlagEmbedding | 1.2.11 | BGE-M3 and BGE reranker |
| scikit-learn | latest | MinMaxScaler for score normalisation |

---

## Startup (`start.sh`)

```
bash start.sh
```

Steps performed:
1. Creates Python 3.11 virtual environment with `uv` (`.venv/`) if absent.
2. Activates the environment.
3. Upgrades pip and installs `requirements.txt`.
4. Kills any existing processes on ports 8000, 8001, 8002.
5. Starts `rag_server.py` → waits for `/health` to return `{"status": "ok"}` (up to 240 s).
6. Starts `llm_server.py` → waits for `/health`.
7. Starts `main.py` (bot) → waits for `/health`.
8. Saves the three PIDs to `.pids`.
9. Attempts to register the Telegram webhook automatically if `TELEGRAM_BOT_TOKEN` and `WEBHOOK_HOST` (or `LIGHTNING_CLOUDSPACE_HOST`) are set.

Logs are written to `logs/rag.log`, `logs/llm.log`, `logs/bot.log`.

### ⚠️ Mandatory Post-Startup Step — Register the Telegram Webhook

After all three services are confirmed running, you **must** register the webhook manually if the automatic step in `start.sh` did not complete (e.g., `WEBHOOK_HOST` was not set). Replace `<your-studio-id>` with your actual Lightning AI CloudSpace ID:

```bash
export TELEGRAM_BOT_TOKEN="your_token_here"

curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
     -d "url=https://8000-<your-studio-id>.cloudspaces.litng.ai/webhook"
```

A successful registration returns:

```json
{"ok":true,"result":true,"description":"Webhook was set"}
```

To verify the webhook is active:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo"
```

The bot will **not receive any messages from Telegram** until this step is completed.

## Shutdown (`stop.sh`)

```
bash stop.sh
```

Steps performed:
1. Reads PIDs from `.pids` (if present).
2. Sends `SIGTERM` to each process; waits up to 5 s for graceful exit.
3. Sends `SIGKILL` if the process is still alive after 5 s.
4. Force-kills any remaining processes bound to ports 8000, 8001, 8002 via `lsof`.
5. Removes `.pids`.

---

## Deployment Notes

- The system was developed and tested on **Lightning AI CloudSpaces**. The `start.sh` script derives the public webhook URL from the `LIGHTNING_CLOUDSPACE_HOST` environment variable injected by the platform.
- `TELEGRAM_BOT_TOKEN` and `GEMINI_API_KEY` should be stored as platform secrets / environment variables, not hard-coded.
- `rag_server.py` must run with exactly **one Uvicorn worker** because all three embedding models and the corpus index live in global Python memory.
- All services communicate on `localhost`; no external network exposure is needed for inter-service calls.
