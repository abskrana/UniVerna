"""
main.py  —  Telegram Bot webhook server (port 8000).
Full pipeline: Webhook → Onboarding → Queue → RAG → LLM → DB → Telegram reply.

New features:
  ✅ Multi-step onboarding (age, gender, occupation, education) via inline keyboards
  ✅ User profile stored in DB and injected as context into every LLM prompt
  ✅ /menu command — persistent reply keyboard with main actions
  ✅ /profile — view or edit profile
  ✅ /clear  — clear chat history (with confirmation)
  ✅ /delete — delete account & all data

Start:
    python main.py
"""

import asyncio
import logging
import time
import os
import json

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from config import (
    TELEGRAM_API_URL, TELEGRAM_BOT_TOKEN,
    LLM_SERVICE_URL, RAG_SERVICE_URL,
    MAX_QUEUE_SIZE, MAX_CONCURRENT_TASKS, NUM_WORKERS,
    MAX_MEMORY, CACHE_TTL,
    RATE_LIMIT_MESSAGES, RATE_LIMIT_WINDOW, MAX_MESSAGE_LENGTH,
    RAG_TIMEOUT, LLM_TIMEOUT,
    BOT_PORT,
)
from database import (
    init_db, upsert_user, get_user, get_history,
    save_message, clear_history, get_user_stats,
    cache_get, cache_set, cache_purge_expired,
    # profile
    get_profile, upsert_profile, delete_profile,
    set_onboarding_step, complete_onboarding, reset_onboarding,
    get_user_questions,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(levelname)s  %(message)s"
)
logger = logging.getLogger("bot")

# ── Globals ──────────────────────────────────────────────────────────────────
task_queue  = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
semaphore   = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
in_flight   : dict[str, asyncio.Task] = {}
user_hits   : dict[int, list[float]]  = {}

workers:  list[asyncio.Task] = []
bg_tasks: list[asyncio.Task] = []

# ── Onboarding config ────────────────────────────────────────────────────────
ONBOARDING_STEPS = ["age", "gender", "occupation", "education"]

ONBOARDING_OPTIONS: dict[str, list[str]] = {
    "age": [
        "Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"
    ],
    "gender": [
        "Male", "Female", "Non-binary", "Prefer not to say"
    ],
    "occupation": [
        "Student", "Employed (Private)", "Employed (Government)",
        "Self-employed / Business", "Unemployed", "Retired", "Other"
    ],
    "education": [
        "No formal education", "Primary school",
        "Secondary school / High school",
        "Diploma / Vocational", "Bachelor's degree",
        "Master's degree", "Doctorate (PhD)", "Other"
    ],
}

ONBOARDING_QUESTIONS: dict[str, str] = {
    "age":        "📅 <b>Step 1/4 — Age</b>\nWhat is your age group?",
    "gender":     "⚧ <b>Step 2/4 — Gender</b>\nHow do you identify?",
    "occupation": "💼 <b>Step 3/4 — Occupation</b>\nWhat best describes your current occupation?",
    "education":  "🎓 <b>Step 4/4 — Education</b>\nWhat is your highest level of education?",
}

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    for i in range(NUM_WORKERS):
        t = asyncio.create_task(worker(i + 1))
        workers.append(t)
    bg_tasks.append(asyncio.create_task(cache_purge_loop()))
    logger.info("🚀 Bot server started with %d workers", NUM_WORKERS)
    yield
    logger.info("🛑 Shutting down…")
    for t in workers + bg_tasks:
        t.cancel()
    await asyncio.gather(*(workers + bg_tasks), return_exceptions=True)
    logger.info("✅ Shutdown complete")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Telegram RAG Bot", version="2.0.0", lifespan=lifespan)

# ─────────────────────────────────────────────────────────────────────────────
# Telegram helpers
# ─────────────────────────────────────────────────────────────────────────────

import re

def sanitize_for_telegram(text: str) -> str:
    text = re.sub(r'</?(p|div|span|br|img|h\d)[^>]*>', '', text)
    text = re.sub(r'<(?!/?(b|i|u|s|code|pre|a)\b)[^>]+>', '', text)
    return text.strip()


async def _telegram_post(endpoint: str, payload: dict):
    """Generic Telegram API caller."""
    url = f"{TELEGRAM_API_URL}/{endpoint}"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                logger.warning("Telegram %s error %s: %s", endpoint, res.status_code, res.text)
            return res
        except Exception as e:
            logger.exception("❌ Telegram %s failed: %s", endpoint, e)


async def send_message(chat_id: int, text: str,
                       parse_mode: str = "HTML",
                       reply_markup: dict | None = None):
    """Send a plain text message, optionally with an inline or reply keyboard."""
    MAX_LEN = 4096
    if parse_mode == "HTML":
        text = sanitize_for_telegram(text)
    parts = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    for idx, part in enumerate(parts):
        payload: dict = {"chat_id": chat_id, "text": part, "parse_mode": parse_mode}
        # only attach keyboard to last chunk
        if reply_markup and idx == len(parts) - 1:
            payload["reply_markup"] = reply_markup
        await _telegram_post("sendMessage", payload)


async def answer_callback_query(callback_query_id: str, text: str = ""):
    await _telegram_post("answerCallbackQuery",
                         {"callback_query_id": callback_query_id, "text": text})


async def send_typing(chat_id: int):
    await _telegram_post("sendChatAction",
                         {"chat_id": chat_id, "action": "typing"})


# ── Keyboard builders ─────────────────────────────────────────────────────────

def build_inline_keyboard(options: list[str], prefix: str) -> dict:
    """Build an inline keyboard from a list of option strings."""
    rows = []
    # 2 per row
    for i in range(0, len(options), 2):
        row = []
        for opt in options[i:i+2]:
            row.append({"text": opt, "callback_data": f"{prefix}:{opt}"})
        rows.append(row)
    return {"inline_keyboard": rows}


def build_main_menu_keyboard() -> dict:
    """Persistent reply keyboard shown after onboarding."""
    return {
        "keyboard": [
            [{"text": "📊 My Stats"}, {"text": "👤 My Profile"}],
            [{"text": "🗑️ Clear History"}, {"text": "✏️ Edit Profile"}],
            [{"text": "❓ Help"}],
        ],
        "resize_keyboard": True,
        "persistent": True,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Onboarding flow
# ─────────────────────────────────────────────────────────────────────────────

async def start_onboarding(chat_id: int, first_name: str):
    """Send the very first onboarding message."""
    await set_onboarding_step(chat_id, "age")
    await send_message(
        chat_id,
        f"👋 Welcome <b>{first_name}</b>!\n\n"
        "Before we begin, I'd like to know a bit about you "
        "so I can give you better, personalised answers.\n\n"
        + ONBOARDING_QUESTIONS["age"],
        reply_markup=build_inline_keyboard(ONBOARDING_OPTIONS["age"], "ob_age"),
    )


async def handle_onboarding_callback(chat_id: int, step: str, value: str):
    """
    Save the user's answer for the current step and advance.
    Called when an inline button from the onboarding flow is tapped.
    """
    # Persist the answer
    if step == "age":
        await upsert_profile(chat_id, age=value)
        next_step = "gender"
    elif step == "gender":
        await upsert_profile(chat_id, gender=value)
        next_step = "occupation"
    elif step == "occupation":
        await upsert_profile(chat_id, occupation=value)
        next_step = "education"
    elif step == "education":
        await upsert_profile(chat_id, education=value)
        next_step = "done"
    else:
        return

    if next_step == "done":
        await complete_onboarding(chat_id)
        profile = await get_profile(chat_id)
        await send_message(
            chat_id,
            "✅ <b>Profile saved!</b>\n\n"
            f"📅 Age: {profile.get('age', '—')}\n"
            f"⚧ Gender: {profile.get('gender', '—')}\n"
            f"💼 Occupation: {profile.get('occupation', '—')}\n"
            f"🎓 Education: {profile.get('education', '—')}\n\n"
            "You're all set! Ask me anything 🚀",
            reply_markup=build_main_menu_keyboard(),
        )
    else:
        await set_onboarding_step(chat_id, next_step)
        await send_message(
            chat_id,
            f"✅ Got it!\n\n{ONBOARDING_QUESTIONS[next_step]}",
            reply_markup=build_inline_keyboard(
                ONBOARDING_OPTIONS[next_step], f"ob_{next_step}"
            ),
        )

# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

def is_rate_limited(chat_id: int) -> bool:
    now = time.time()
    hits = user_hits.setdefault(chat_id, [])
    user_hits[chat_id] = [t for t in hits if now - t < RATE_LIMIT_WINDOW]
    if len(user_hits[chat_id]) >= RATE_LIMIT_MESSAGES:
        return True
    user_hits[chat_id].append(now)
    return False

# ─────────────────────────────────────────────────────────────────────────────
# RAG call
# ─────────────────────────────────────────────────────────────────────────────

async def call_rag(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
            res = await client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"query": query, "top_k": 3}
            )
            if res.status_code == 200:
                data = res.json()
                logger.info("📚 RAG returned %d chunks", len(data.get("chunks", [])))
                return data.get("context", "")
            else:
                logger.warning("RAG service error %s", res.status_code)
                return ""
    except httpx.TimeoutException:
        logger.warning("⏰ RAG call timed out")
        return ""
    except Exception as e:
        logger.warning("❌ RAG call failed: %s", e)
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder  (now profile-aware)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(query: str, context: str, history: list[dict],
                 profile: dict | None = None,
                 past_questions: list[str] | None = None) -> str:
    """
    Multilingual RAG prompt injected with:
      - user profile (age, gender, occupation, education)
      - recent question history
    """

    # ── User profile block ───────────────────────────────────────────────────
    profile_text = ""
    if profile:
        lines = []
        if profile.get("age"):
            lines.append(f"- Age group   : {profile['age']}")
        if profile.get("gender"):
            lines.append(f"- Gender      : {profile['gender']}")
        if profile.get("occupation"):
            lines.append(f"- Occupation  : {profile['occupation']}")
        if profile.get("education"):
            lines.append(f"- Education   : {profile['education']}")
        if lines:
            profile_text = "USER PROFILE:\n" + "\n".join(lines)

    # ── Past questions block ─────────────────────────────────────────────────
    past_q_text = ""
    if past_questions:
        numbered = [f"  {i+1}. {q}" for i, q in enumerate(past_questions[-5:])]
        past_q_text = "USER'S RECENT QUESTIONS:\n" + "\n".join(numbered)

    # ── Conversation history ─────────────────────────────────────────────────
    history_text = ""
    if history:
        lines = []
        for turn in history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(lines)

    # ── RAG context ──────────────────────────────────────────────────────────
    context_text = context.strip() if context.strip() else "No relevant context found."

    # ── Assemble ─────────────────────────────────────────────────────────────
    prompt = f"""
            You are a precise and reliable multilingual AI assistant.
            Tailor your response appropriately to the user's background when relevant.

            STRICT RULES:
            - Answer ONLY using the provided CONTEXT.
            - Do NOT use outside knowledge.
            - If answer is not found, say:
            "I don't know based on the available information."
            - Respond in the SAME LANGUAGE as the user's question.
            - Do NOT translate unless necessary.
            - Keep answer clear and concise.
            - Use plain text (no HTML tags like <p>, <div>).

            ---------------------
            {profile_text}
            CONTEXT:
            {context_text}
            ---------------------

            CONVERSATION:
            {history_text if history_text else "None"}
            ---------------------

            USER QUESTION:
            {query}
            ---------------------

            FINAL ANSWER (same language as question):
            """
    return prompt.strip()

# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(prompt: str) -> tuple[str, float]:
    MAX_RETRIES = 3
    BASE_DELAY  = 2

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                res = await client.post(
                    f"{LLM_SERVICE_URL}/infer",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": 20480,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                )
            if res.status_code == 200:
                data = res.json()
                return data["answer"], data.get("latency_ms", 0.0)
            elif res.status_code == 503:
                wait_time = BASE_DELAY * (attempt + 1)
                logger.warning("⚠️ LLM overloaded (attempt %d/%d). Retrying in %ds…",
                               attempt + 1, MAX_RETRIES, wait_time)
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error("LLM error %s: %s", res.status_code, res.text)
                return "⚠️ AI service returned an error. Please try again.", 0.0
        except httpx.TimeoutException:
            wait_time = BASE_DELAY * (attempt + 1)
            logger.warning("⏰ LLM timeout (attempt %d/%d). Retrying in %ds…",
                           attempt + 1, MAX_RETRIES, wait_time)
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.exception("❌ LLM call failed: %s", e)
            break

    return (
        "⚠️ The AI is currently busy due to high demand. "
        "Please try again in a few seconds.",
        0.0
    )

# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline  (now passes profile + past questions to prompt)
# ─────────────────────────────────────────────────────────────────────────────

async def process_query(chat_id: int, query: str) -> str:
    dedup_key = f"{chat_id}::{query.strip().lower()}"
    if dedup_key in in_flight:
        logger.info("♻️  Dedup hit for chat %d", chat_id)
        return await in_flight[dedup_key]

    task = asyncio.ensure_future(_run_pipeline(chat_id, query))
    in_flight[dedup_key] = task
    try:
        return await task
    finally:
        in_flight.pop(dedup_key, None)


# ---------------------------------------------------------------------------
# PURE follow-up signal words — words that CANNOT anchor a topic on their own.
# A query is a follow-up if EVERY content word is in this set.
# Single-word topic nouns like "Fisherman", "PM-Kisan", "MGNREGA" will NOT
# be in this set, so they correctly survive as topic anchors.
# ---------------------------------------------------------------------------
_FOLLOWUP_WORDS = {
    # question / auxiliary verbs
    "what", "whats", "how", "why", "when", "who", "where", "is", "are",
    "was", "were", "can", "could", "does", "do", "did", "will", "would",
    "should", "shall",
    # filler / discourse
    "tell", "explain", "more", "give", "show", "list", "any", "and",
    "also", "about", "for", "its", "that", "this", "same", "which",
    "their", "those", "these", "okay", "ok", "yes", "no", "please",
    "me", "my", "i", "the", "a", "an", "of", "in", "on", "to", "with",
    # common follow-up intent words (no topic noun on their own)
    "other", "states", "state", "another", "else", "different",
    "next", "previous", "above", "below", "same", "similar",
    "age", "benefits", "eligibility", "criteria", "process", "details",
    "apply", "application", "documents", "required", "needed",
    "limit", "amount", "subsidy", "relief", "scheme", "schemes",
    "who", "whom", "what", "know", "want", "need",
}


def _content_words(text: str) -> list[str]:
    """Return lowercase, punctuation-stripped tokens from text."""
    import re
    return [re.sub(r"[^a-z0-9]", "", w) for w in text.lower().split()
            if re.sub(r"[^a-z0-9]", "", w)]


def _is_followup(query: str) -> bool:
    """
    True if the query carries NO standalone topic signal.

    Logic: strip punctuation, lowercase every token. If ALL content tokens
    are pure follow-up / function words the query is a follow-up.

    Examples:
        "Fisherman"          → NOT a follow-up  (topic noun not in set)
        "other states ?"     → follow-up         (other, states both in set)
        "age criteria"       → follow-up         (age, criteria both in set)
        "PM-Kisan scheme"    → NOT a follow-up   ("pmkisan" not in set)
        "how to apply"       → follow-up
    """
    tokens = _content_words(query)
    if not tokens:
        return True
    # If ANY token is NOT a pure follow-up word → the query has topic signal
    return all(t in _FOLLOWUP_WORDS for t in tokens)


def _find_topic_anchor(user_turns: list[str]) -> str:
    """
    Walk user turns OLDEST → NEWEST, return the first turn that has
    real topic signal (i.e. is not a follow-up).

    This prevents topic drift:
      Turn 1: "Fisherman"        ← has topic signal  → anchor
      Turn 2: "age criteria"     ← follow-up
      Turn 3: "other states ?"   ← follow-up
    Both Turns 2 & 3 anchor on Turn 1, not on each other.
    """
    for turn in user_turns:
        if not _is_followup(turn):
            return turn.strip()
    # Every prior turn was also a follow-up — use the oldest as best-effort
    return user_turns[0].strip() if user_turns else ""


def _build_rag_query(current_query: str,
                     history: list[dict],
                     profile: dict | None = None) -> str:
    """
    Build a self-contained RAG retrieval string.

    Steps
    ─────
    1. Collect prior user turns (oldest → newest).
    2. If current query is a follow-up, prepend the earliest topic-bearing
       turn as an anchor so the retriever gets a meaningful search string.
    3. Keep profile signals OUT of the retrieval query — they pollute keyword
       matching.  Profile is already injected into the LLM prompt separately.

    Resulting query examples
    ────────────────────────
    User: "Fisherman"          → "Fisherman"
    User: "age criteria"       → "Fisherman age criteria"
    User: "other states ?"     → "Fisherman other states"
    User: "application process"→ "Fisherman application process"
    """
    user_turns = [t["content"] for t in history if t["role"] == "user"]

    if _is_followup(current_query) and user_turns:
        anchor = _find_topic_anchor(user_turns)
        anchor_lower   = anchor.lower().strip()
        current_lower  = current_query.lower().strip()
        if anchor and not current_lower.startswith(anchor_lower):
            return f"{anchor} {current_query}".strip()

    return current_query.strip()


async def _run_pipeline(chat_id: int, query: str) -> str:
    async with semaphore:
        t_start = time.time()

        # 2. Fetch history + profile + past questions in parallel
        history, profile, past_questions = await asyncio.gather(
            get_history(chat_id, limit=MAX_MEMORY),
            get_profile(chat_id),
            get_user_questions(chat_id, limit=10),
        )

        # 3. Build a context-aware RAG query
        rag_query = _build_rag_query(query, history, profile=profile)
        logger.info("📚 RAG query: %r  (original: %r)", rag_query, query)
        rag_context = await call_rag(rag_query)

        # 4. Build prompt (with profile + past questions)
        prompt = build_prompt(query, rag_context, history,
                              profile=profile,
                              past_questions=past_questions)

        # 5. LLM inference
        logger.info("🤖 Calling LLM for chat %d", chat_id)
        answer, llm_latency = await call_llm(prompt)

        total_ms = (time.time() - t_start) * 1000

        # 6. Persist
        await asyncio.gather(
            save_message(chat_id, "user",      query),
            save_message(chat_id, "assistant", answer,
                         rag_context=rag_context, latency_ms=total_ms),
        )

        # 7. Cache — skip error replies and "no information" fallbacks
        NO_CACHE_PHRASES = (
            "i don't know based on the available information",
            "i do not know based on the available information",
            "don't know based on the available",
            "no relevant context",
        )
        answer_lower = answer.strip().lower()
        is_error     = answer.startswith("⚠️")
        is_no_info   = any(phrase in answer_lower for phrase in NO_CACHE_PHRASES)

        if not is_error and not is_no_info:
            await cache_set(query, answer)
        else:
            logger.info("🚫 Cache skipped for chat %d (uninformative answer)", chat_id)

        logger.info("✅ Pipeline done for chat %d in %.0f ms", chat_id, total_ms)
        return answer

# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_command(chat_id: int, command: str, user_info: dict):
    command = command.split("@")[0].lower().strip()
    name    = user_info.get("first_name") or "there"

    # ── /start ────────────────────────────────────────────────────────────────
    if command == "/start":
        user = await get_user(chat_id)
        if user and user.get("onboarding_done"):
            await send_message(
                chat_id,
                f"👋 Welcome back, <b>{name}</b>!\n\n"
                "Just ask me anything — I'm here to help.\n\n"
                "Use the menu below for options.",
                reply_markup=build_main_menu_keyboard(),
            )
        else:
            await start_onboarding(chat_id, name)

    # ── /clear ────────────────────────────────────────────────────────────────
    elif command == "/clear":
        await clear_history(chat_id)
        await send_message(chat_id, "🗑️ Your conversation history has been cleared.")

    # ── /profile ──────────────────────────────────────────────────────────────
    elif command == "/profile":
        await show_profile(chat_id)

    # ── /editprofile ──────────────────────────────────────────────────────────
    elif command in ("/editprofile", "/edit"):
        await reset_onboarding(chat_id)
        await start_onboarding(chat_id, name)

    # ── /deleteaccount ────────────────────────────────────────────────────────
    elif command in ("/deleteaccount", "/delete"):
        await send_message(
            chat_id,
            "⚠️ <b>Delete my account</b>\n\n"
            "This will erase ALL your data — history, profile, everything. "
            "This action cannot be undone.\n\n"
            "Are you sure?",
            reply_markup={
                "inline_keyboard": [[
                    {"text": "✅ Yes, delete everything", "callback_data": "menu:delete_confirm"},
                    {"text": "❌ Cancel", "callback_data": "menu:delete_cancel"},
                ]]
            },
        )

    # ── /stats ────────────────────────────────────────────────────────────────
    elif command == "/stats":
        await show_stats(chat_id)

    # ── /help ─────────────────────────────────────────────────────────────────
    elif command in ("/help", "/h", "/menu"):
        await send_message(
            chat_id,
            "🤖 <b>How to use me</b>\n\n"
            "Type your question and I'll search my knowledge base "
            "and generate a personalised answer.\n\n"
            "<b>Commands:</b>\n"
            "  /start         — Welcome / restart\n"
            "  /profile       — View your profile\n"
            "  /editprofile   — Update your profile\n"
            "  /clear         — Clear conversation history\n"
            "  /stats         — Your usage stats\n"
            "  /deleteaccount — Delete all your data\n"
            "  /help          — This message\n\n"
            f"⚠️ Max message length: {MAX_MESSAGE_LENGTH} characters\n"
            f"⚠️ Rate limit: {RATE_LIMIT_MESSAGES} messages per {RATE_LIMIT_WINDOW}s",
            reply_markup=build_main_menu_keyboard(),
        )

    else:
        await send_message(
            chat_id,
            f"❓ Unknown command: <code>{command}</code>\nSend /help for available commands."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI actions
# ─────────────────────────────────────────────────────────────────────────────

async def show_profile(chat_id: int):
    profile = await get_profile(chat_id)
    if not profile:
        await send_message(
            chat_id,
            "You don't have a profile yet. Send /start to set one up."
        )
        return
    await send_message(
        chat_id,
        "👤 <b>Your Profile</b>\n\n"
        f"📅 Age group : {profile.get('age') or '—'}\n"
        f"⚧ Gender    : {profile.get('gender') or '—'}\n"
        f"💼 Occupation: {profile.get('occupation') or '—'}\n"
        f"🎓 Education : {profile.get('education') or '—'}\n\n"
        "Use <b>✏️ Edit Profile</b> or /editprofile to update.",
    )


async def show_stats(chat_id: int):
    stats = await get_user_stats(chat_id)
    last_active = (
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(stats["last_active"]))
        if stats["last_active"] else "N/A"
    )
    await send_message(
        chat_id,
        f"📊 <b>Your Stats</b>\n\n"
        f"👤 Name         : {stats.get('first_name') or 'Unknown'}\n"
        f"💬 Total messages: {stats['message_count']}\n"
        f"❓ Questions asked: {stats['query_count']}\n"
        f"🕐 Last active  : {last_active}",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Callback query handler  (inline buttons)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_callback_query(callback_query: dict):
    cq_id    = callback_query["id"]
    chat_id  = callback_query["from"]["id"]
    data     = callback_query.get("data", "")
    name     = callback_query["from"].get("first_name", "there")

    await answer_callback_query(cq_id)   # always acknowledge immediately

    # ── Onboarding callbacks  ob_<step>:<value> ───────────────────────────────
    if data.startswith("ob_"):
        rest = data[3:]          # "age:25–34"
        step, _, value = rest.partition(":")
        user = await get_user(chat_id)
        current_step = user.get("onboarding_step", "age") if user else "age"

        # Guard: ignore taps on a step that's already been passed
        if current_step not in ONBOARDING_STEPS or \
           ONBOARDING_STEPS.index(current_step) != ONBOARDING_STEPS.index(step) \
           if step in ONBOARDING_STEPS else True:
            pass

        await handle_onboarding_callback(chat_id, step, value)
        return

    # ── Menu / action callbacks  menu:<action> ────────────────────────────────
    if data.startswith("menu:"):
        action = data[5:]

        if action == "clear_history":
            await clear_history(chat_id)
            await send_message(chat_id, "🗑️ Conversation history cleared.")

        elif action == "view_profile":
            await show_profile(chat_id)

        elif action == "edit_profile":
            await reset_onboarding(chat_id)
            await start_onboarding(chat_id, name)

        elif action == "view_stats":
            await show_stats(chat_id)

        elif action == "delete_confirm":
            await clear_history(chat_id)
            await delete_profile(chat_id)
            await send_message(
                chat_id,
                "🗑️ All your data has been deleted.\n\n"
                "Send /start to create a new profile anytime.",
            )

        elif action == "delete_cancel":
            await send_message(chat_id, "✅ Deletion cancelled. Your data is safe.")

# ─────────────────────────────────────────────────────────────────────────────
# Reply-keyboard text handler  (button labels → actions)
# ─────────────────────────────────────────────────────────────────────────────

MENU_BUTTON_MAP: dict[str, str] = {
    "📊 my stats":      "stats",
    "👤 my profile":    "profile",
    "🗑️ clear history": "clear",
    "✏️ edit profile":  "editprofile",
    "❓ help":          "help",
}

async def handle_menu_button(chat_id: int, text: str, user_info: dict) -> bool:
    """
    If the text matches a menu button label, handle it and return True.
    Otherwise return False (caller should treat it as a regular query).
    """
    key = text.strip().lower()
    action = MENU_BUTTON_MAP.get(key)
    if action is None:
        return False

    name = user_info.get("first_name", "there")

    if action == "stats":
        await show_stats(chat_id)
    elif action == "profile":
        await show_profile(chat_id)
    elif action == "clear":
        await clear_history(chat_id)
        await send_message(chat_id, "🗑️ Conversation history cleared.")
    elif action == "editprofile":
        await reset_onboarding(chat_id)
        await start_onboarding(chat_id, name)
    elif action == "help":
        await handle_command(chat_id, "/help", user_info)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Worker loop
# ─────────────────────────────────────────────────────────────────────────────

async def worker(worker_id: int):
    logger.info("🔧 Worker %d started", worker_id)
    while True:
        item = await task_queue.get()
        chat_id, query = item
        try:
            await send_typing(chat_id)
            answer = await process_query(chat_id, query)
            await send_message(chat_id, answer)
        except Exception as e:
            logger.exception("❌ Worker %d error: %s", worker_id, e)
            await send_message(chat_id, "⚠️ An unexpected error occurred. Please try again.")
        finally:
            task_queue.task_done()

# ─────────────────────────────────────────────────────────────────────────────
# Background tasks
# ─────────────────────────────────────────────────────────────────────────────

async def cache_purge_loop():
    while True:
        await asyncio.sleep(300)
        try:
            await cache_purge_expired()
        except Exception as e:
            logger.warning("Cache purge error: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "queue_size": task_queue.qsize(),
        "in_flight":  len(in_flight),
        "workers":    NUM_WORKERS,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Webhook endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # ── Callback query (inline button taps) ───────────────────────────────────
    if "callback_query" in data:
        asyncio.create_task(handle_callback_query(data["callback_query"]))
        return {"ok": True}

    # ── Regular / edited message ──────────────────────────────────────────────
    message   = data.get("message") or data.get("edited_message", {})
    if not message:
        return {"ok": True}

    chat      = message.get("chat", {})
    from_user = message.get("from", {})
    chat_id   = chat.get("id")
    text      = message.get("text", "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # ── Upsert user ───────────────────────────────────────────────────────────
    asyncio.create_task(upsert_user(
        chat_id,
        username   = from_user.get("username"),
        first_name = from_user.get("first_name"),
    ))

    # ── Command handling ──────────────────────────────────────────────────────
    if text.startswith("/"):
        asyncio.create_task(handle_command(chat_id, text, from_user))
        return {"ok": True}

    # ── Check if user has completed onboarding ─────────────────────────────────
    user = await get_user(chat_id)
    if not user:
        # brand-new user: upsert happened but might not be committed yet
        await asyncio.sleep(0.2)
        user = await get_user(chat_id)

    if user and not user.get("onboarding_done"):
        # User is mid-onboarding — guide them back
        step = user.get("onboarding_step", "age")
        if step in ONBOARDING_QUESTIONS:
            await send_message(
                chat_id,
                f"Please complete your profile first:\n\n{ONBOARDING_QUESTIONS[step]}",
                reply_markup=build_inline_keyboard(ONBOARDING_OPTIONS[step], f"ob_{step}"),
            )
        else:
            await start_onboarding(chat_id, from_user.get("first_name", "there"))
        return {"ok": True}

    # ── Menu button shortcuts ─────────────────────────────────────────────────
    handled = await handle_menu_button(chat_id, text, from_user)
    if handled:
        return {"ok": True}

    # ── Input validation ──────────────────────────────────────────────────────
    if len(text) > MAX_MESSAGE_LENGTH:
        await send_message(
            chat_id,
            f"❌ Message too long ({len(text)} chars). Maximum is {MAX_MESSAGE_LENGTH} characters."
        )
        return {"ok": True}

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if is_rate_limited(chat_id):
        await send_message(chat_id,
                           "⚠️ Too many requests. Please wait a moment before sending again.")
        return {"ok": True}

    # ── Acknowledge ───────────────────────────────────────────────────────────
    await send_message(chat_id, "⏳ Thinking…")

    # ── Queue check ───────────────────────────────────────────────────────────
    if task_queue.full():
        await send_message(chat_id,
                           "⚠️ Server is busy right now. Please try again in a few seconds.")
        return {"ok": True}

    # ── Enqueue ───────────────────────────────────────────────────────────────
    await task_queue.put((chat_id, text))
    logger.info("📥 Queued query from chat %d | queue size: %d",
                chat_id, task_queue.qsize())
    return {"ok": True}

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=BOT_PORT, reload=False)