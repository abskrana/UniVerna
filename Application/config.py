import os
BACKEND        = os.getenv("BACKEND", "gemini").lower()          # "hf" | "gemini"

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API_URL   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Gemini-specific config
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")    # gemini-3.1-flash-lite-preview
GEMINI_THINK_LEVEL = os.getenv("GEMINI_THINK_LEVEL", "MINIMAL")  # MINIMAL | MEDIUM | HIGH

# ── Internal service ports ─────────────────────────────────────────────────────
BOT_PORT = int(os.getenv("BOT_PORT", "8000"))        # main webhook server
LLM_PORT = int(os.getenv("LLM_PORT", "8001"))        # LLM inference server
RAG_PORT = int(os.getenv("RAG_PORT", "8002"))        # RAG server (dummy now)

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", f"http://localhost:{LLM_PORT}")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", f"http://localhost:{RAG_PORT}")

# ── Queue / Concurrency ────────────────────────────────────────────────────────
MAX_QUEUE_SIZE       = int(os.getenv("MAX_QUEUE_SIZE",       "50"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "4"))
NUM_WORKERS          = int(os.getenv("NUM_WORKERS",          "4"))

# ── Memory / Cache ─────────────────────────────────────────────────────────────
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "10"))   # conversation turns kept per user
CACHE_TTL  = int(os.getenv("CACHE_TTL",  "300"))  # seconds

# ── Rate limiting ──────────────────────────────────────────────────────────────
RATE_LIMIT_MESSAGES = int(os.getenv("RATE_LIMIT_MESSAGES", "5"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   "10"))
MAX_MESSAGE_LENGTH  = int(os.getenv("MAX_MESSAGE_LENGTH",  "500"))

# ── Timeouts (seconds) ─────────────────────────────────────────────────────────
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "5"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

# ── SQLite database path ───────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "bot.db")

# ── Model settings (used by llm_server.py) ────────────────────────────────────
MODEL_ID          = os.getenv("MODEL_ID", "bharatgenai/Param2-17B-A2.4B-Thinking")
MODEL_MAX_TOKENS  = int(os.getenv("MODEL_MAX_TOKENS",  "512"))
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MODEL_TOP_P       = float(os.getenv("MODEL_TOP_P",       "0.9"))

# ── Config ─────────────────────────────────────────────────────────────────────
INFER_TIMEOUT  = float(os.getenv("INFER_TIMEOUT", "300"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))

