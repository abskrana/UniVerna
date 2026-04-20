"""
database.py  —  Pure Python sqlite3 storage via aiosqlite.

Tables
──────
  users         : one row per Telegram user (extended with profile)
  user_profiles : age, gender, occupation, education level
  messages      : full conversation history per user
  query_cache   : response cache with TTL
"""

import time
import hashlib
import logging
import aiosqlite

from config import DB_PATH, MAX_MEMORY, CACHE_TTL

logger = logging.getLogger("database")

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER UNIQUE NOT NULL,
    username      TEXT,
    first_name    TEXT,
    created_at    REAL    NOT NULL,
    last_active   REAL    NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    is_blocked    INTEGER NOT NULL DEFAULT 0,
    onboarding_done INTEGER NOT NULL DEFAULT 0,
    onboarding_step TEXT   DEFAULT 'age'
);

CREATE TABLE IF NOT EXISTS user_profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER UNIQUE NOT NULL REFERENCES users(chat_id) ON DELETE CASCADE,
    age           TEXT,
    gender        TEXT,
    occupation    TEXT,
    education     TEXT,
    updated_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL REFERENCES users(chat_id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,   -- 'user' | 'assistant'
    content     TEXT    NOT NULL,
    rag_context TEXT,
    latency_ms  REAL,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_chat ON messages(chat_id, created_at);

CREATE TABLE IF NOT EXISTS query_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash  TEXT UNIQUE NOT NULL,
    query_text  TEXT NOT NULL,
    answer      TEXT NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_cache_hash    ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS ix_cache_expires ON query_cache(expires_at);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(SCHEMA)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.commit()
    logger.info("✅ SQLite database ready → %s", DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# User helpers
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_user(chat_id: int,
                      username: str | None = None,
                      first_name: str | None = None):
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        async with db.execute(
            "SELECT id FROM users WHERE chat_id = ?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            await db.execute(
                """INSERT INTO users (chat_id, username, first_name,
                                     created_at, last_active, message_count,
                                     onboarding_done, onboarding_step)
                   VALUES (?, ?, ?, ?, ?, 0, 0, 'age')""",
                (chat_id, username, first_name, now, now)
            )
        else:
            await db.execute(
                """UPDATE users
                   SET last_active   = ?,
                       message_count = message_count + 1,
                       username      = COALESCE(?, username),
                       first_name    = COALESCE(?, first_name)
                   WHERE chat_id = ?""",
                (now, username, first_name, chat_id)
            )
        await db.commit()


async def get_user(chat_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE chat_id = ?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_all_users() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users ORDER BY last_active DESC") as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def set_onboarding_step(chat_id: int, step: str):
    """Update which onboarding step the user is on."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET onboarding_step = ? WHERE chat_id = ?",
            (step, chat_id)
        )
        await db.commit()


async def complete_onboarding(chat_id: int):
    """Mark user as fully onboarded."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET onboarding_done = 1, onboarding_step = 'done' WHERE chat_id = ?",
            (chat_id,)
        )
        await db.commit()


async def reset_onboarding(chat_id: int):
    """Re-start onboarding (used by /profile edit)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET onboarding_done = 0, onboarding_step = 'age' WHERE chat_id = ?",
            (chat_id,)
        )
        await db.commit()

# ─────────────────────────────────────────────────────────────────────────────
# Profile helpers
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_profile(chat_id: int,
                         age: str | None = None,
                         gender: str | None = None,
                         occupation: str | None = None,
                         education: str | None = None):
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        async with db.execute(
            "SELECT id FROM user_profiles WHERE chat_id = ?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            await db.execute(
                """INSERT INTO user_profiles (chat_id, age, gender, occupation, education, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chat_id, age, gender, occupation, education, now)
            )
        else:
            await db.execute(
                """UPDATE user_profiles
                   SET age       = COALESCE(?, age),
                       gender    = COALESCE(?, gender),
                       occupation= COALESCE(?, occupation),
                       education = COALESCE(?, education),
                       updated_at= ?
                   WHERE chat_id = ?""",
                (age, gender, occupation, education, now, chat_id)
            )
        await db.commit()


async def get_profile(chat_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM user_profiles WHERE chat_id = ?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def delete_profile(chat_id: int):
    """Wipe profile data but keep the user row."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM user_profiles WHERE chat_id = ?", (chat_id,))
        await db.execute(
            "UPDATE users SET onboarding_done = 0, onboarding_step = 'age' WHERE chat_id = ?",
            (chat_id,)
        )
        await db.commit()

# ─────────────────────────────────────────────────────────────────────────────
# Message / History helpers
# ─────────────────────────────────────────────────────────────────────────────

async def save_message(chat_id: int, role: str, content: str,
                       rag_context: str | None = None,
                       latency_ms: float | None = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(
            """INSERT INTO messages
                   (chat_id, role, content, rag_context, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chat_id, role, content, rag_context, latency_ms, time.time())
        )
        await db.commit()


async def get_history(chat_id: int, limit: int = MAX_MEMORY) -> list[dict]:
    """Return last `limit` turns (oldest first) as [{role, content}]."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT role, content FROM messages
               WHERE chat_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (chat_id, limit)
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def get_user_questions(chat_id: int, limit: int = 10) -> list[str]:
    """Return the last N questions the user asked (user-role only)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT content FROM messages
               WHERE chat_id = ? AND role = 'user'
               ORDER BY created_at DESC
               LIMIT ?""",
            (chat_id, limit)
        ) as cur:
            rows = await cur.fetchall()
    return [r["content"] for r in reversed(rows)]


async def clear_history(chat_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()


async def get_user_stats(chat_id: int) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE chat_id = ?", (chat_id,)
        ) as cur:
            user = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?", (chat_id,)
        ) as cur:
            msg_count = (await cur.fetchone())["cnt"]
        async with db.execute(
            """SELECT COUNT(*) as cnt FROM messages
               WHERE chat_id = ? AND role = 'user'""", (chat_id,)
        ) as cur:
            query_count = (await cur.fetchone())["cnt"]

    return {
        "chat_id":       chat_id,
        "username":      user["username"] if user else None,
        "first_name":    user["first_name"] if user else None,
        "message_count": msg_count,
        "query_count":   query_count,
        "last_active":   user["last_active"] if user else None,
        "created_at":    user["created_at"] if user else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash(q: str) -> str:
    return hashlib.md5(q.strip().lower().encode()).hexdigest()


async def cache_get(query: str) -> str | None:
    key = _hash(query)
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, answer FROM query_cache
               WHERE query_hash = ? AND expires_at > ?""",
            (key, now)
        ) as cur:
            row = await cur.fetchone()
        if row:
            await db.execute(
                "UPDATE query_cache SET hits = hits + 1 WHERE id = ?",
                (row["id"],)
            )
            await db.commit()
            return row["answer"]
    return None


async def cache_set(query: str, answer: str, ttl: int = CACHE_TTL):
    key  = _hash(query)
    now  = time.time()
    exp  = now + ttl
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id FROM query_cache WHERE query_hash = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row:
            await db.execute(
                """UPDATE query_cache
                   SET answer = ?, expires_at = ?, hits = 1, created_at = ?
                   WHERE id = ?""",
                (answer, exp, now, row[0])
            )
        else:
            await db.execute(
                """INSERT INTO query_cache
                       (query_hash, query_text, answer, hits, created_at, expires_at)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (key, query, answer, now, exp)
            )
        await db.commit()


async def cache_purge_expired():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM query_cache WHERE expires_at < ?", (time.time(),)
        )
        await db.commit()
    logger.debug("🗑️  Expired cache entries purged")