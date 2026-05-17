"""
Async SQLite database layer using aiosqlite.
Tables: users, sessions, favorites, emergency_logs.
"""

import aiosqlite
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "yatri.db"


class Database:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path

    async def init_db(self):
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    password_hash TEXT NOT NULL,
                    preferred_language TEXT DEFAULT 'hi',
                    avatar_url TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    query_text TEXT,
                    response_text TEXT,
                    language TEXT,
                    query_type TEXT CHECK(query_type IN ('voice', 'text', 'emergency')),
                    conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS favorites (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    place_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS emergency_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    scenario TEXT,
                    language TEXT,
                    location_lat REAL,
                    location_lon REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    type TEXT,
                    message TEXT,
                    rating INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_conv ON sessions(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);
                CREATE INDEX IF NOT EXISTS idx_emergency_user ON emergency_logs(user_id);
            """)
            await conn.commit()
        log.info(f"Database initialized at {self.db_path}")

    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Run a write query (INSERT, UPDATE, DELETE)."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(sql, params)
            await conn.commit()

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dict."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows as a list of dicts."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


# Module-level singleton
db = Database()
