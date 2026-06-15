"""
DatabaseManager: Stores every visitor session and individual Q&A answers
in a local SQLite database (database/visitor_logs.db).

Schema
------
sessions
  id            TEXT PRIMARY KEY   (UUID)
  started_at    TEXT               (ISO-8601)
  ended_at      TEXT               (ISO-8601, nullable)
  detected_lang TEXT

answers
  id            INTEGER PRIMARY KEY AUTOINCREMENT
  session_id    TEXT               (FK → sessions.id)
  question_key  TEXT
  question_text TEXT
  answer_text   TEXT
  detected_lang TEXT
  recorded_at   TEXT               (ISO-8601)

events
  id            INTEGER PRIMARY KEY AUTOINCREMENT
  session_id    TEXT
  event_type    TEXT               (greeting / farewell / error)
  question_key  TEXT               (nullable)
  message       TEXT
  lang          TEXT
  recorded_at   TEXT
"""

import sqlite3
import uuid
import os
from datetime import datetime, timezone
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "visitor_logs.db")


class DatabaseManager:
    """
    Handles all SQLite operations for session and answer logging.

    Uses context managers for every operation so connections are always
    cleanly closed even on error.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()
        logger.info(f"DatabaseManager ready. DB path: {self.db_path}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            TEXT PRIMARY KEY,
                    started_at    TEXT NOT NULL,
                    ended_at      TEXT,
                    detected_lang TEXT DEFAULT 'en'
                );

                CREATE TABLE IF NOT EXISTS answers (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT NOT NULL,
                    question_key  TEXT NOT NULL,
                    question_text TEXT,
                    answer_text   TEXT,
                    detected_lang TEXT,
                    recorded_at   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    question_key TEXT,
                    message      TEXT,
                    lang         TEXT,
                    recorded_at  TEXT NOT NULL
                );
            """)
        logger.debug("Database schema verified.")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new session row and return its UUID."""
        session_id = str(uuid.uuid4())
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                (session_id, now),
            )
        logger.info(f"Session created: {session_id}")
        return session_id

    def close_session(self, session_id: str, detected_lang: str = "en") -> None:
        """Mark a session as ended and store the final detected language."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, detected_lang = ? WHERE id = ?",
                (now, detected_lang, session_id),
            )
        logger.info(f"Session closed: {session_id}")

    # ------------------------------------------------------------------
    # Answer logging
    # ------------------------------------------------------------------

    def log_answer(
        self,
        session_id: str,
        question_key: str,
        question_text: str,
        answer_text: str,
        detected_lang: str = "en",
    ) -> None:
        """Insert one Q&A row."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO answers
                  (session_id, question_key, question_text, answer_text, detected_lang, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, question_key, question_text, answer_text, detected_lang, now),
            )
        logger.debug(f"Answer logged [{question_key}]: '{answer_text}'")

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        session_id: str,
        event_type: str,
        question_key: str | None,
        message: str,
        lang: str = "en",
    ) -> None:
        """Insert a greeting, farewell, or error event."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events
                  (session_id, event_type, question_key, message, lang, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, event_type, question_key, message, lang, now),
            )
        logger.debug(f"Event logged [{event_type}]: '{message}'")

    # ------------------------------------------------------------------
    # Query helpers (optional, for reporting/debugging)
    # ------------------------------------------------------------------

    def get_session_summary(self, session_id: str) -> dict:
        """Return a summary dict for a completed session."""
        with self._connect() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            answers = conn.execute(
                "SELECT question_key, answer_text, detected_lang FROM answers WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {
            "session": dict(session) if session else {},
            "answers": [dict(a) for a in answers],
        }

    def get_all_sessions(self) -> list[dict]:
        """Return a list of all sessions (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()