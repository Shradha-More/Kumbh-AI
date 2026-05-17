"""
Session / conversation routes — chat history with named conversations.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.models.database import db
from api.models.schemas import UserStats
from api.services.auth import get_current_user, get_optional_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class LogRequest(BaseModel):
    query_text: str
    response_text: str
    language: str = "en"
    query_type: str = "text"
    conversation_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


# ── Conversations ─────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List all conversations for the user, newest first."""
    rows = await db.fetch_all(
        """SELECT c.id, c.title, c.created_at, c.updated_at,
                  (SELECT COUNT(*) FROM sessions s WHERE s.conversation_id = c.id) as message_count
           FROM conversations c WHERE c.user_id = ?
           ORDER BY c.updated_at DESC LIMIT 50""",
        (current_user["user_id"],),
    )
    return {"conversations": rows}


@router.post("/conversations")
async def create_conversation(current_user: dict = Depends(get_current_user)):
    """Create a new conversation."""
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, current_user["user_id"], "New Chat", now, now),
    )
    return {"id": conv_id, "title": "New Chat", "created_at": now}


@router.put("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str,
    req: RenameRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename a conversation."""
    await db.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
        (req.title, conv_id, current_user["user_id"]),
    )
    return {"id": conv_id, "title": req.title}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a conversation and its messages."""
    await db.execute(
        "DELETE FROM sessions WHERE conversation_id = ? AND user_id = ?",
        (conv_id, current_user["user_id"]),
    )
    await db.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, current_user["user_id"]),
    )
    return {"status": "deleted"}


# ── Messages within a conversation ────────────────────────────

@router.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get all messages in a conversation."""
    rows = await db.fetch_all(
        """SELECT id, query_text, response_text, language, query_type, created_at
           FROM sessions WHERE conversation_id = ? AND user_id = ?
           ORDER BY created_at ASC LIMIT 500""",
        (conv_id, current_user["user_id"]),
    )
    messages = []
    for r in rows:
        messages.append({"type": "user", "text": r["query_text"], "lang": r["language"], "time": r["created_at"]})
        messages.append({"type": "bot", "text": r["response_text"], "lang": r["language"], "time": r["created_at"]})
    return {"messages": messages, "conversation_id": conv_id}


# ── Log a message ─────────────────────────────────────────────

@router.post("/log")
async def log_session(
    req: LogRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Log a new message. Auto-creates conversation if needed. Auto-names from first query."""
    user_id = current_user["user_id"] if current_user else "guest"
    now = datetime.now(timezone.utc).isoformat()
    conv_id = req.conversation_id

    # Auto-create conversation if none provided
    if current_user and not conv_id:
        conv_id = str(uuid.uuid4())
        # Title from first query (truncated)
        title = (req.query_text or "New Chat")[:60]
        if len(req.query_text or "") > 60:
            title += "..."
        await db.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, user_id, title, now, now),
        )
    elif current_user and conv_id:
        # Update conversation timestamp
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, conv_id, user_id),
        )
        # Auto-name if still "New Chat"
        conv = await db.fetch_one(
            "SELECT title FROM conversations WHERE id = ?", (conv_id,)
        )
        if conv and conv["title"] == "New Chat":
            title = (req.query_text or "New Chat")[:60]
            if len(req.query_text or "") > 60:
                title += "..."
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conv_id),
            )

    session_id = str(uuid.uuid4())
    await db.execute(
        """INSERT INTO sessions (id, user_id, query_text, response_text, language, query_type, conversation_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, user_id, req.query_text, req.response_text, req.language, req.query_type, conv_id, now),
    )

    return {"id": session_id, "conversation_id": conv_id, "status": "logged"}


# ── Stats ─────────────────────────────────────────────────────

@router.get("/stats")
async def user_stats(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    total_row = await db.fetch_one("SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?", (user_id,))
    lang_rows = await db.fetch_all("SELECT DISTINCT language FROM sessions WHERE user_id = ? AND language IS NOT NULL", (user_id,))
    fav_row = await db.fetch_one("SELECT COUNT(*) as cnt FROM favorites WHERE user_id = ?", (user_id,))
    return UserStats(
        total_queries=total_row["cnt"] if total_row else 0,
        languages_used=[r["language"] for r in lang_rows],
        favorite_places=fav_row["cnt"] if fav_row else 0,
    )


# ── Legacy endpoints ──────────────────────────────────────────

@router.get("/history")
async def get_chat_history(current_user: dict = Depends(get_current_user)):
    """Legacy: get all messages across conversations."""
    rows = await db.fetch_all(
        """SELECT query_text, response_text, language, query_type, created_at
           FROM sessions WHERE user_id = ? ORDER BY created_at ASC LIMIT 200""",
        (current_user["user_id"],),
    )
    messages = []
    for r in rows:
        messages.append({"type": "user", "text": r["query_text"], "lang": r["language"], "time": r["created_at"]})
        messages.append({"type": "bot", "text": r["response_text"], "lang": r["language"], "time": r["created_at"]})
    return {"messages": messages}


@router.delete("/history")
async def clear_all_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    return {"status": "cleared"}
