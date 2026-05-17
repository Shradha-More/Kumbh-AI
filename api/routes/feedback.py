"""Feedback route — sends user feedback to admin email via Resend API."""

import logging
import threading
import os
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from api.models.database import db

log = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])

ADMIN_EMAIL = os.environ.get("SMTP_EMAIL", "siddharthnavnath7@gmail.com")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")


class FeedbackRequest(BaseModel):
    name: str = "Anonymous"
    email: str = ""
    type: str = "feedback"  # feedback, bug, feature, support
    message: str
    rating: int = 0  # 1-5 stars, 0 = not rated


@router.post("")
async def submit_feedback(req: FeedbackRequest):
    """Submit feedback — saves to DB and emails admin."""
    now = datetime.now(timezone.utc).isoformat()

    # Save to DB
    import uuid
    feedback_id = str(uuid.uuid4())
    try:
        await db.execute(
            """INSERT INTO feedback (id, name, email, type, message, rating, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (feedback_id, req.name, req.email, req.type, req.message, req.rating, now),
        )
    except Exception as e:
        log.warning(f"Failed to save feedback to DB: {e}")

    # Email admin in background
    threading.Thread(
        target=send_feedback_email,
        args=(req.name, req.email, req.type, req.message, req.rating),
        daemon=True,
    ).start()

    return {"status": "received", "id": feedback_id, "message": "Thank you for your feedback!"}


def send_feedback_email(name, email, ftype, message, rating):
    if not RESEND_API_KEY:
        log.info(f"Resend not configured — feedback from {name} saved to DB only")
        return

    try:
        stars = "★" * rating + "☆" * (5 - rating) if rating > 0 else "Not rated"
        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;max-width:500px;margin:0 auto;background:#fff;border-radius:12px;border:1px solid #eee;padding:24px">
          <h2 style="color:#E8652B;margin:0 0 16px">New {ftype.title()}</h2>
          <table style="font-size:14px;color:#333;line-height:2">
            <tr><td style="color:#888;padding-right:12px">From:</td><td><b>{name}</b></td></tr>
            <tr><td style="color:#888">Email:</td><td>{email or 'Not provided'}</td></tr>
            <tr><td style="color:#888">Type:</td><td>{ftype}</td></tr>
            <tr><td style="color:#888">Rating:</td><td style="color:#D4A017;font-size:18px">{stars}</td></tr>
          </table>
          <div style="margin-top:16px;padding:16px;background:#FFF8F0;border-radius:8px;border-left:4px solid #E8652B">
            <p style="font-size:14px;color:#333;line-height:1.7;margin:0">{message}</p>
          </div>
          <p style="font-size:11px;color:#aaa;margin-top:16px">Sent from Yatri AI — Nashik Kumbh Mela 2027</p>
        </div>
        """

        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": "Yatri AI <onboarding@resend.dev>",
                "to": [ADMIN_EMAIL],
                "subject": f"[Yatri AI] {ftype.title()} from {name}",
                "html": html,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        log.info(f"Feedback email sent from {name} via Resend")
    except Exception as e:
        log.error(f"Failed to send feedback email: {e}")
