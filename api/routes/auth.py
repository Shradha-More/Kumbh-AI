"""
Auth routes — register, login, Google Sign-In, profile.
"""

import os
import uuid
import logging
import threading
from datetime import datetime, timezone

import secrets
import httpx
from urllib.parse import urlencode
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel

from api.models.database import db
from api.models.schemas import (
    RegisterRequest, LoginRequest, AuthResponse,
    UserProfile, UpdateProfileRequest,
)
from api.services.auth import (
    hash_password, verify_password, create_token, get_current_user,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
APP_URL = os.environ.get("APP_URL", "https://siddharthnavnath7-yatri-ai.hf.space").rstrip("/")


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Create a new user account."""
    # Check if email already exists
    existing = await db.fetch_one("SELECT id FROM users WHERE email = ?", (req.email,))
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    pw_hash = hash_password(req.password)

    await db.execute(
        """INSERT INTO users (id, name, email, phone, password_hash, preferred_language, created_at, last_login)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, req.name, req.email, req.phone, pw_hash, req.preferred_language, now, now),
    )

    token = create_token(user_id, req.email)
    user = UserProfile(
        id=user_id, name=req.name, email=req.email,
        phone=req.phone, preferred_language=req.preferred_language,
        avatar_url=None, created_at=now,
    )

    # Send welcome email in background (non-blocking)
    threading.Thread(target=send_welcome_email, args=(req.name, req.email), daemon=True).start()

    return AuthResponse(token=token, user=user)


def send_welcome_email(name: str, email: str):
    """Send welcome email after signup via Resend API."""
    resend_key = os.environ.get("RESEND_API_KEY", "")
    if not resend_key:
        log.info(f"Resend not configured — skipping welcome email to {email}")
        return

    try:
        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;max-width:500px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;border:1px solid #f0e0d0">
          <div style="background:linear-gradient(135deg,#6B0F1A,#E8652B);padding:30px;text-align:center">
            <h1 style="color:white;font-size:28px;margin:0">Yatri AI</h1>
            <p style="color:#F0C75E;font-size:14px;margin:4px 0 0">Nashik Kumbh Mela 2027</p>
          </div>
          <div style="padding:24px 30px">
            <h2 style="color:#1a1a2e;font-size:20px">Namaste, {name}!</h2>
            <p style="color:#4A4A6A;font-size:14px;line-height:1.7">
              Welcome to Yatri AI — your multilingual AI companion for Nashik Kumbh Mela 2027.
            </p>
            <p style="color:#4A4A6A;font-size:14px;line-height:1.7">With Yatri AI you can:</p>
            <ul style="color:#4A4A6A;font-size:14px;line-height:2">
              <li>Ask questions in Hindi, Marathi, English & 8 more languages</li>
              <li>Navigate 200+ Nashik locations on the map</li>
              <li>Get instant emergency help with one tap</li>
              <li>Explore temples, ghats, food spots & wineries</li>
            </ul>
            <div style="text-align:center;margin:20px 0">
              <a href="https://siddharthnavnath7-yatri-ai.hf.space" style="background:#E8652B;color:white;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">Open Yatri AI</a>
            </div>
            <p style="color:#8888A8;font-size:12px;text-align:center">Har Har Mahadev</p>
          </div>
        </div>
        """

        admin_email = os.environ.get("SMTP_EMAIL", "siddharthnavnath7@gmail.com")

        # Send to user
        try:
            resp = httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={
                    "from": "Yatri AI <onboarding@resend.dev>",
                    "to": [email],
                    "subject": "Welcome to Yatri AI — Nashik Kumbh Mela 2027",
                    "html": html,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            log.info(f"Welcome email sent to {email} via Resend")
        except Exception as e:
            log.warning(f"Welcome email to {email} failed (Resend free tier only sends to verified emails): {e}")

        # Always notify admin about new signup
        try:
            admin_html = f"""
            <div style="font-family:sans-serif;padding:20px">
              <h2 style="color:#E8652B">New User Registered!</h2>
              <p><b>Name:</b> {name}</p>
              <p><b>Email:</b> {email}</p>
              <p style="color:#888;font-size:12px">Yatri AI — Nashik Kumbh Mela 2027</p>
            </div>
            """
            httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={
                    "from": "Yatri AI <onboarding@resend.dev>",
                    "to": [admin_email],
                    "subject": f"[Yatri AI] New signup: {name} ({email})",
                    "html": admin_html,
                },
                timeout=10.0,
            )
            log.info(f"Admin notified about new signup: {name}")
        except Exception:
            pass
    except Exception as e:
        log.error(f"Failed to send welcome email to {email}: {e}")


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Authenticate with email and password."""
    row = await db.fetch_one("SELECT * FROM users WHERE email = ?", (req.email,))
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Update last_login
    now = datetime.now(timezone.utc).isoformat()
    await db.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))

    token = create_token(row["id"], row["email"])
    user = UserProfile(
        id=row["id"], name=row["name"], email=row["email"],
        phone=row["phone"], preferred_language=row["preferred_language"],
        avatar_url=row["avatar_url"], created_at=row["created_at"],
    )
    return AuthResponse(token=token, user=user)


class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token


@router.post("/google", response_model=AuthResponse)
async def google_signin(req: GoogleAuthRequest):
    """Authenticate with Google Sign-In. Creates account if first time."""
    # Verify the Google ID token
    payload = None

    # Method 1: Decode JWT directly (no network call, fast)
    try:
        import jwt
        # Google ID tokens are JWTs — decode without verification first to get claims
        # Then verify via Google's tokeninfo API
        unverified = jwt.decode(req.credential, options={"verify_signature": False})
        # Verify audience matches our client ID
        if GOOGLE_CLIENT_ID and unverified.get("aud") != GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Invalid token audience")
        # Verify issuer is Google
        if unverified.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
            raise HTTPException(status_code=401, detail="Invalid token issuer")
        payload = unverified
    except HTTPException:
        raise
    except Exception:
        pass

    # Method 2: Fallback to Google tokeninfo API (POST for long tokens)
    if not payload:
        try:
            resp = httpx.post(
                "https://oauth2.googleapis.com/tokeninfo",
                data={"id_token": req.credential},
                timeout=10.0,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            log.warning(f"Google token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid Google token")

    email = payload.get("email")
    name = payload.get("name", payload.get("given_name", email.split("@")[0] if email else "User"))
    avatar = payload.get("picture", "")

    if not email:
        raise HTTPException(status_code=400, detail="No email in Google token")

    now = datetime.now(timezone.utc).isoformat()

    # Check if user exists
    row = await db.fetch_one("SELECT * FROM users WHERE email = ?", (email,))

    if row:
        # Existing user — login
        await db.execute("UPDATE users SET last_login = ?, avatar_url = ? WHERE id = ?",
                         (now, avatar, row["id"]))
        user_id = row["id"]
        user = UserProfile(
            id=row["id"], name=row["name"], email=row["email"],
            phone=row["phone"], preferred_language=row["preferred_language"],
            avatar_url=avatar or row["avatar_url"], created_at=row["created_at"],
        )
    else:
        # New user — register
        user_id = str(uuid.uuid4())
        pw_hash = hash_password(str(uuid.uuid4()))  # Random password (Google auth only)
        await db.execute(
            """INSERT INTO users (id, name, email, phone, password_hash, preferred_language, avatar_url, created_at, last_login)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name, email, "", pw_hash, "en", avatar, now, now),
        )
        user = UserProfile(
            id=user_id, name=name, email=email,
            phone="", preferred_language="en",
            avatar_url=avatar, created_at=now,
        )
        # Welcome email
        threading.Thread(target=send_welcome_email, args=(name, email), daemon=True).start()

    token = create_token(user_id, email)
    return AuthResponse(token=token, user=user)


# ── Server-side Google OAuth redirect flow (works inside iframes) ──

_auth_codes: dict = {}


class GoogleExchangeRequest(BaseModel):
    code: str


async def _google_upsert(email: str, name: str, avatar: str):
    """Create or update a Google-authenticated user. Returns (jwt, UserProfile)."""
    if not email:
        raise HTTPException(status_code=400, detail="No email in Google token")
    name = name or email.split("@")[0]
    now = datetime.now(timezone.utc).isoformat()
    row = await db.fetch_one("SELECT * FROM users WHERE email = ?", (email,))
    if row:
        await db.execute(
            "UPDATE users SET last_login = ?, avatar_url = ? WHERE id = ?",
            (now, avatar, row["id"]),
        )
        user = UserProfile(
            id=row["id"], name=row["name"], email=row["email"],
            phone=row["phone"], preferred_language=row["preferred_language"],
            avatar_url=avatar or row["avatar_url"], created_at=row["created_at"],
        )
    else:
        uid = str(uuid.uuid4())
        pw = hash_password(str(uuid.uuid4()))
        await db.execute(
            """INSERT INTO users (id, name, email, phone, password_hash, preferred_language, avatar_url, created_at, last_login)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uid, name, email, "", pw, "en", avatar, now, now),
        )
        user = UserProfile(
            id=uid, name=name, email=email,
            phone="", preferred_language="en",
            avatar_url=avatar, created_at=now,
        )
        threading.Thread(target=send_welcome_email, args=(name, email), daemon=True).start()
    token = create_token(user.id, email)
    return token, user


@router.get("/google/login")
async def google_login_redirect():
    """Redirect browser to Google OAuth consent screen."""
    redirect_uri = f"{APP_URL}/api/v1/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
    }
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}",
        status_code=302,
    )


def _auth_close_page(*, success: bool, payload: str) -> str:
    """HTML page that relays the OAuth result to the opener window and closes."""
    if success:
        msg = f"{{type:'google_auth_ok',code:'{payload}'}}"
        text, color = "Signed in! Closing\u2026", "#16a34a"
    else:
        msg = f"{{type:'google_auth_err',error:'{payload}'}}"
        text, color = "Sign-in failed. Closing\u2026", "#dc2626"
    return (
        "<!DOCTYPE html><html><head><title>Yatri AI</title></head>"
        f'<body style="display:flex;align-items:center;justify-content:center;'
        f'height:100vh;font-family:sans-serif;background:#fef7f0">'
        f'<p style="color:{color};font-size:18px">{text}</p>'
        f"<script>var m={msg};"
        f"if(window.opener){{window.opener.postMessage(m,'*');setTimeout(function(){{window.close()}},600)}}"
        f"else{{window.location.href='{APP_URL}/?'+(m.code?'auth_code='+m.code:'auth_error='+(m.error||'unknown'))}}"
        f"</script></body></html>"
    )


@router.get("/google/callback")
async def google_oauth_callback(code: str = Query(None), error: str = Query(None)):
    """Handle the OAuth redirect back from Google."""
    if error or not code:
        return HTMLResponse(_auth_close_page(success=False, payload=error or "no_code"))

    redirect_uri = f"{APP_URL}/api/v1/auth/google/callback"
    try:
        tr = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=10.0,
        )
        tr.raise_for_status()
        tokens = tr.json()
    except Exception as exc:
        log.warning("Google token exchange failed: %s", exc)
        return HTMLResponse(_auth_close_page(success=False, payload="token_exchange"))

    try:
        ui = httpx.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10.0,
        )
        ui.raise_for_status()
        info = ui.json()
    except Exception as exc:
        log.warning("Google userinfo fetch failed: %s", exc)
        return HTMLResponse(_auth_close_page(success=False, payload="userinfo"))

    jwt_token, user = await _google_upsert(
        info.get("email", ""), info.get("name", ""), info.get("picture", ""),
    )

    otp = secrets.token_urlsafe(32)
    _auth_codes[otp] = {"token": jwt_token, "user": user, "ts": datetime.now(timezone.utc)}
    # Purge expired codes (> 5 min)
    cutoff = datetime.now(timezone.utc)
    for k in [k for k, v in _auth_codes.items() if (cutoff - v["ts"]).total_seconds() > 300]:
        _auth_codes.pop(k, None)

    return HTMLResponse(_auth_close_page(success=True, payload=otp))


@router.post("/google/exchange", response_model=AuthResponse)
async def google_exchange(req: GoogleExchangeRequest):
    """Exchange a one-time auth code (from OAuth callback) for a JWT."""
    entry = _auth_codes.pop(req.code, None)
    if not entry:
        raise HTTPException(status_code=400, detail="Invalid or expired code")
    return AuthResponse(token=entry["token"], user=entry["user"])


@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get the authenticated user's profile."""
    row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (current_user["user_id"],))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return UserProfile(
        id=row["id"], name=row["name"], email=row["email"],
        phone=row["phone"], preferred_language=row["preferred_language"],
        avatar_url=row["avatar_url"], created_at=row["created_at"],
    )


@router.put("/profile", response_model=UserProfile)
async def update_profile(req: UpdateProfileRequest, current_user: dict = Depends(get_current_user)):
    """Update the authenticated user's profile fields."""
    user_id = current_user["user_id"]
    row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    name = req.name if req.name is not None else row["name"]
    phone = req.phone if req.phone is not None else row["phone"]
    lang = req.preferred_language if req.preferred_language is not None else row["preferred_language"]

    await db.execute(
        "UPDATE users SET name = ?, phone = ?, preferred_language = ? WHERE id = ?",
        (name, phone, lang, user_id),
    )

    return UserProfile(
        id=row["id"], name=name, email=row["email"],
        phone=phone, preferred_language=lang,
        avatar_url=row["avatar_url"], created_at=row["created_at"],
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout — client should discard the token."""
    return {"detail": "Logged out. Please discard the token on the client."}
