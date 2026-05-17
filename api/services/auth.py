"""
Authentication service — password hashing (passlib sha256) + JWT tokens (PyJWT).
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import jwt
from passlib.hash import sha256_crypt
from fastapi import Request, HTTPException

log = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "nashik-kumbh-2027-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7


def hash_password(password: str) -> str:
    """Hash a plaintext password with sha256_crypt."""
    return sha256_crypt.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a hash."""
    return sha256_crypt.verify(password, password_hash)


def create_token(user_id: str, email: str) -> str:
    """Create a JWT token with 7-day expiry."""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        log.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        log.warning(f"Invalid token: {e}")
        return None


async def get_current_user(request: Request) -> Dict:
    """
    FastAPI dependency — extracts user_id and email from Authorization Bearer token.
    Raises 401 if token is missing or invalid.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # strip "Bearer "
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"user_id": payload["sub"], "email": payload["email"]}


async def get_optional_user(request: Request) -> Optional[Dict]:
    """
    FastAPI dependency — returns user dict if authenticated, None if guest.
    Does NOT raise 401.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    payload = decode_token(token)
    if payload is None:
        return None
    return {"user_id": payload["sub"], "email": payload["email"]}
