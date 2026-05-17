"""Pydantic schemas for all API routes."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


Language = Literal["en", "hi", "mr", "gu", "ta", "te", "kn", "ml", "auto"]


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    language: Optional[Language] = "auto"
    session_id: Optional[str] = None
    domain: Optional[str] = None   # schedule, places, emergency, transport, etc.


class SourceDoc(BaseModel):
    text: str
    domain: str
    source: str
    score: float


class QueryResponse(BaseModel):
    response: str
    language: str
    sources: List[SourceDoc] = []
    domain: str
    confidence: float
    session_id: str


# ── Voice ──────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    language: Language


class STTResponse(BaseModel):
    transcript: str
    language: str
    confidence: float


class VoiceInputResponse(BaseModel):
    audio_base64: str
    transcript: str
    response_text: str
    language: str
    duration_ms: int


# ── Emergency ─────────────────────────────────────────────────────────────────

class LocationInput(BaseModel):
    lat: float
    lon: float


class EmergencyRequest(BaseModel):
    query: str
    language: Language = "en"
    location: Optional[LocationInput] = None


class EmergencyContact(BaseModel):
    name: str
    number: str


class NearestFacility(BaseModel):
    name: str
    distance_m: float
    phone: str
    address: str
    coordinates: Optional[LocationInput] = None


class EmergencyResponse(BaseModel):
    type: str
    response: str
    contacts: List[EmergencyContact]
    nearest_facility: Optional[NearestFacility] = None


# ── Places ────────────────────────────────────────────────────────────────────

class Coordinates(BaseModel):
    lat: float
    lon: float


class Place(BaseModel):
    id: str
    name: str
    category: str
    subcategory: Optional[str] = None
    description: str
    coordinates: Optional[Coordinates] = None
    timings: Optional[str] = None
    entry_fee: Optional[str] = None
    how_to_reach: Optional[str] = None
    tips: Optional[str] = None
    distance_from_nashik: Optional[str] = None
    crowd_level_kumbh: Optional[str] = None


class ItineraryRequest(BaseModel):
    interests: List[str] = []
    language: Language = "en"
    days_available: int = Field(1, ge=1, le=10)


class ItineraryDay(BaseModel):
    day: int
    places: List[str]
    description: str


class ItineraryResponse(BaseModel):
    days: List[ItineraryDay]
    total_places: int
    language: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    uptime_seconds: float
    gpu_memory_used_gb: Optional[float] = None
    total_documents: int
    version: str = "1.0.0"


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=200)
    phone: Optional[str] = None
    password: str = Field(..., min_length=6, max_length=128)
    preferred_language: str = "hi"


class LoginRequest(BaseModel):
    email: str
    password: str


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    preferred_language: str = "hi"
    avatar_url: Optional[str] = None
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserProfile


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    preferred_language: Optional[str] = None


# ── Sessions ─────────────────────────────────────────────────────────────────

class SessionLog(BaseModel):
    id: str
    query_text: Optional[str] = None
    response_text: Optional[str] = None
    language: Optional[str] = None
    query_type: Optional[str] = None
    created_at: str


class UserStats(BaseModel):
    total_queries: int
    languages_used: List[str] = []
    favorite_places: int = 0
