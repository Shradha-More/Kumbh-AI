"""RAG retrieval service using ChromaDB + multilingual-e5-large."""

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
# Use /data for HF Spaces persistent storage, fallback to local
import os
_hf_data = Path("/data/chroma_db") if os.path.isdir("/data") else None
CHROMA_DIR = _hf_data if _hf_data else ROOT / "vectordb" / "chroma_db"
EMERGENCY_PATH = ROOT / "data" / "emergency_responses.json"
COLLECTION_NAME = "kumbh_mela_2027"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class RAGService:
    def __init__(self):
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        try:
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"ChromaDB ready — {self.collection.count()} documents")
        except Exception as e:
            log.warning(f"ChromaDB failed: {e}. RAG will return empty results (Groq handles responses).")
            self.client = None
            self.collection = None

        # Load emergency data for fast hardcoded lookup
        self._emergency_data = {}
        if EMERGENCY_PATH.exists():
            with open(EMERGENCY_PATH) as f:
                self._emergency_data = json.load(f)

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(
            [f"query: {text}"], normalize_embeddings=True
        ).tolist()[0]

    def retrieve(
        self,
        query: str,
        language: str = "en",
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve top-k relevant chunks. Falls back to cross-lingual if sparse."""
        if self.collection is None:
            return []
        embedding = self._embed(query)

        where = {"language": language}
        if domain:
            where = {"$and": [{"language": language}, {"domain": domain}]}

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where if self.collection.count() > 0 else None,
        )

        docs = []
        if results["documents"] and results["documents"][0]:
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append({
                    "text": text,
                    "domain": meta.get("domain", "general"),
                    "source": meta.get("source", ""),
                    "language": meta.get("language", language),
                    "score": round(1 - dist, 4),
                })

        # Cross-lingual fallback: if fewer than 2 results, search English too
        if len(docs) < 2 and language != "en":
            fallback = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"language": "en"},
            )
            if fallback["documents"] and fallback["documents"][0]:
                for text, meta, dist in zip(
                    fallback["documents"][0],
                    fallback["metadatas"][0],
                    fallback["distances"][0],
                ):
                    docs.append({
                        "text": text,
                        "domain": meta.get("domain", "general"),
                        "source": meta.get("source", ""),
                        "language": "en",
                        "score": round(1 - dist, 4),
                    })

        # Deduplicate by text prefix
        seen = set()
        unique = []
        for d in docs:
            key = d["text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique[:top_k]

    def retrieve_emergency(self, query: str, language: str = "en") -> Optional[dict]:
        """Fast hardcoded emergency lookup — no vector search."""
        query_lower = query.lower()
        scenarios = self._emergency_data.get("emergency_scenarios", {})

        for scenario_key, scenario in scenarios.items():
            keywords = scenario.get("keywords", [])
            lang_keywords = scenario.get(f"keywords_{language}", [])
            all_keywords = keywords + lang_keywords

            if any(kw.lower() in query_lower for kw in all_keywords):
                responses = scenario.get("response", {})
                response_text = responses.get(language, responses.get("en", ""))
                helplines = self._emergency_data.get("helplines", {})
                return {
                    "type": scenario_key,
                    "response": response_text,
                    "helplines": helplines,
                }

        return None

    def get_all_helplines(self) -> dict:
        return self._emergency_data.get("helplines", {})

    def get_hospitals(self) -> list:
        return self._emergency_data.get("hospitals", [])

    def get_police_stations(self) -> list:
        return self._emergency_data.get("police_stations", [])

    def nearest_facility(self, lat: float, lon: float, ftype: str) -> Optional[dict]:
        """Return nearest facility of given type by Haversine distance."""
        import math

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        pool = []
        if ftype in ("hospital", "medical"):
            pool = self._emergency_data.get("hospitals", [])
        elif ftype == "police":
            pool = self._emergency_data.get("police_stations", [])

        best, best_dist = None, float("inf")
        for facility in pool:
            coords = facility.get("coordinates")
            if not coords:
                continue
            d = haversine(lat, lon, coords["lat"], coords["lon"])
            if d < best_dist:
                best_dist = d
                best = {**facility, "distance_m": round(d)}
        return best

    def doc_count(self) -> int:
        return self.collection.count() if self.collection else 0


# Singleton
_rag_instance: Optional[RAGService] = None


def get_rag(force_reload: bool = False) -> RAGService:
    global _rag_instance
    if _rag_instance is None or force_reload:
        _rag_instance = RAGService()
    return _rag_instance
