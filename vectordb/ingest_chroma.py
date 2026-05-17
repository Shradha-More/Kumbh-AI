"""
ChromaDB ingestion script for Nashik Kumbh Mela 2027 knowledge base.
Embeds all knowledge base documents and seed JSON data into ChromaDB.
Run after the pipeline is complete.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge_base" / "final"
SEED_DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "vectordb" / "chroma_db"
SYNTHETIC_QA_DIR = ROOT / "data" / "synthetic_qa"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "kumbh_mela_2027"
BATCH_SIZE = 64

LANG_CODES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi",
    "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam"
}


class KumbhVectorDB:
    def __init__(self):
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        log.info(f"ChromaDB collection '{COLLECTION_NAME}' ready. "
                 f"Current docs: {self.collection.count()}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with multilingual-e5-large (requires 'query: ' prefix for queries)."""
        prefixed = [f"passage: {t}" for t in texts]
        return self.embedder.encode(prefixed, normalize_embeddings=True).tolist()

    def add_documents(self, docs: list[dict]):
        """Add a batch of documents to ChromaDB."""
        ids, texts, metadatas, embeddings = [], [], [], []

        seen_ids = set()
        for doc in docs:
            text = doc.get("text", "").strip()
            if not text or len(text) < 30:
                continue
            doc_id = doc.get("id", str(uuid.uuid4()))
            # Ensure unique IDs within this batch
            while doc_id in seen_ids:
                doc_id = str(uuid.uuid4())
            seen_ids.add(doc_id)
            ids.append(doc_id)
            texts.append(text)
            metadatas.append({
                "language": doc.get("language", "en"),
                "domain": doc.get("domain", "general"),
                "source": doc.get("source", "unknown"),
                "source_url": doc.get("source_url", ""),
            })

        if not ids:
            return 0

        for i in range(0, len(ids), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_embs = self.embed_texts(batch_texts)
            self.collection.add(
                ids=ids[i:i+BATCH_SIZE],
                documents=batch_texts,
                embeddings=batch_embs,
                metadatas=metadatas[i:i+BATCH_SIZE],
            )
        return len(ids)

    def ingest_knowledge_base_files(self):
        """Ingest processed knowledge base chunks from the pipeline output."""
        log.info("Ingesting knowledge base chunks...")
        total = 0

        if not KNOWLEDGE_DIR.exists():
            log.warning(f"Knowledge base dir not found: {KNOWLEDGE_DIR}. "
                        "Run pipeline first. Skipping.")
            return

        files = list(KNOWLEDGE_DIR.glob("**/*.json"))
        log.info(f"Found {len(files)} chunk files")

        docs_batch = []
        for f in tqdm(files, desc="KB chunks"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                # Support both single doc and list
                items = data if isinstance(data, list) else [data]
                docs_batch.extend(items)
                if len(docs_batch) >= BATCH_SIZE * 4:
                    total += self.add_documents(docs_batch)
                    docs_batch = []
            except Exception as e:
                log.error(f"Failed to read {f}: {e}")

        if docs_batch:
            total += self.add_documents(docs_batch)
        log.info(f"  -> Ingested {total} KB documents")

    def ingest_seed_json(self, file_path: Path, domain: str):
        """Flatten a seed JSON file into text chunks and ingest."""
        log.info(f"Ingesting seed data: {file_path.name} (domain={domain})")

        with open(file_path) as f:
            data = json.load(f)

        docs = []

        def flatten_object(obj, path="", parent_lang="en"):
            """Recursively extract text strings from nested JSON."""
            if isinstance(obj, str) and len(obj) > 40:
                # Detect language from key suffix
                lang = "en"
                for code in LANG_CODES:
                    if path.endswith(f"_{code}") or path.endswith(f"description_{code}") \
                            or path.endswith(f"name_{code}"):
                        lang = code
                        break
                docs.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path.name}:{path}:{obj[:30]}")),
                    "text": obj,
                    "language": lang,
                    "domain": domain,
                    "source": file_path.name,
                    "source_url": "",
                })
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    flatten_object(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    flatten_object(item, f"{path}[{i}]")

        flatten_object(data)

        # Also create full-place documents for rich context
        if "places" in data:
            for place in data["places"]:
                for lang_code in LANG_CODES:
                    desc_key = f"description_{lang_code}"
                    if desc_key not in place and "description_en" not in place:
                        continue
                    desc = place.get(desc_key, place.get("description_en", ""))
                    name = place.get(f"name_{lang_code}", place.get("name", ""))
                    reach = place.get(f"how_to_reach_{lang_code}", place.get("how_to_reach_en", ""))
                    tips = place.get("tips_en", "")

                    full_text = f"{name}\n{desc}"
                    if reach:
                        full_text += f"\nHow to reach: {reach}"
                    if tips:
                        full_text += f"\nTips: {tips}"

                    docs.append({
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"place:{place.get('id','?')}:{lang_code}")),
                        "text": full_text.strip(),
                        "language": lang_code if desc_key in place else "en",
                        "domain": domain,
                        "source": file_path.name,
                        "source_url": "",
                    })

        added = self.add_documents(docs)
        log.info(f"  -> Added {added} documents from {file_path.name}")

    def ingest_emergency_data(self):
        """Ingest emergency responses as high-priority documents."""
        log.info("Ingesting emergency response data...")
        path = SEED_DATA_DIR / "emergency_responses.json"
        if not path.exists():
            log.warning("emergency_responses.json not found")
            return

        with open(path) as f:
            data = json.load(f)

        docs = []
        for scenario_key, scenario in data.get("emergency_scenarios", {}).items():
            responses = scenario.get("response", {})
            for lang, text in responses.items():
                if text and len(text) > 30:
                    docs.append({
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"emergency:{scenario_key}:{lang}")),
                        "text": text,
                        "language": lang,
                        "domain": "emergency",
                        "source": "emergency_responses.json",
                        "source_url": "",
                    })

        # Add helpline numbers as searchable doc
        helplines = data.get("helplines", {})
        helpline_texts = {
            "en": f"Emergency helpline numbers Nashik Kumbh 2027: Police 100, Ambulance 108, Fire 101, "
                  f"Missing Persons 1800-222-2027, Kumbh Mela Helpline 1800-120-2027, "
                  f"Women Helpline 1091, Child Helpline 1098, Tourist Helpline 1363.",
            "hi": f"आपातकालीन हेल्पलाइन नंबर नाशिक कुंभ 2027: पुलिस 100, एम्बुलेंस 108, अग्निशमन 101, "
                  f"लापता व्यक्ति 1800-222-2027, कुंभ मेला हेल्पलाइन 1800-120-2027, "
                  f"महिला हेल्पलाइन 1091, बाल हेल्पलाइन 1098।",
            "mr": f"आपत्कालीन हेल्पलाइन नंबर नाशिक कुंभ 2027: पोलीस 100, रुग्णवाहिका 108, "
                  f"अग्निशमन 101, हरवलेल्या व्यक्ती 1800-222-2027, कुंभ हेल्पलाइन 1800-120-2027।",
            "gu": f"કટોકટી હેલ્પલાઇન નાશિક કુંભ 2027: પોલીસ 100, એમ્બ્યુલન્સ 108, "
                  f"અગ્નિ 101, ગુમ 1800-222-2027, કુંભ 1800-120-2027.",
            "ta": f"அவசர ஹெல்ப்லைன் நாசிக் கும்ப 2027: காவல் 100, ஆம்புலன்ஸ் 108, "
                  f"தீயணைப்பு 101, காணாமல் 1800-222-2027, கும்ப 1800-120-2027.",
            "te": f"అత్యవసర హెల్ప్‌లైన్ నాసిక్ కుంభ 2027: పోలీసు 100, అంబులెన్స్ 108, "
                  f"అగ్ని 101, తప్పిపోయిన 1800-222-2027, కుంభ 1800-120-2027.",
            "kn": f"ತುರ್ತು ಹೆಲ್ಪ್‌ಲೈನ್ ನಾಸಿಕ್ ಕುಂಭ 2027: ಪೊಲೀಸ್ 100, ಆಂಬ್ಯುಲೆನ್ಸ್ 108, "
                  f"ಅಗ್ನಿ 101, ಕಾಣೆ 1800-222-2027, ಕುಂಭ 1800-120-2027.",
            "ml": f"അടിയന്തര ഹെൽപ്‌ലൈൻ നാസിക് കുംഭ 2027: പൊലീസ് 100, ആംബുലൻസ് 108, "
                  f"തീ 101, കാണാതായ 1800-222-2027, കുംഭ 1800-120-2027.",
        }

        for lang, text in helpline_texts.items():
            docs.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"helplines:{lang}")),
                "text": text,
                "language": lang,
                "domain": "emergency",
                "source": "emergency_responses.json",
                "source_url": "",
            })

        added = self.add_documents(docs)
        log.info(f"  -> Added {added} emergency documents")

    def ingest_synthetic_qa(self):
        """Ingest synthetic QA pairs as documents (answers become retrievable passages)."""
        log.info("Ingesting synthetic QA answers as knowledge documents...")
        if not SYNTHETIC_QA_DIR.exists():
            log.warning(f"Synthetic QA dir not found: {SYNTHETIC_QA_DIR}. Skipping.")
            return

        total = 0
        for qa_file in SYNTHETIC_QA_DIR.glob("**/*.jsonl"):
            docs = []
            with open(qa_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        answer = item.get("output", "")
                        question = item.get("instruction", "")
                        lang = item.get("language", "en")
                        domain = item.get("domain", "general")
                        if len(answer) < 30:
                            continue
                        docs.append({
                            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"qa:{question[:40]}:{answer[:20]}")),
                            "text": f"Q: {question}\nA: {answer}",
                            "language": lang,
                            "domain": domain,
                            "source": qa_file.name,
                            "source_url": "",
                        })
                    except json.JSONDecodeError:
                        continue
            added = self.add_documents(docs)
            total += added
        log.info(f"  -> Added {total} QA documents")

    def print_stats(self):
        """Print collection statistics."""
        count = self.collection.count()
        log.info(f"\n{'='*50}")
        log.info(f"ChromaDB Collection: {COLLECTION_NAME}")
        log.info(f"Total documents: {count}")
        log.info(f"Storage: {CHROMA_DIR}")
        log.info(f"{'='*50}\n")

    def test_retrieval(self):
        """Run a few test queries to verify retrieval works."""
        test_queries = [
            ("When is the main Kumbh Mela bathing date in 2027?", "en"),
            ("रामकुंड कहां है और कैसे पहुंचें?", "hi"),
            ("त्र्यंबकेश्वर मंदिर कसे आहे?", "mr"),
            ("ઇમર્જન્સી નંબર શું છે?", "gu"),
            ("நாசிக்கில் என்ன பார்க்க வேண்டும்?", "ta"),
        ]
        log.info("\nRunning retrieval tests...")
        for query, lang in test_queries:
            query_emb = self.embedder.encode(
                [f"query: {query}"], normalize_embeddings=True
            ).tolist()
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=2,
                where={"language": lang} if lang != "en" else None,
            )
            log.info(f"\nQ [{lang}]: {query[:60]}")
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                log.info(f"  [{meta['domain']}|{meta['language']}] (score={1-dist:.3f}) {doc[:80]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB")
    parser.add_argument("--skip-kb", action="store_true", help="Skip pipeline KB files")
    parser.add_argument("--skip-seed", action="store_true", help="Skip seed JSON files")
    parser.add_argument("--skip-qa", action="store_true", help="Skip synthetic QA")
    parser.add_argument("--test", action="store_true", help="Run retrieval test after ingest")
    args = parser.parse_args()

    db = KumbhVectorDB()

    if not args.skip_seed:
        seed_files = [
            ("kumbh_2027_schedule.json", "schedule"),
            ("nashik_places.json", "places"),
            ("ghats_and_transport.json", "transport"),
        ]
        for filename, domain in seed_files:
            path = SEED_DATA_DIR / filename
            if path.exists():
                db.ingest_seed_json(path, domain)
            else:
                log.warning(f"Seed file not found: {path}")

        db.ingest_emergency_data()

    if not args.skip_kb:
        db.ingest_knowledge_base_files()

    if not args.skip_qa:
        db.ingest_synthetic_qa()

    db.print_stats()

    if args.test:
        db.test_retrieval()


if __name__ == "__main__":
    main()
