"""Clean ChromaDB ingestion from deduplicated chunks. Small batches, no dupes."""

import json, hashlib, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

import os
ROOT = Path(__file__).parent.parent
DEDUP_DIR = ROOT / "knowledge_base" / "deduplicated"
# Use /data for HF Spaces persistent storage
_hf_data = Path("/data/chroma_db") if os.path.isdir("/data") else None
CHROMA_PATH = _hf_data if _hf_data else ROOT / "vectordb" / "chroma_db"
BATCH = 24  # small to avoid OOM on 14GB RAM

def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model: intfloat/multilingual-e5-large")
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_or_create_collection("kumbh_mela_2027", metadata={"hnsw:space": "cosine"})
    log.info(f"ChromaDB collection ready. Current docs: {col.count()}")

    # Load all deduplicated chunks, dedupe by ID
    seen_ids = set()
    chunks = []
    for f in sorted(DEDUP_DIR.rglob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for item in items:
                text = item.get("text", "").strip()
                if len(text) < 30:
                    continue
                cid = hashlib.md5(text.encode()).hexdigest()[:16]  # content-based ID
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                chunks.append({
                    "id": cid, "text": text,
                    "language": item.get("language", "en"),
                    "domain": item.get("domain", "general"),
                })
        except Exception as e:
            log.warning(f"Skip {f.name}: {e}")

    log.info(f"Unique chunks to ingest: {len(chunks)}")

    added = 0
    total_batches = (len(chunks) + BATCH - 1) // BATCH
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metas = [{"language": c["language"], "domain": c["domain"]} for c in batch]

        prefixed = [f"passage: {t}" for t in texts]
        embs = embedder.encode(prefixed, normalize_embeddings=True).tolist()

        try:
            col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
        except Exception as e:
            log.warning(f"Batch {i//BATCH+1} error: {e}")
            # Try one by one
            for j in range(len(ids)):
                try:
                    col.add(ids=[ids[j]], documents=[texts[j]], embeddings=[embs[j]], metadatas=[metas[j]])
                    added += 1
                except:
                    pass
            continue

        added += len(ids)
        bn = i // BATCH + 1
        if bn % 10 == 0 or bn == total_batches:
            log.info(f"  Batch {bn}/{total_batches} — {added} docs ingested")

    log.info(f"Done. Total in ChromaDB: {col.count()}")

if __name__ == "__main__":
    main()
