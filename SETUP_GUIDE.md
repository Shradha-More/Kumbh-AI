# Yatri AI — Nashik Kumbh Mela 2027
### Multilingual Voice AI Assistant

---

## What is this?

A fully offline, multilingual AI assistant for Nashik Kumbh Mela 2027 pilgrims and tourists. Speak in Hindi, Marathi, English (or 5 more Indian languages) and get instant answers about schedules, ghats, transport, food, wineries, and emergencies — with voice output.

**Live demo:** Run the server and open `http://localhost:8000`

---

## Architecture

```
Browser / Mobile PWA (static/index.html)
         |
         v  REST API + WebSocket
FastAPI Backend (api/)
    ├── Whisper small         ← Voice → Text (auto-detects language)
    ├── ChromaDB + E5-Large   ← RAG retrieval (4500+ docs)
    ├── Qwen2.5-3B (fine-tuned GGUF) ← Answer generation
    ├── gTTS                  ← Text → Voice
    └── SQLite                ← Users, sessions, conversations
```

**Supported Languages:** Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, English

---

## Quick Start (5 minutes)

### Prerequisites
- Python 3.11+ 
- 8GB+ RAM (16GB recommended)
- [Ollama](https://ollama.com) installed
- ffmpeg installed (`sudo apt install ffmpeg`)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Kumbh.git
cd Kumbh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

> **ChromaDB is pre-built** — `vectordb/chroma_db/` (41MB, 4500+ embedded docs) ready to use.
> 
> **Download the fine-tuned model (one time):**
> ```bash
> mkdir -p models
> huggingface-cli download siddharthnavnath7/Kumbh kumbh_model_q4_k_m.gguf --local-dir models
> ```
> Or manually from: https://huggingface.co/siddharthnavnath7/Kumbh
>
> The server auto-detects `models/kumbh_model_q4_k_m.gguf` — no Ollama needed.
> If you skip this step, install Ollama and run `ollama pull qwen2.5:1.5b` as fallback.

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Open the App

Open `http://localhost:8000` in your browser. That's it.

On mobile: open the same URL and tap "Add to Home Screen" for PWA install.

---

## If You Have the Fine-Tuned Model

If `models/kumbh_model_q4_k_m.gguf` exists (1.8GB), the server **automatically uses it** instead of Ollama — faster and more accurate. No extra setup needed.

**To get the GGUF model:**
- Download from releases / shared drive, OR
- Train it yourself (see [Training](#training-optional) below)

Place it at:
```
models/kumbh_model_q4_k_m.gguf
```

The server auto-detects on startup:
```
GGUF model loaded: kumbh_model_q4_k_m.gguf    ← uses llama-cpp
# OR if no GGUF:
Using Ollama backend: qwen2.5:1.5b             ← uses Ollama
```

---

## What's Included (No Setup Needed)

These are already in the repo — you do NOT need to regenerate them:

| Data | Count | Location |
|------|-------|----------|
| Nashik places (temples, ghats, food, hotels, transport) | 178 places with coordinates | `static/places_geo.json` |
| Knowledge base articles | 9 JSON files | `data/*.json` |
| Cleaned + chunked + deduplicated KB | 4,636 chunks | `knowledge_base/` |
| QA training pairs | 3,366 pairs | `data/synthetic_qa/` |
| Emergency responses | 6 scenarios, 7 helplines | Hardcoded in `api/routes/` |
| Map geodata | 178 markers with lat/lon | `static/places_geo.json` |

**What you DO need to build:** ChromaDB (step 3 above) — it embeds the chunks into vectors.

---

## Project Structure

```
Kumbh/
├── api/                    FastAPI backend
│   ├── main.py             App entry, lifespan, CORS, routes
│   ├── routes/
│   │   ├── query.py        Text query (with emergency bypass)
│   │   ├── voice.py        Voice input/output, STT, TTS
│   │   ├── emergency.py    Hardcoded emergency responses
│   │   ├── places.py       Places API (loads all data sources)
│   │   ├── auth.py         Register, login, profile (JWT)
│   │   └── sessions.py     Chat history, conversations, stats
│   ├── services/
│   │   ├── asr.py          Whisper speech-to-text
│   │   ├── llm.py          GGUF / Ollama LLM inference
│   │   ├── rag.py          ChromaDB retrieval
│   │   ├── tts.py          gTTS text-to-speech
│   │   └── auth.py         JWT + password hashing
│   └── models/
│       ├── database.py     SQLite async (users, sessions, conversations)
│       └── schemas.py      Pydantic request/response models
│
├── static/                 Web frontend (PWA)
│   ├── index.html          Complete SPA — home, chat, explore, map, SOS, auth, profile
│   ├── places_geo.json     178 Nashik places with coordinates
│   ├── manifest.json       PWA manifest
│   ├── sw.js               Service worker (offline caching)
│   └── icon-*.png          PWA icons
│
├── data/                   Knowledge base source files
│   ├── kumbh_2027_schedule.json
│   ├── kumbh_2027_detailed.json      15 detailed Kumbh articles
│   ├── nashik_complete_places.json   59 places with full details
│   ├── nashik_culture_history.json   19 culture/history articles
│   ├── nashik_food_wine.json         Food, wine, restaurant guide
│   ├── nashik_routes_transport.json  Transport routes
│   ├── nashik_places.json            Original 12 places
│   ├── emergency_responses.json      Emergency data
│   ├── ghats_and_transport.json      Ghats + transport
│   └── synthetic_qa/                 3,366 QA training pairs
│
├── knowledge_base/         Pipeline output stages
│   ├── raw/                Flattened seed + crawled data
│   ├── cleaned/            After cleaning
│   ├── chunked/            After chunking (300-500 word chunks)
│   ├── deduplicated/       After MinHash deduplication
│   └── translated/         After translation to 8 languages
│
├── vectordb/
│   ├── ingest_chunks.py    Incremental ChromaDB ingestion
│   ├── ingest_chroma.py    Full ChromaDB rebuild (legacy)
│   └── chroma_db/          Persistent vector store (NOT in git)
│
├── pipeline/               Data processing pipeline
│   ├── ingest_all.py       Master: flatten → clean → chunk → dedup
│   ├── flatten_seed.py     Flatten multilingual JSONs
│   ├── clean.py            HTML strip, normalize, redact
│   ├── chunk.py            Split into 300-500 word chunks
│   ├── deduplicate.py      MD5 + MinHash dedup
│   └── run_pipeline.py     Sequential pipeline runner
│
├── generate/               QA pair generation
│   ├── qa_generator.py     Ollama-based QA (needs running Ollama)
│   ├── generate_from_kb.py Template-based QA (no LLM needed)
│   └── paraphrase.py       Augment with paraphrases
│
├── crawler/                Web scrapers
│   ├── run_all_crawlers.py Master crawler runner
│   └── spiders/
│       ├── wikipedia_spider.py   128 Wikipedia articles
│       ├── osm_places.py         OpenStreetMap data
│       ├── news_crawler.py       Indian news sites
│       └── indic_datasets.py     HuggingFace datasets
│
├── training/
│   └── train_qlora.py      QLoRA fine-tuning (Unsloth + HF PEFT)
│
├── translate/
│   └── batch_translate.py  IndicTrans2 EN→Indic translation
│
├── models/                 Model files (NOT in git)
│   └── kumbh_model_q4_k_m.gguf   Fine-tuned GGUF (1.8GB)
│
├── Yatri_AI_Training.ipynb Google Colab training notebook
├── requirements.txt        Python dependencies
├── Dockerfile              Container config
├── docker-compose.yml      Docker services
└── .gitignore
```

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/` | Web app (PWA) | No |
| `POST` | `/api/v1/query` | Text query → response (emergency bypass built-in) | No |
| `POST` | `/api/v1/voice/input` | Voice → Voice (full pipeline) | No |
| `POST` | `/api/v1/voice/stt` | Speech → Text only | No |
| `POST` | `/api/v1/voice/tts` | Text → Speech (MP3) | No |
| `POST` | `/api/v1/emergency` | Emergency response (instant, no LLM) | No |
| `GET` | `/api/v1/emergency/contacts` | All helpline numbers | No |
| `GET` | `/api/v1/places` | All places (178+) | No |
| `GET` | `/api/v1/places/{id}` | Place detail | No |
| `POST` | `/api/v1/auth/register` | Create account | No |
| `POST` | `/api/v1/auth/login` | Sign in → JWT token | No |
| `GET` | `/api/v1/auth/profile` | User profile | Yes |
| `GET` | `/api/v1/sessions/conversations` | Chat history list | Yes |
| `POST` | `/api/v1/sessions/log` | Save chat message | Optional |
| `GET` | `/api/v1/health` | Server status | No |
| `WS` | `/ws/chat` | WebSocket streaming | No |

---

## Features

### Voice Assistant
- Tap mic, speak in any language → auto-detects → responds in same language
- Emergency keywords bypass LLM for instant response (<100ms)
- Chat history saved locally + server-side (if signed in)
- Conversation management (new chat, rename, delete, switch)

### Map (OpenStreetMap + OSRM)
- 178 places tagged with category-colored markers
- Filter by: temples, ghats, tourist, wineries, food, hotels, transport, emergency, markets
- From → To routing with turn-by-turn directions (OSRM, free)
- Set location via GPS, search, or map tap
- Integrated with assistant — "how to reach X" shows route button

### Emergency SOS
- 6 scenarios: Medical, Missing Person, Stampede, Fire, Drowning, Lost Items
- 7 helplines with tap-to-call
- Hardcoded responses in Hindi, Marathi, English — no server needed
- Nearest hospital finder on map

### Explore
- Rich cards with images, ratings, timings, fees
- Category filters, expandable details
- "View on Map" and "Get Directions" per place

### Auth & Profile
- Register / Login with JWT
- Language preference (changes entire UI)
- Chat history synced across devices
- User stats (queries, languages used)

### PWA
- Installable on mobile (Add to Home Screen)
- Fullscreen app mode
- Offline caching via service worker

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama fallback model (only if no GGUF) |
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base/small/medium/large-v3) |
| `JWT_SECRET` | `nashik-kumbh-2027-secret-key` | Change in production! |

Example:
```bash
OLLAMA_MODEL=qwen2.5:3b WHISPER_MODEL=base python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Training (Optional)

Fine-tuning improves response quality but is NOT required — the app works with base Ollama models + RAG.

**The model included in this repo (`models/kumbh_model_q4_k_m.gguf`) was trained using this process.**
You do NOT need to retrain unless you want to improve it with more data.

### How It Was Trained (Google Colab, Free T4 GPU)

- **Base model:** Qwen2.5-3B-Instruct (fits T4's 15GB VRAM in 4-bit)
- **Method:** QLoRA (4-bit quantization, LoRA r=16, alpha=32)
- **Data:** 3,366 QA pairs from `data/synthetic_qa/all_languages_combined.jsonl`
- **Epochs:** 3, batch size 2, gradient accumulation 8
- **Training time:** ~20 minutes on free Colab T4
- **Exported:** Q4_K_M GGUF quantization (1.8GB)

### To Retrain (if you add more data)

1. Upload `Yatri_AI_Training.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type`)
3. Run all cells — it will ask you to upload `all_languages_combined.jsonl`
4. Download the output `kumbh_model_q4_k_m.gguf` (~1.8GB)
5. Replace `models/kumbh_model_q4_k_m.gguf` and restart server

### Training on Cloud GPU (RunPod/Lambda)

```bash
python3 training/train_qlora.py \
  --model qwen2.5-3b \
  --data data/synthetic_qa/all_languages_combined.jsonl \
  --max-steps 500
```

### Expanding the Dataset

```bash
# Generate more QA pairs from knowledge base (no GPU needed)
python3 generate/generate_from_kb.py --target 10000

# Generate with Ollama (slower, higher quality)
python3 generate/qa_generator.py --model qwen2.5:1.5b --max-pairs 5000

# Re-run pipeline to process new data
python3 pipeline/ingest_all.py

# Rebuild ChromaDB
python3 vectordb/ingest_chunks.py
```

---

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Or build manually
docker build -t yatri-ai .
docker run -p 8000:8000 -v ./models:/app/models -v ./vectordb/chroma_db:/app/vectordb/chroma_db yatri-ai
```

---

## Rebuilding Everything from Scratch

If you want to rebuild the entire pipeline (not needed if cloning with data):

```bash
# 1. Flatten all seed data
python3 pipeline/ingest_all.py

# 2. Build ChromaDB
python3 vectordb/ingest_chunks.py

# 3. Generate QA pairs
python3 generate/generate_from_kb.py --target 8000

# 4. (Optional) Run crawlers for more data
python3 crawler/run_all_crawlers.py

# 5. (Optional) Train model on Colab
# Upload Yatri_AI_Training.ipynb + data/synthetic_qa/all_languages_combined.jsonl
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Ollama error: timed out` | Model too large for RAM. Use `qwen2.5:1.5b` instead of 7b |
| `float16 not supported` | CPU-only machine. Server auto-detects and uses int8 |
| `No module 'TTS'` | Coqui TTS not installed. gTTS fallback works fine |
| `pydub/audioop error` | Python 3.13 removed audioop. Already fixed — gTTS returns MP3 directly |
| Whisper detects wrong language | Use `small` model instead of `base`: `WHISPER_MODEL=small` |
| ChromaDB segfault | Delete `vectordb/chroma_db/` and re-run `ingest_chunks.py` |
| Slow responses (>30s) | Use smaller model: `OLLAMA_MODEL=qwen2.5:1.5b` |
| Emergency query gives bad answer | Emergency keywords trigger hardcoded bypass — working as intended |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| LLM | Qwen2.5-3B fine-tuned, Q4_K_M GGUF via llama-cpp-python (Ollama fallback) |
| ASR (Speech-to-Text) | faster-whisper (Whisper small) |
| TTS (Text-to-Speech) | gTTS (Google TTS) |
| RAG | ChromaDB + intfloat/multilingual-e5-large |
| Database | SQLite (aiosqlite) |
| Auth | JWT (PyJWT) + sha256_crypt |
| Frontend | Vanilla HTML/CSS/JS (PWA) |
| Maps | Leaflet.js + OpenStreetMap + OSRM routing |
| Icons | Remix Icons |
| Fonts | Poppins + Noto Sans Devanagari |
| Training | QLoRA via Unsloth / HuggingFace PEFT |

---

## License

MIT

---

*Built for Nashik Simhastha Kumbh Mela 2027*
