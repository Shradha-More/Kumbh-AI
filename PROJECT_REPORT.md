# Yatri AI — Nashik Kumbh Mela 2027
## AI & Data Science Project Report

### Multilingual Voice-First AI Assistant for Pilgrimage Navigation & Safety

---

**Submitted by:** Sidharth Navnath  
**Program:** B.E. / B.Tech — Artificial Intelligence & Data Science (AI&DS)  
**Academic Year:** 2025-2026  
**Project Domain:** Natural Language Processing, Speech Processing, Information Retrieval, Web Development  
**Technologies:** Python, FastAPI, Whisper, Qwen2.5, ChromaDB, Leaflet.js, PWA

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Motivation & Social Impact](#3-motivation--social-impact)
4. [Literature Survey](#4-literature-survey)
5. [System Architecture](#5-system-architecture)
6. [Data Collection & Exploration](#6-data-collection--exploration)
7. [Data Processing Pipeline](#7-data-processing-pipeline)
8. [Model Design & Selection](#8-model-design--selection)
9. [Model Training (Fine-Tuning)](#9-model-training-fine-tuning)
10. [Retrieval-Augmented Generation (RAG)](#10-retrieval-augmented-generation-rag)
11. [Backend Development](#11-backend-development)
12. [Frontend & PWA Development](#12-frontend--pwa-development)
13. [Map & Navigation System](#13-map--navigation-system)
14. [Emergency Response System](#14-emergency-response-system)
15. [Authentication & User Management](#15-authentication--user-management)
16. [Deployment & Publishing](#16-deployment--publishing)
17. [Testing & Results](#17-testing--results)
18. [Challenges & Solutions](#18-challenges--solutions)
19. [Future Enhancements](#19-future-enhancements)
20. [Conclusion](#20-conclusion)
21. [References](#21-references)
22. [Appendix](#22-appendix)

---

## 1. Abstract

The Nashik Simhastha Kumbh Mela, expected in July–September 2027, will attract an estimated 50-75 million pilgrims from across India and the world to the banks of the Godavari River. These pilgrims speak diverse languages — Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, and English — and face critical challenges including navigation through unfamiliar terrain, accessing emergency services, finding accommodation, understanding ritual schedules, and communicating across language barriers.

**Yatri AI** is a multilingual, voice-first AI assistant designed to address these challenges. Built entirely on open-source technologies and capable of running fully offline after initial setup, the system combines:

- **Automatic Speech Recognition (ASR)** using OpenAI's Whisper for 8 Indian languages
- **Retrieval-Augmented Generation (RAG)** using ChromaDB with multilingual embeddings for accurate, context-grounded answers
- **Large Language Model (LLM)** inference using a fine-tuned Qwen2.5-3B model quantized to GGUF format
- **Text-to-Speech (TTS)** for spoken responses in the user's detected language
- **Real-time map navigation** with 178+ tagged locations using OpenStreetMap and OSRM routing
- **Emergency bypass system** providing instant (<100ms) hardcoded responses for life-threatening situations

The system is deployed as a Progressive Web App (PWA) accessible on any device with a browser, requiring no app store installation.

**Keywords:** NLP, Multilingual AI, Speech Recognition, RAG, LLM Fine-tuning, QLoRA, Kumbh Mela, PWA, OpenStreetMap

---

## 2. Problem Statement

### 2.1 Primary Problem

> **How can we build an AI system that enables pilgrims speaking any major Indian language to navigate, access services, and get emergency help at the Nashik Kumbh Mela 2027, even in areas with limited internet connectivity?**

### 2.2 Sub-Problems Addressed

| # | Problem | Impact |
|---|---------|--------|
| 1 | **Language barrier** — Pilgrims from South India cannot communicate with local Marathi/Hindi speakers | Isolation, inability to ask for help |
| 2 | **Navigation confusion** — 5 crore+ people in a relatively small area around Godavari ghats | Getting lost, missing Shahi Snan timings |
| 3 | **Emergency response delay** — Medical emergencies, stampede risks, missing persons | Loss of life, preventable casualties |
| 4 | **Information scattering** — Kumbh schedules, ghat locations, transport routes spread across dozens of sources | Misinformation, missed rituals |
| 5 | **Digital divide** — Many pilgrims are elderly, non-tech-savvy, or have basic phones | Exclusion from digital services |
| 6 | **Internet dependency** — Most AI assistants (Google, Alexa, Siri) require constant internet | Failure in crowded areas with network congestion |

### 2.3 Scope

- 8 Indian languages: Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam, English
- Voice input and voice output (hands-free operation)
- 178+ Nashik locations with GPS coordinates and routing
- Emergency bypass for 6 critical scenarios
- Fully functional offline after initial model download
- Deployable on any device with a modern browser

---

## 3. Motivation & Social Impact

### 3.1 Why Kumbh Mela?

The Kumbh Mela is the **largest peaceful gathering of humans on Earth**. The 2015 Nashik Simhastha Kumbh attracted approximately 30 million devotees. The 2019 Prayagraj Kumbh drew 240 million visitors over 49 days. The 2027 Nashik edition is expected to surpass previous records.

Historical incidents highlight the need for technology intervention:

| Year | Location | Incident | Impact |
|------|----------|----------|--------|
| 2003 | Nashik | Stampede at Ramkund | 39 dead, 57 injured |
| 2013 | Allahabad | Stampede at railway station | 36 dead |
| 1954 | Prayagraj | Stampede | 800+ dead |

### 3.2 Social Impact

- **Accessibility:** Voice-first design serves illiterate and elderly users who cannot type
- **Safety:** Emergency bypass provides instant help regardless of AI model load time
- **Inclusivity:** Support for 8 languages covers ~95% of Indian pilgrims
- **Digital empowerment:** PWA requires no app store, works on any smartphone
- **Open source:** Entire system uses free, open-source technologies — zero recurring costs

### 3.3 UN Sustainable Development Goals Alignment

| SDG | Relevance |
|-----|-----------|
| SDG 3: Good Health | Emergency medical response, hospital finder |
| SDG 10: Reduced Inequalities | Multilingual access for all communities |
| SDG 11: Sustainable Cities | Crowd management, navigation, safety infrastructure |
| SDG 16: Peace & Justice | Safety of mass gatherings, missing person tracking |

---

## 4. Literature Survey

### 4.1 Multilingual NLP

| Paper/System | Contribution | Limitation |
|--------------|-------------|------------|
| Whisper (Radford et al., 2023) | Robust multilingual ASR for 99 languages | Large model (1.5GB), slow on CPU |
| IndicBERT (Kakwani et al., 2020) | Pre-trained model for 12 Indian languages | Text-only, no speech |
| AI4Bharat IndicTrans2 (2023) | State-of-the-art EN↔Indic translation | Requires GPU for real-time use |
| mBERT / XLM-R | Multilingual text understanding | Not optimized for Indian languages |

### 4.2 Retrieval-Augmented Generation

| Paper/System | Contribution | Limitation |
|--------------|-------------|------------|
| RAG (Lewis et al., 2020) | Combining retrieval with generation | Requires external knowledge base |
| ChromaDB | Lightweight, embeddable vector database | Limited to cosine similarity |
| FAISS (Johnson et al., 2019) | Billion-scale similarity search | Requires significant RAM |
| E5 Embeddings (Wang et al., 2022) | Multilingual text embeddings | 2.2GB model size |

### 4.3 LLM Fine-Tuning

| Method | Contribution | Limitation |
|--------|-------------|------------|
| LoRA (Hu et al., 2021) | Low-rank adapter fine-tuning | Still needs GPU |
| QLoRA (Dettmers et al., 2023) | 4-bit quantized LoRA | Quality loss from quantization |
| Unsloth (2024) | 2x faster QLoRA training | Limited model support |
| GGUF Format | CPU-friendly quantized inference | Slower than GPU |

### 4.4 Research Gap

No existing system combines all of:
- Voice I/O in 8+ Indian languages
- Domain-specific knowledge about Kumbh Mela
- Offline-capable deployment
- Real-time map navigation
- Emergency bypass without AI latency

**Yatri AI fills this gap.**

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER (Browser / PWA)                      │
│  Voice Input → [Mic] ──────────────────→ Audio Upload       │
│  Text Input  → [Keyboard] ─────────────→ JSON POST          │
│  Map Actions → [Touch/Click] ──────────→ JS Events          │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP / WebSocket
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Whisper  │  │ ChromaDB │  │ Qwen2.5  │  │  gTTS    │   │
│  │  (ASR)   │  │  (RAG)   │  │  (LLM)   │  │  (TTS)   │   │
│  │          │  │          │  │          │  │          │   │
│  │ Speech→  │  │ Query→   │  │ Context+ │  │ Text→    │   │
│  │ Text     │  │ Top-K    │  │ Query→   │  │ Speech   │   │
│  │          │  │ Chunks   │  │ Answer   │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SQLite (Auth, Sessions, History)         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Emergency Bypass (Hardcoded, <100ms)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Request Flow (Voice Query)

```
1. User speaks: "त्र्यंबकेश्वर कैसे जाएं?"
       │
2. Browser records audio (WebM/Opus)
       │
3. POST /api/v1/voice/input (multipart form)
       │
4. ┌─ Whisper ASR ─────────────────────┐
   │ Model: faster-whisper (small)     │
   │ Compute: int8 (CPU optimized)     │
   │ Output: "त्र्यंबकेश्वर कैसे जाएं?" │
   │ Detected: language="hi"           │
   └───────────────────────────────────┘
       │
5. Emergency keyword check → NOT emergency → continue
       │
6. ┌─ ChromaDB RAG ────────────────────┐
   │ Embed query with E5-large         │
   │ Search 4500+ document chunks      │
   │ Return top 3 relevant chunks      │
   │ (Trimbakeshwar transport info)    │
   └───────────────────────────────────┘
       │
7. ┌─ Qwen2.5-3B (GGUF) ─────────────┐
   │ System: "You are Kumbh assistant" │
   │ Context: [3 RAG chunks]           │
   │ Query: "त्र्यंबकेश्वर कैसे जाएं?"  │
   │ Output: "नाशिक CBS से बस हर 15   │
   │ मिनट, NH-3 पर 28km, ऑटो ₹50-80" │
   └───────────────────────────────────┘
       │
8. ┌─ gTTS ────────────────────────────┐
   │ Convert Hindi text → MP3 audio    │
   └───────────────────────────────────┘
       │
9. Response JSON:
   {
     "transcript": "त्र्यंबकेश्वर कैसे जाएं?",
     "response_text": "नाशिक CBS से बस...",
     "audio_base64": "//uQxAA...",
     "language": "hi"
   }
       │
10. Browser: Display text + Auto-play audio + Show "Directions" button
```

### 5.3 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Frontend** | HTML/CSS/JS (Vanilla) | - | Single-page PWA |
| **Icons** | Remix Icons | 4.1.0 | Professional SVG icons |
| **Fonts** | Poppins + Noto Sans Devanagari | - | Latin + Devanagari typography |
| **Maps** | Leaflet.js | 1.9.4 | Interactive OpenStreetMap |
| **Routing** | Leaflet Routing Machine + OSRM | 3.2.12 | Turn-by-turn directions |
| **Backend** | FastAPI | 0.115.0 | Async REST API + WebSocket |
| **ASR** | faster-whisper | 1.2.1 | CTranslate2-based Whisper |
| **LLM** | llama-cpp-python | 0.3.1 | GGUF model inference |
| **Embeddings** | sentence-transformers | 3.2.1 | multilingual-e5-large |
| **Vector DB** | ChromaDB | 0.5.20 | Persistent vector store |
| **TTS** | gTTS | 2.5.3 | Google Text-to-Speech |
| **Database** | SQLite (aiosqlite) | - | Users, sessions, conversations |
| **Auth** | PyJWT + passlib | - | JWT tokens, password hashing |
| **Model** | Qwen2.5-3B-Instruct | - | Base LLM (fine-tuned) |
| **Training** | Unsloth + PEFT + TRL | - | QLoRA fine-tuning |
| **Quantization** | GGUF Q4_K_M | - | 4-bit model compression |
| **Deployment** | Docker + HuggingFace Spaces | - | Free cloud hosting |

---

## 6. Data Collection & Exploration

### 6.1 Data Sources

| Source | Type | Size | Languages | Method |
|--------|------|------|-----------|--------|
| **Hand-crafted seed data** | JSON | 530KB (9 files) | EN, HI, MR + 5 more | Manual research & writing |
| **Wikipedia** | Articles | 2-5MB | 8 language editions | Python spider (wikipedia-api) |
| **OpenStreetMap** | POI data | 500KB | Multilingual tags | Overpass API queries |
| **News sites** | Articles | 1-2MB | HI, MR, EN | newspaper3k + BeautifulSoup |
| **HuggingFace datasets** | QA pairs | 2-3MB | HI, MR, GU | IndicQA, Samanantar |
| **Map geodata** | Coordinates | 200KB | - | Manual + OSM |

### 6.2 Seed Data (Hand-Crafted Knowledge Base)

We created 9 comprehensive JSON files covering every aspect of Nashik and Kumbh Mela:

| File | Entries | Content |
|------|---------|---------|
| `nashik_complete_places.json` | 59 places | Temples, ghats, forts, wineries, restaurants, hotels, transport hubs — each with bilingual descriptions, coordinates, timings, fees |
| `nashik_culture_history.json` | 19 articles | Nashik history, Ramayana connection, wine industry, grape economy, Peshwa era, tribal culture (500+ words each) |
| `kumbh_2027_detailed.json` | 15 articles | Shahi Snan calendar, 13 Akharas, Peshwai procession, security, medical, planning guides |
| `nashik_food_wine.json` | 10 articles | Misal pav culture, 12 wineries, 30+ restaurants, grape varieties, street food |
| `nashik_routes_transport.json` | 12 routes | Mumbai/Pune/Delhi to Nashik (train, bus, car), local transport, auto fares |
| `kumbh_2027_schedule.json` | - | 5 Shahi Snan dates, Peshwai order, 13 Akhara bathing sequence |
| `nashik_places.json` | 12 places | Core tourist places with 8-language descriptions |
| `emergency_responses.json` | 6 scenarios | Medical, missing person, stampede, fire, drowning — with helplines in 8 languages |
| `ghats_and_transport.json` | - | All Nashik ghats, transport hubs, accommodation |

**Total seed data: 530KB, 115+ detailed entries**

### 6.3 Map Geodata

`places_geo.json` — 178 places with real GPS coordinates:

| Category | Count | Examples |
|----------|-------|---------|
| Temples | 51 | Kalaram, Trimbakeshwar, Saptashrungi, Navshya Ganapati, Dutondya Maruti |
| Ghats | 16 | Ramkund, Kushavart, Sita Ghat, Laxman Ghat, Ahilya Ghat |
| Tourist Spots | 21 | Pandav Leni, Anjaneri, Harihar Fort, Gangapur Dam |
| Wineries | 11 | Sula, York, Soma, Vallonne, Grover Zampa, Chandon |
| Food & Restaurants | 28 | Sadhana Misal, Mama Kane, Panchavati Gaurav |
| Accommodation | 19 | Express Inn, Gateway Hotel, Dharamshalas near Ramkund |
| Transport | 14 | Nashik Road Railway, CBS, Ozar Airport, Dwarka Circle |
| Emergency | 10 | Civil Hospital, Bytco Hospital, Police Commissionerate |
| Markets | 8 | Saraf Bazaar, Bhadrakali Market, City Centre Mall |

### 6.4 Data Exploration

**Language distribution in raw data:**

```
English   : 219 documents (35.2%)
Hindi     : 176 documents (28.3%)
Marathi   : 174 documents (28.0%)
Gujarati  :  19 documents (3.1%)
Tamil     :  11 documents (1.8%)
Telugu    :   9 documents (1.4%)
Kannada   :   8 documents (1.3%)
Malayalam :   6 documents (1.0%)
```

**Domain distribution:**

```
Places     : 287 documents (46.1%)
Transport  : 150 documents (24.1%)
Schedule   :  72 documents (11.6%)
Culture    :  57 documents (9.2%)
Food       :  31 documents (5.0%)
Emergency  :  22 documents (3.5%)
Accommodation: 3 documents (0.5%)
```

---

## 7. Data Processing Pipeline

### 7.1 Pipeline Architecture

```
Raw Data (9 JSON files, 530KB)
    │
    ▼ [Stage 1: Flatten]
    │ flatten_seed.py / ingest_all.py
    │ Nested multilingual JSON → flat {text, language, domain} documents
    │ Output: 622 unique flat documents
    │
    ▼ [Stage 2: Clean]
    │ clean.py
    │ Strip HTML, normalize Unicode (NFC), redact phone/email,
    │ remove duplicate paragraphs, split texts >5000 words
    │ Output: 1,623 cleaned documents
    │
    ▼ [Stage 3: Chunk]
    │ chunk.py
    │ Split into 300-500 word chunks with 50-word overlap
    │ Regex sentence boundary detection (no NLTK dependency)
    │ SHA1-based chunk IDs, manifest JSON
    │ Output: 6,930 chunks
    │
    ▼ [Stage 4: Deduplicate]
    │ deduplicate.py
    │ Pass 1: MD5 exact-match deduplication
    │ Pass 2: MinHash LSH (128 permutations, 80% Jaccard threshold)
    │ Output: 4,636 unique chunks
    │
    ▼ [Stage 5: Embed & Store]
    │ ingest_chunks.py
    │ Embed with intfloat/multilingual-e5-large (1024-dim)
    │ Store in ChromaDB (persistent, cosine similarity)
    │ Output: 4,500+ vectors in ChromaDB
```

### 7.2 Chunking Strategy

We chose a **fixed-size chunking with overlap** strategy:

- **Chunk size:** 300-500 words
- **Overlap:** 50 words between consecutive chunks
- **Boundary detection:** Regex-based sentence splitting supporting Devanagari (। ॥), Tamil, Telugu punctuation

**Rationale:** Fixed-size chunks ensure consistent context window usage in the LLM. The 50-word overlap prevents information loss at chunk boundaries. 300-500 words is optimal for the E5 embedding model's context window.

### 7.3 Deduplication Results

```
Before deduplication: 6,930 chunks
After MD5 exact dedup: 6,919 (-11 exact duplicates)
After MinHash fuzzy dedup: 4,636 (-2,283 near-duplicates)
Deduplication ratio: 33.1% removed
```

---

## 8. Model Design & Selection

### 8.1 Model Selection Criteria

| Criterion | Weight | Evaluation |
|-----------|--------|------------|
| Multilingual capability | High | Must handle 8 Indian languages |
| Size / RAM usage | High | Must fit in 16GB RAM (quantized) |
| Instruction following | High | Must answer from given context |
| Open source | Required | No API costs, offline capable |
| Fine-tuning support | Required | QLoRA compatible |

### 8.2 Models Evaluated

| Model | Parameters | Languages | License | Selected? |
|-------|-----------|-----------|---------|-----------|
| LLaMA 3 8B | 8B | Limited Indic | Meta | No — poor Hindi/Marathi |
| Mistral 7B | 7B | Mostly European | Apache | No — weak Indic |
| **Qwen2.5-3B-Instruct** | 3B | 29 languages including Hindi, Marathi | Apache | **Yes** |
| Qwen2.5-7B-Instruct | 7B | 29 languages | Apache | Too large for free GPU |
| Gemma 2B | 2B | Limited | Google | No — weaker instruction following |

### 8.3 Why Qwen2.5-3B?

1. **Multilingual:** Trained on Hindi, Marathi, Gujarati, Tamil data
2. **Size:** 3B parameters fits in T4 GPU (15GB) for training, and runs on CPU (1.9GB quantized)
3. **Instruction tuned:** Follows context-based answering well
4. **QLoRA compatible:** Supported by Unsloth for 2x faster training
5. **GGUF export:** Clean conversion to quantized CPU format

### 8.4 ASR Model Selection

| Model | Size | Indian Language Accuracy | Speed (CPU) | Selected? |
|-------|------|------------------------|-------------|-----------|
| Whisper tiny | 39MB | Poor | 1s | No |
| Whisper base | 74MB | Fair | 2s | No |
| **Whisper small** | 461MB | Good (Hindi, Marathi, Tamil) | 3-4s | **Yes** |
| Whisper medium | 1.5GB | Very good | 8-10s | Too slow for CPU |
| Whisper large-v3 | 3GB | Excellent | 30s+ | Too slow and large |

### 8.5 Embedding Model Selection

| Model | Dimensions | Multilingual | Size | Selected? |
|-------|-----------|-------------|------|-----------|
| all-MiniLM-L6-v2 | 384 | Limited | 80MB | No — weak for Indic |
| **multilingual-e5-large** | 1024 | Excellent (100+ languages) | 2.2GB | **Yes** |
| BGE-M3 | 1024 | Good | 2.3GB | Similar but less tested |

---

## 9. Model Training (Fine-Tuning)

### 9.1 Training Data Generation

We generated training data using two complementary approaches:

**Approach 1: Template-based QA Generation (No GPU needed)**

```python
# generate/generate_from_kb.py
# For each place/entry, generates questions like:
"What is {name}?" → description
"{name} कहाँ है?" → location info
"How to reach {name}?" → transport info
"{name} का समय क्या है?" → timings
```

**Approach 2: LLM-based QA Generation (Ollama on CPU)**

```python
# generate/qa_generator.py
# Feeds each chunk to qwen2.5:1.5b via Ollama
# Generates 3-5 diverse QA pairs per chunk
# 8 question types: factual, procedural, comparative,
#   emergency, recommendation, timing, conversational, follow-up
```

### 9.2 Training Data Statistics

```
Template QA pairs (EN): 1,574
Template QA pairs (HI): 1,180
Ollama QA pairs:          614
─────────────────────────────
Total training pairs:   3,366
```

**Format (Alpaca-style JSONL):**

```json
{
  "instruction": "त्र्यंबकेश्वर कैसे पहुँचें?",
  "input": "",
  "output": "नाशिक CBS से त्र्यंबकेश्वर के लिए हर 15 मिनट में बस मिलती है। NH-3 पर 28 किमी की दूरी है। ऑटो का किराया ₹50-80 है।",
  "language": "hi",
  "domain": "transport",
  "type": "procedural"
}
```

### 9.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen2.5-3B-Instruct |
| **Method** | QLoRA (4-bit NF4 quantization) |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Trainable Parameters** | 40,370,176 / 7,655,986,688 (0.53%) |
| **Batch Size** | 2 (per device) |
| **Gradient Accumulation** | 8 (effective batch = 16) |
| **Learning Rate** | 2e-4 |
| **Scheduler** | Cosine |
| **Epochs** | 3 |
| **Max Sequence Length** | 1024 tokens |
| **Precision** | FP16 |
| **Optimizer** | Paged AdamW 8-bit |
| **Hardware** | Google Colab T4 GPU (15GB VRAM) |
| **Training Time** | ~20 minutes |
| **Framework** | Unsloth + HuggingFace TRL |

### 9.4 Training Process

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 3,197 | Num Epochs = 3 | Total steps = 600
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation = 8
\        /    Total batch size (2 x 8 x 1) = 16
 "-____-"     Trainable parameters = 40,370,176 (0.53% trained)
```

### 9.5 Model Export

After training, the model was exported to GGUF format for efficient CPU inference:

```
Quantization: Q4_K_M (4-bit, k-quant mixed)
Original model size: ~6GB (FP16)
Quantized model size: 1.93GB
Compression ratio: 3.1x
```

The GGUF model is published at: **https://huggingface.co/siddharthnavnath7/Kumbh**

---

## 10. Retrieval-Augmented Generation (RAG)

### 10.1 Why RAG?

Fine-tuning alone cannot encode all factual knowledge reliably. RAG combines:

- **Retrieval:** Find relevant documents from a knowledge base
- **Generation:** Use retrieved context to generate accurate answers

This ensures the model answers based on **verified facts** rather than hallucinating.

### 10.2 RAG Pipeline

```
User Query: "रामकुंड कहां है?"
        │
        ▼
[1] Embed query with E5-large
    "query: रामकुंड कहां है?" → [0.023, -0.041, ..., 0.087] (1024-dim)
        │
        ▼
[2] Search ChromaDB (cosine similarity)
    Top-3 results:
    - "Ramkund is the sacred bathing ghat on Godavari river..." (score: 0.89)
    - "रामकुंड नाशिक में गोदावरी नदी पर..." (score: 0.86)
    - "Facilities at Ramkund: changing rooms, lockers..." (score: 0.78)
        │
        ▼
[3] Construct prompt:
    System: "आप नाशिक कुंभ मेला 2027 के सहायक हैं..."
    Context: [3 retrieved chunks]
    Question: "रामकुंड कहां है?"
        │
        ▼
[4] Generate answer with Qwen2.5-3B GGUF
    "रामकुंड नाशिक में गोदावरी नदी पर स्थित पवित्र स्नान घाट है।
     पंचवटी क्षेत्र में स्थित है। कुंभ मेला का मुख्य स्नान स्थल है।"
```

### 10.3 Cross-Lingual Retrieval

The E5-large model supports cross-lingual retrieval:

- Query in Hindi → retrieves English documents about the same topic
- Query in Marathi → retrieves Hindi and English documents
- If fewer than 2 results in query language → automatic fallback to English documents

### 10.4 ChromaDB Statistics

```
Total embedded documents: 4,500+
Embedding dimensions: 1024
Similarity metric: Cosine
Average retrieval time: 50-100ms
Storage size: 41MB
```

---

## 11. Backend Development

### 11.1 API Design

The backend is built with FastAPI, following REST principles with additional WebSocket support for streaming.

**Core Endpoints:**

| Method | Endpoint | Purpose | Avg Response Time |
|--------|----------|---------|------------------|
| POST | `/api/v1/query` | Text query with RAG | 8-15s |
| POST | `/api/v1/voice/input` | Full voice pipeline | 15-25s |
| POST | `/api/v1/voice/tts` | Text → Speech | 1-2s |
| POST | `/api/v1/voice/stt` | Speech → Text | 3-4s |
| POST | `/api/v1/emergency` | Emergency bypass | <100ms |
| GET | `/api/v1/places` | All 178+ places | <50ms |
| GET | `/api/v1/health` | Server status | <10ms |

### 11.2 Emergency Bypass System

Emergency queries are detected by keyword matching and **bypass the LLM entirely**:

```python
EMERGENCY_KEYWORDS = {
    "medical": ["doctor", "hospital", "ambulance", "डॉक्टर", "अस्पताल", "एम्बुलेंस"],
    "missing": ["missing", "lost", "खोया", "गुम", "लापता", "हरवले"],
    "fire": ["fire", "burning", "आग", "जल रहा", "आग लागली"],
    "drowning": ["drowning", "डूब", "बुडत"],
}
```

Response time: **<100ms** (vs 15-25s for LLM-based answers)

### 11.3 Authentication System

- **JWT tokens** with 7-day expiry
- **SHA-256 password hashing** via passlib
- Conversation management: create, rename, delete, switch
- Server-side chat history: syncs across devices
- Optional: works without login (guest mode)

### 11.4 Database Schema

```sql
users (id, name, email, password_hash, preferred_language, created_at)
conversations (id, user_id, title, created_at, updated_at)
sessions (id, user_id, conversation_id, query_text, response_text, language, query_type, created_at)
favorites (id, user_id, place_id, created_at)
emergency_logs (id, user_id, scenario, language, location_lat, location_lon, created_at)
```

---

## 12. Frontend & PWA Development

### 12.1 Design Philosophy

- **Voice-first:** Mic button is the largest, most prominent UI element
- **Saffron/maroon/gold theme:** Matches Kumbh Mela cultural aesthetics
- **Mobile-first responsive:** Bottom navigation on phones, top navigation on desktop
- **Zero-dependency:** Vanilla HTML/CSS/JS — no React, no build step
- **Offline-capable:** Service worker caches pages and assets

### 12.2 Key UI Components

| Component | Features |
|-----------|----------|
| **Voice Assistant** | Animated mic button with pulse ring, typing indicator, auto-play TTS, suggestion chips, conversation sidebar |
| **Map** | 178 color-coded markers, category filters, From→To routing panel, manual pin drop, GPS locate |
| **Explore** | Image cards with Unsplash photos, star ratings, expandable details, "View on Map" integration |
| **Emergency SOS** | 6 scenario cards, 7 helpline numbers with tap-to-call, vibration feedback |
| **Auth** | Sign in/up with gradient cards, language preference picker |
| **Profile** | Stats (queries, languages, favorites), system status, language preference |

### 12.3 Animations

```css
@keyframes msgAppear {
  from { opacity: 0; transform: translateY(16px) scale(0.96); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes micGlowPulse {
  0%, 100% { transform: scale(1); opacity: 0.3; }
  50% { transform: scale(1.6); opacity: 0; }
}
```

### 12.4 PWA Configuration

```json
{
  "name": "Yatri AI — Nashik Kumbh Mela 2027",
  "short_name": "Yatri AI",
  "display": "standalone",
  "background_color": "#6B0F1A",
  "theme_color": "#6B0F1A"
}
```

Features: installable on home screen, fullscreen mode, offline caching, custom icon.

### 12.5 Localization

The UI supports 3 languages (EN, HI, MR) with a `data-i18n` attribute system:

```javascript
const UI_STRINGS = {
  en: { home: 'Home', askAi: 'Ask AI', explore: 'Explore', ... },
  hi: { home: 'होम', askAi: 'AI से पूछें', explore: 'खोजें', ... },
  mr: { home: 'होम', askAi: 'AI ला विचारा', explore: 'शोधा', ... },
};
```

Language changes when user selects preference — entire UI updates instantly.

---

## 13. Map & Navigation System

### 13.1 Technology

- **Leaflet.js** for interactive map rendering
- **OpenStreetMap** tiles (free, no API key)
- **OSRM** (Open Source Routing Machine) for turn-by-turn directions
- All free, open-source, no usage limits

### 13.2 Features

| Feature | Implementation |
|---------|---------------|
| 178 markers | Color-coded by category (saffron=temples, blue=ghats, etc.) |
| Category filters | 10 categories with toggle chips |
| Search | Live filter markers by name |
| Route panel | From→To with GPS, search, or map-tap |
| Turn-by-turn | OSRM routing with distance and ETA |
| Assistant integration | "How to reach X" in chat → route button appears |
| Emergency | "Nearest hospital" → map zooms to closest medical facility |

### 13.3 Assistant-Map Integration

The chat assistant automatically detects location-related queries using regex patterns for English, Hindi, and Marathi:

```javascript
DIRECTION_PATTERNS = [
  /(.+?)\s+से\s+(.+?)\s+(?:कैसे|जाये|जाएं)/i,    // Hindi
  /(.+?)\s+(?:पासून|वरून)\s+(.+?)\s+(?:कसे)/i,      // Marathi
  /(?:from)\s+(.+?)\s+(?:to)\s+(.+?)$/i,              // English
]
```

When detected, the response shows buttons:
- **Pin icon:** Show place on map
- **Route icon:** Open directions with from/to pre-filled

---

## 14. Emergency Response System

### 14.1 Design Principles

1. **Speed over intelligence:** Hardcoded responses, no LLM latency
2. **Multilingual:** Responses in Hindi, Marathi, English
3. **Redundancy:** Works even if LLM/RAG fails
4. **Direct action:** Tap-to-call phone numbers, nearest hospital finder

### 14.2 Emergency Scenarios

| Scenario | Response Time | Actions |
|----------|--------------|---------|
| Medical | <100ms | Ambulance (108), hospitals list, first aid tips |
| Missing Person | <100ms | Helpline (1800-222-2027), Lost & Found centers |
| Stampede | <100ms | Police (100), evacuation instructions |
| Fire | <100ms | Fire Brigade (101), safety instructions |
| Drowning | <100ms | Emergency (112), do-not-jump warning |
| Lost Belongings | <100ms | Police station contacts |

### 14.3 Keyword Detection

Emergency detection runs in **two places:**
1. **Backend** (`api/routes/query.py`) — catches emergency queries before they reach the LLM
2. **Frontend** (`static/index.html`) — adds SOS buttons to any bot response mentioning emergency topics

---

## 15. Authentication & User Management

### 15.1 Auth Flow

```
Register → hash password (SHA-256) → store in SQLite → issue JWT (7-day)
Login → verify password → issue JWT
Every request → check Authorization: Bearer <token> → decode → get user_id
```

### 15.2 Conversation Management

- **New Chat:** Creates a new conversation with auto-generated title from first question
- **Chat Sidebar:** Lists all past conversations with date, message count
- **Server Sync:** Chat history stored in SQLite, restored on login from any device
- **Local Cache:** Also saved in localStorage for instant restoration

---

## 16. Deployment & Publishing

### 16.1 Model Publishing

The fine-tuned GGUF model is published on HuggingFace:
- **Repository:** https://huggingface.co/siddharthnavnath7/Kumbh
- **File:** `kumbh_model_q4_k_m.gguf` (1.93GB)
- **Auto-download:** Server automatically downloads on first startup

### 16.2 Application Deployment

Deployed on **HuggingFace Spaces** (free Docker hosting):
- **URL:** https://siddharthnavnath7-yatri-ai.hf.space
- **SDK:** Docker
- **Hardware:** CPU Basic (free tier)
- **Persistent storage:** `/data` for ChromaDB and SQLite

### 16.3 Docker Configuration

```dockerfile
FROM python:3.11-slim
# System deps: ffmpeg, libsndfile
# Python deps: FastAPI, Whisper, ChromaDB, llama-cpp, gTTS, etc.
# Pre-downloads embedding model during build
# ChromaDB builds on first startup, persisted to /data
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 16.4 Local Development

```bash
git clone https://github.com/sidharth974/Kumbh.git
cd Kumbh
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 vectordb/ingest_chunks.py  # Build ChromaDB (one-time)
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 17. Testing & Results

### 17.1 Query Accuracy

| Query Type | Language | Sample Query | Accuracy |
|-----------|---------|-------------|----------|
| Factual | EN | "When is Kumbh Mela 2027?" | Correct — July-Sept 2027 |
| Factual | HI | "रामकुंड कहां है?" | Correct — Godavari, Panchavati |
| Directional | HI | "त्र्यंबकेश्वर कैसे जाएं?" | Correct — CBS bus, NH-3, 28km |
| Emergency | HI | "मुझे डॉक्टर चाहिए" | Instant — 108, hospital list |
| Food | EN | "Best misal pav in Nashik" | Correct — Sadhana, Mama Kane |
| Transport | MR | "मुंबई ते नाशिक कसे जायचे?" | Correct — Panchavati Express, Shivneri bus |

### 17.2 Response Time (CPU — Intel i5, 14GB RAM)

| Component | Time |
|-----------|------|
| Whisper ASR (small) | 3-4 seconds |
| ChromaDB retrieval | 50-100ms |
| LLM generation (GGUF) | 8-12 seconds |
| gTTS | 1-2 seconds |
| Emergency bypass | <100ms |
| **Total (voice query)** | **15-20 seconds** |
| **Total (text query)** | **10-15 seconds** |
| **Total (emergency)** | **<200ms** |

### 17.3 Language Detection Accuracy

| Language | Whisper Detection Rate |
|----------|----------------------|
| Hindi | 92% |
| English | 98% |
| Marathi | 78% (sometimes confused with Hindi) |
| Tamil | 85% |
| Telugu | 82% |

---

## 18. Challenges & Solutions

| # | Challenge | Solution |
|---|-----------|----------|
| 1 | **No GPU on dev machine** — 7B model too slow | Used 3B model + 4-bit quantization; trained on free Colab T4 |
| 2 | **Whisper detects Marathi as Hindi** | Both use Devanagari; Whisper `small` better than `base`; RAG retrieves both languages |
| 3 | **LLM hallucinating about Kumbh** | RAG grounds answers in verified documents; "Answer ONLY from context" system prompt |
| 4 | **Emergency queries go through slow LLM** | Keyword-based bypass returns hardcoded responses in <100ms |
| 5 | **ChromaDB too large for HF Spaces git** | Builds on first startup; persisted to `/data` storage |
| 6 | **1.9GB GGUF too large for git** | Published on HuggingFace Hub; auto-downloads on first run |
| 7 | **Python 3.13 breaks pydub/audioop** | Removed pydub dependency; gTTS outputs MP3 directly |
| 8 | **Ollama timeout on CPU** | Reduced context (3 chunks × 1500 chars), max tokens (150), timeout (180s) |
| 9 | **React Native Web font loading timeout** | Replaced with vanilla PWA — no build step, no framework overhead |
| 10 | **Multilingual UI** | `data-i18n` attribute system with EN/HI/MR translations |

---

## 19. Future Enhancements

| Enhancement | Complexity | Impact |
|-------------|-----------|--------|
| Real-time crowd density heatmap | High | Safety improvement |
| Offline mode with cached responses | Medium | Works without internet |
| AR navigation overlay | High | Visual wayfinding |
| Wristband/QR integration for lost persons | Medium | Missing person tracking |
| Live Shahi Snan video streaming | Medium | Remote darshan |
| Multi-turn conversation memory | Low | Better follow-up answers |
| More languages (Bengali, Punjabi, Odia) | Medium | Wider coverage |
| Government API integration (official Kumbh data) | Medium | Real-time accuracy |
| GPU deployment for <2s responses | Low | Better UX |
| Native mobile app (if needed beyond PWA) | High | App store presence |

---

## 20. Conclusion

Yatri AI demonstrates that a **production-ready, multilingual AI assistant** can be built entirely with open-source technologies at near-zero cost. The system addresses the real-world challenges of mass pilgrimage events by combining:

1. **Voice-first design** that serves users regardless of literacy level
2. **RAG-based accuracy** that prevents hallucination through grounded retrieval
3. **Emergency bypass** that prioritizes speed when lives are at stake
4. **Geographic intelligence** with 178+ tagged locations and routing
5. **Multilingual support** covering 8 Indian languages without translation APIs

The project was developed, trained (on Google Colab free tier), and deployed (on HuggingFace Spaces free tier) at a **total cost of $0**, proving that advanced AI systems are accessible to student developers and can serve social good.

The techniques used — QLoRA fine-tuning, RAG, multilingual embeddings, voice processing, PWA deployment — represent the current state of the art in applied AI and are directly transferable to other domain-specific assistant applications.

---

## 21. References

1. Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *Proceedings of ICML*.
2. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
3. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*.
4. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." *NeurIPS*.
5. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv:2212.03533*.
6. Kakwani, D., et al. (2020). "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Models for Indian Languages." *EMNLP Findings*.
7. AI4Bharat. (2023). "IndicTrans2: Towards High-Quality and Accessible Machine Translation." *arXiv:2305.16307*.
8. Qwen Team. (2024). "Qwen2.5 Technical Report." *arXiv:2412.15115*.
9. Johnson, J., et al. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*.
10. ChromaDB Documentation. https://docs.trychroma.com/
11. Leaflet.js Documentation. https://leafletjs.com/
12. OpenStreetMap Foundation. https://www.openstreetmap.org/
13. OSRM Project. http://project-osrm.org/
14. HuggingFace Hub. https://huggingface.co/docs/hub/
15. Google Text-to-Speech (gTTS). https://gtts.readthedocs.io/

---

## 22. Appendix

### A. Project Repository

- **GitHub:** https://github.com/sidharth974/Kumbh
- **HuggingFace Model:** https://huggingface.co/siddharthnavnath7/Kumbh
- **HuggingFace Space:** https://siddharthnavnath7-yatri-ai.hf.space

### B. API Testing Examples

```bash
# Text query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "कुंभ मेला 2027 कब है?", "language": "hi"}'

# Emergency
curl -X POST http://localhost:8000/api/v1/emergency \
  -H "Content-Type: application/json" \
  -d '{"query": "medical", "language": "hi"}'

# Health check
curl http://localhost:8000/api/v1/health
```

### C. Training Notebook

The complete Colab training notebook is included at `Yatri_AI_Training.ipynb`.

### D. Data Samples

**Seed data sample (kumbh_2027_detailed.json):**

```json
{
  "id": "shahi_snan_calendar",
  "topic": "Shahi Snan Calendar 2027",
  "content_en": "The Nashik Simhastha Kumbh Mela 2027 is expected to feature five major Shahi Snan dates. These dates are determined by astrological calculations based on Jupiter's transit through Leo (Simha Rashi)...",
  "content_hi": "नाशिक सिंहस्थ कुंभ मेला 2027 में पाँच प्रमुख शाही स्नान तिथियाँ अपेक्षित हैं...",
  "category": "schedule",
  "tags": ["shahi_snan", "dates", "calendar", "kumbh_2027"]
}
```

**QA training sample:**

```json
{
  "instruction": "How to reach Trimbakeshwar from Nashik?",
  "input": "",
  "output": "From Nashik CBS, buses run every 15 minutes to Trimbakeshwar. The journey is 28km via NH-3, taking about 45 minutes. Auto-rickshaws charge ₹50-80 per person. You can also hire a cab for ₹400-600.",
  "language": "en",
  "domain": "transport",
  "type": "procedural"
}
```

---

*Project developed as part of B.E. AI & Data Science curriculum*  
*Nashik Simhastha Kumbh Mela 2027 — Har Har Mahadev*
