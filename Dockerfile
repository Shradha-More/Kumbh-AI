FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps (deploy-only, no training/crawling deps)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy app code + data
COPY api/ ./api/
COPY data/ ./data/
COPY static/ ./static/
COPY knowledge_base/ ./knowledge_base/
COPY vectordb/ ./vectordb/
RUN mkdir -p models

# Pre-download the embedding model during build (cached in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large'); print('Model cached')"

# HF Spaces uses 7860, standard uses 8000
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
