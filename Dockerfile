FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /build

COPY package.json package-lock.json* ./
COPY scripts/build-frontend.mjs ./scripts/build-frontend.mjs
COPY frontend ./frontend

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
RUN npm run build:frontend

FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    g++ \
    gcc \
    libgomp1 \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --prefer-binary -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user frontend ./frontend
COPY --chown=user package.json package-lock.json* ./
COPY --chown=user scripts ./scripts
COPY --chown=user start.sh README.md FEATURE.md PROJECT_LOGIC.md docker-compose.yml pyproject.toml ./

COPY --from=frontend-builder --chown=user /build/frontend/static ./frontend/static

RUN mkdir -p backend/runs backend/tmp && chmod +x start.sh

ENV PYTHONPATH=$HOME/app/backend:$PYTHONPATH
ENV MAX_UPLOAD_MB=500
ENV PORT=7860
ENV AUTOML_ALLOWED_ORIGINS=*

EXPOSE 7860

CMD ["bash", "start.sh"]
