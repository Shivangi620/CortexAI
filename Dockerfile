FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    redis-server \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Setup Nginx for non-root user
RUN mkdir -p /var/lib/nginx /var/log/nginx /run/nginx && \
    chown -R 1000:1000 /var/lib/nginx /var/log/nginx /run/nginx

# 🚀 CRITICAL FOR HUGGING FACE SPACES 🚀
# HF Spaces run as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy and install dependencies first (caches this step)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Create necessary directories for the backend
RUN mkdir -p backend/runs backend/tmp

RUN chmod +x start.sh

ENV PYTHONPATH=$HOME/app/backend:$PYTHONPATH

# Tell your start.sh script to boot Streamlit on 7860 (the only port HF exposes)
ENV PORT=7860
EXPOSE 7860

# Start up using your existing robust launch script (handles FastAPI + Streamlit simultaneously)
CMD ["bash", "start.sh"]
