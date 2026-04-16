FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p backend/runs backend/tmp

ENV PYTHONPATH=/app/backend:$PYTHONPATH

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "./start.sh"]