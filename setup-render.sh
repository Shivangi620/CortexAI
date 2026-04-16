#!/bin/bash

# Render Deployment Setup
echo "🚀 Setting up AutoML Studio for Render deployment..."

# Create render.yaml for multi-service deployment
cat > render.yaml << EOF
services:
  - type: web
    name: automl-studio
    runtime: python3
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port \$PORT
    envVars:
      - key: REDIS_URL
        value: redis://redis:6379/0

  - type: redis
    name: automl-redis
    ipAllowList: []  # Only accessible by other services
EOF

echo "✅ Render configuration created!"
echo "1. Go to https://render.com"
echo "2. Connect your GitHub repository"
echo "3. Use the render.yaml file for deployment"
echo "4. Or deploy the web service manually with the commands above"