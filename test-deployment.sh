#!/bin/bash

# Deployment Test Script
echo "🧪 Testing AutoML Studio deployment readiness..."

# Check if required files exist
files=("requirements.txt" "backend/main.py" "frontend/app.py" "Dockerfile" "docker-compose.yml")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Test Python imports
echo "Testing Python dependencies..."
python3 -c "
import fastapi
import uvicorn
import pandas
import sklearn
import xgboost
print('✅ All core dependencies available')
"

# Check if ports are available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  Port 8000 is already in use"
else
    echo "✅ Port 8000 is available"
fi

if lsof -Pi :6379 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️  Port 6379 (Redis) is already in use"
else
    echo "✅ Port 6379 (Redis) is available"
fi

echo "🎉 Deployment test complete!"
echo ""
echo "Ready to deploy with:"
echo "  Railway: railway up"
echo "  Docker: docker-compose up -d"
echo "  Local: bash run.sh"