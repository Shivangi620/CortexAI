#!/bin/bash

# Railway Deployment Script
echo "🚀 Deploying AutoML Studio to Railway..."

# Install Railway CLI if not present
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway (user will need to authenticate)
railway login

# Initialize Railway project
railway init automl-studio

# Set environment variables
railway variables set REDIS_URL=redis://redis:6379/0

# Deploy
railway up

echo "✅ Deployment complete!"
echo "Your app will be available at the URL shown above"