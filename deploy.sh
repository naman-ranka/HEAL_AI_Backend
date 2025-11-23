#!/bin/bash

# HEAL.AI Easy Deployment Script
echo "🚀 HEAL.AI Deployment Script"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the HEAL project root directory"
    exit 1
fi

echo "📦 Building frontend..."
cd frontend-clean
npm install
npm run build
cd ..

echo "📁 Copying frontend build to backend static folder..."
rm -rf backend/static
cp -r frontend-clean/dist backend/static

echo "🐳 Building Docker image..."
docker build -t heal-ai .

echo "✅ Build complete!"
echo ""
echo "🚀 Deployment Options:"
echo "1. Local Docker: docker run -p 8000:8000 -e GEMINI_API_KEY=your_key heal-ai"
echo "2. Railway: railway up"
echo "3. Render: git push (if connected to Render)"
echo "4. Docker Compose: docker-compose up"
echo ""
echo "📝 Don't forget to set your GEMINI_API_KEY environment variable!"
