@echo off
REM HEAL.AI Easy Deployment Script for Windows
echo 🚀 HEAL.AI Deployment Script
echo ==============================

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Please run this script from the HEAL project root directory
    exit /b 1
)

if not exist "backend" (
    echo ❌ Backend directory not found
    exit /b 1
)

if not exist "frontend" (
    echo ❌ Frontend directory not found
    exit /b 1
)

echo 📦 Building frontend...
cd frontend-clean
call npm install
call npm run build
cd ..

echo 📁 Copying frontend build to backend static folder...
if exist "backend\static" rmdir /s /q "backend\static"
xcopy /e /i "frontend-clean\dist" "backend\static"

echo 🐳 Building Docker image...
docker build -t heal-ai .

echo ✅ Build complete!
echo.
echo 🚀 Deployment Options:
echo 1. Local Docker: docker run -p 8000:8000 -e GEMINI_API_KEY=your_key heal-ai
echo 2. Railway: railway up
echo 3. Render: git push (if connected to Render)
echo 4. Docker Compose: docker-compose up
echo.
echo 📝 Don't forget to set your GEMINI_API_KEY environment variable!
