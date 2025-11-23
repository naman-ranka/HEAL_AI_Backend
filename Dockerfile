# Multi-stage Docker build for HEAL.AI
# Stage 1: Build Frontend
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend-clean
COPY frontend-clean/package*.json ./
RUN npm ci

COPY frontend-clean/ ./
RUN npm run build

# Stage 2: Python Backend with Frontend
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend-clean/dist ./static

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "main.py"]
