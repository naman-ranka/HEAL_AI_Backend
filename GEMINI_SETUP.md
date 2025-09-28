# Gemini AI Setup Guide for HEAL

This guide provides step-by-step instructions for setting up Google's Gemini AI in the HEAL application.

## 🚀 Quick Setup

### 1. Get Your Gemini API Key

1. **Visit Google AI Studio**: Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Sign in** with your Google account
3. **Create API Key**: Click "Create API Key" 
4. **Copy the key**: Save it securely - you'll need it for configuration

### 2. Configure the Backend

#### Option A: Automatic Setup (Recommended)
```bash
# Run the setup script from the project root
python setup_backend.py
```

#### Option B: Manual Setup
```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Create environment file
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env

# 3. Replace 'your_actual_api_key_here' with your real API key
```

### 3. Test the Setup

```bash
cd backend
python test_backend.py
```

### 4. Start the Server

```bash
cd backend
python main.py
```

## 🏗️ Architecture Overview

### Gemini Service (`services/gemini_service.py`)

The HEAL application uses a dedicated `GeminiService` class that provides:

- **Document Analysis**: Images and PDFs
- **Chat Functionality**: For future chatbot features
- **Error Handling**: Robust error management
- **Model Management**: Multiple Gemini models support
- **Safety Settings**: Production-ready safety configurations

### Available Models

| Model | Purpose | Use Case |
|-------|---------|----------|
| `gemini-1.5-flash` | Fast responses | Image analysis, quick tasks |
| `gemini-1.5-pro` | Complex analysis | PDF processing, detailed analysis |
| `gemini-pro-vision` | Vision tasks | Image analysis (fallback) |

### Key Features

#### 🖼️ Image Analysis
- Supports JPEG, PNG, and other common formats
- Automatic image preprocessing
- Optimized prompts for insurance document extraction

#### 📄 PDF Processing
- Direct PDF upload to Gemini
- Fallback text extraction using PyMuPDF
- Handles multi-page documents

#### 💬 Chat Foundation
- Ready for chatbot implementation
- Conversation history management
- Context-aware responses

#### 🛡️ Safety & Error Handling
- Production-ready safety settings
- Comprehensive error logging
- Graceful fallbacks for service unavailability

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# Required: Your Gemini API key
GEMINI_API_KEY=your_actual_api_key_here

# Optional: Logging level
LOG_LEVEL=INFO

# Optional: Database path
DATABASE_PATH=heal.db
```

### Model Configuration

The service automatically selects the best model for each task:

- **Images**: `gemini-1.5-flash` (fast, efficient)
- **PDFs**: `gemini-1.5-pro` (handles complex documents)
- **Chat**: `gemini-1.5-pro` (conversational AI)

## 📊 API Endpoints

### Document Analysis
```http
POST /upload
Content-Type: multipart/form-data

# Upload image or PDF file
```

### Health Check
```http
GET /health

# Returns system status including Gemini service health
```

### Chat (Future Feature)
```http
POST /chat
Content-Type: application/json

{
  "message": "Hello, can you help me understand my policy?",
  "history": []
}
```

### Available Models
```http
GET /models

# Returns list of available Gemini models
```

## 🧪 Testing

### Automated Testing
```bash
cd backend
python test_backend.py
```

### Manual Testing

1. **Health Check**: Visit `http://localhost:8000/health`
2. **API Docs**: Visit `http://localhost:8000/docs`
3. **Upload Test**: Use the frontend or curl:
   ```bash
   curl -X POST "http://localhost:8000/upload" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_insurance_card.jpg"
   ```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Gemini API key not configured"
- **Solution**: Ensure `.env` file exists with correct API key
- **Check**: Run `python test_backend.py` to verify configuration

#### 2. "Model not available"
- **Solution**: Check your API key has access to Gemini models
- **Fallback**: The service will use mock data if models are unavailable

#### 3. "PDF processing failed"
- **Solution**: Ensure PyMuPDF is installed: `pip install PyMuPDF`
- **Fallback**: Service will extract text and analyze that instead

#### 4. "Rate limit exceeded"
- **Solution**: Implement request throttling or upgrade API quota
- **Monitoring**: Check logs for rate limit warnings

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

### Service Status

Check service health:
```bash
curl http://localhost:8000/health
```

## 🚀 Future Enhancements

### Planned Features

1. **Chatbot Integration**
   - Insurance policy Q&A
   - Claims assistance
   - Policy comparison

2. **Advanced Analysis**
   - Multi-document comparison
   - Trend analysis
   - Risk assessment

3. **Performance Optimization**
   - Response caching
   - Batch processing
   - Model fine-tuning

### Extensibility

The `GeminiService` class is designed for easy extension:

```python
# Add new analysis types
async def analyze_claim_form(self, file_data: bytes) -> Dict[str, Any]:
    prompt = "Extract claim information from this document..."
    return await self.analyze_image(file_data, prompt)

# Add specialized chat functions
async def insurance_chat(self, message: str) -> Dict[str, Any]:
    system_prompt = "You are an insurance expert assistant..."
    return await self.chat(f"{system_prompt}\n\nUser: {message}")
```

## 📚 Resources

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Google AI Studio](https://aistudio.google.com/)
- [Python SDK Reference](https://ai.google.dev/gemini-api/docs/python-quickstart)
- [Safety Settings Guide](https://ai.google.dev/gemini-api/docs/safety-settings)

## 🆘 Support

If you encounter issues:

1. **Check the logs**: Look for error messages in the console
2. **Run tests**: Use `python test_backend.py` to diagnose issues
3. **Verify API key**: Ensure your Gemini API key is valid and has proper permissions
4. **Check quotas**: Verify you haven't exceeded API rate limits

For additional help, refer to the [Google AI Developer Forum](https://discuss.ai.google.dev/).
