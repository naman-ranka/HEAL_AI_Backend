# HEAL AI - Healthcare Expense Analysis and Legislation

<div align="center">

**🏆 2nd Place Winner - Devlabs Hackathon**

*An AI-powered insurance document analysis platform that empowers patients to understand their medical bills and insurance policies*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0-61DAFB?style=flat&logo=react)](https://reactjs.org/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5-4285F4?style=flat&logo=google)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://www.python.org/)

</div>

---

## 🎯 Overview

HEAL AI is a comprehensive healthcare financial assistant that leverages cutting-edge AI technology to help patients navigate the complex world of medical insurance. Built during the Devlabs Hackathon where it secured **2nd place**, this platform combines document analysis, RAG-powered chatbots, and intelligent bill checking to give users unprecedented clarity over their healthcare expenses.

### 🌟 Key Features

- **📄 Insurance Policy Analysis** - Upload insurance cards or policy documents to automatically extract key information including deductibles, out-of-pocket maximums, copays, and coverage details
- **💰 Medical Bill Checker** - Analyze medical bills against your insurance policy to identify billing errors, coverage discrepancies, and potential disputes
- **🤖 RAG-Powered Chatbot** - Ask natural language questions about your insurance policy and get accurate, source-attributed answers
- **📧 Dispute Generation** - Automatically generate FDCPA-compliant dispute emails for billing errors
- **🔍 Semantic Search** - Advanced vector-based search across your policy documents
- **📊 Session Management** - Persistent chat sessions with full conversation history

---

## 🏗️ Architecture

### Tech Stack

#### Backend
- **FastAPI** - Modern, high-performance Python web framework
- **Google Gemini 2.5** - State-of-the-art multimodal AI (Flash & Pro models)
- **SQLite** - Efficient local database with comprehensive schema
- **scikit-learn** - Vector similarity and machine learning operations
- **pytesseract** - OCR for image processing
- **PyMuPDF** - PDF text extraction
- **NLTK** - Natural language processing and tokenization

#### AI/ML Pipeline
- **Retrieval-Augmented Generation (RAG)** - Context-aware document question answering
- **Vector Embeddings** - Google's text-embedding-004 (768 dimensions)
- **Semantic Search** - Cosine similarity-based retrieval
- **Document Chunking** - Intelligent text segmentation with overlap

#### Frontend
- **React 18** - Modern, component-based UI framework
- **Responsive Design** - Mobile-friendly interface

---

## 📁 Project Structure

```
HEAL_AI_Backend/
├── backend/                             # Backend API server
│   ├── main.py                          # FastAPI application (2091 lines)
│   ├── genkit_api.py                    # Genkit-style API server
│   ├── ai/
│   │   ├── flows/
│   │   │   ├── policy_analysis.py       # Policy extraction flows
│   │   │   └── chatbot.py               # RAG chatbot implementation
│   │   ├── embedder.py                  # Gemini embeddings with fallback
│   │   ├── genkit_config.py             # AI configuration
│   │   └── schemas.py                   # Pydantic data models
│   ├── rag/
│   │   ├── document_processor.py        # Text extraction & chunking
│   │   ├── retriever.py                 # Vector similarity search
│   │   └── chatbot.py                   # RAG-powered chatbot
│   ├── database/
│   │   └── schema.py                    # SQLite schema & connections
│   ├── services/
│   │   ├── gemini_service.py            # Gemini API wrapper
│   │   └── bill_analysis_service.py     # Bill analysis logic
│   └── requirements.txt                 # Python dependencies
├── frontend/ @ heal-ai                  # React frontend (git submodule)
│   └── (React + TypeScript application)
├── docs/                                # Comprehensive documentation
│   ├── RAG_IMPLEMENTATION_GUIDE.md
│   ├── BACKEND_SUCCESS_SUMMARY.md
│   ├── CHAT_IMPROVEMENTS_SUMMARY.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── GEMINI_SETUP.md
│   └── GENKIT_MIGRATION_GUIDE.md
└── .gitmodules                          # Submodule configuration
```

**Note:** The `frontend` directory is a git submodule linking to a separate repository. Make sure to clone with `--recurse-submodules` or run `git submodule update --init` after cloning.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ and npm
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

#### 1. Clone the Repository

**Important:** This repository uses a git submodule for the frontend. Clone with submodules:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/naman-ranka/HEAL_AI_Backend.git
cd HEAL_AI_Backend

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### Configuration

Create a `.env` file in the `backend` directory:

```env
GEMINI_API_KEY=your_api_key_here
ENVIRONMENT=development
```

### Running the Application

#### Start Backend Server

```bash
cd backend
python main.py
```

Backend runs on `http://localhost:8000`

#### Start Frontend Development Server

```bash
cd frontend
npm start
```

Frontend runs on `http://localhost:3000`

---

## 📡 API Endpoints

### Document Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and analyze insurance policy documents |
| `POST` | `/summarize` | Generate comprehensive document summary |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/documents/{id}` | Get specific document details |

### Bill Checker

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/bill-checker/upload` | Upload medical bill for analysis |
| `POST` | `/bill-checker/analyze` | Analyze bill against insurance policy |
| `GET` | `/bill-checker/analysis/{id}` | Retrieve analysis results |
| `POST` | `/bill-checker/analysis/{id}/dispute` | Generate FDCPA-compliant dispute email |
| `GET` | `/bill-checker/history` | Get bill analysis history |

### RAG Chat System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/sessions` | Create new chat session |
| `POST` | `/chat/sessions/{id}/messages` | Send message to chatbot |
| `GET` | `/chat/sessions/{id}/history` | Get conversation history |
| `DELETE` | `/chat/sessions/{id}` | Delete chat session |
| `GET` | `/chat/sessions` | List all sessions |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check with AI service status |
| `GET` | `/models` | Available AI models |

---

## 💡 How It Works

### 1. Policy Analysis Pipeline

```
Upload Document → Text Extraction (OCR/PDF) → Gemini Analysis →
Structured Extraction → Confidence Scoring → Database Storage
```

- Supports images (JPEG, PNG) and PDF documents
- Extracts: policyholder info, coverage costs, common services, prescriptions
- Returns structured JSON with confidence scores

### 2. RAG (Retrieval-Augmented Generation) System

```
Document Upload → Text Chunking (750 words) → Embedding Generation (768-dim) →
Vector Storage → Query Processing → Semantic Search → Context Building →
AI Response Generation → Source Attribution
```

**Features:**
- Intelligent chunking with 2-sentence overlap
- Top-k retrieval with configurable thresholds
- Conversation history integration (last 6 messages)
- Source attribution showing which document sections informed the response

### 3. Bill Analysis

```
Upload Bill + Policy → Multimodal Gemini Analysis →
Financial Breakdown → Coverage Analysis → Discrepancy Detection →
Dispute Recommendations
```

- Identifies billing errors and coverage issues
- Calculates patient responsibility
- Generates professional dispute emails

---

## 🎨 Features in Detail

### Advanced AI Capabilities

- **Multimodal Analysis** - Processes both images and text simultaneously
- **Vector Embeddings** - Google's latest text-embedding-004 model
- **Fallback Handling** - Graceful degradation when AI services are unavailable
- **Confidence Scoring** - All responses include reliability metrics
- **Semantic Search** - Context-aware document retrieval

### Database Schema

```sql
policies          # Uploaded insurance policies
documents         # Policy documents with metadata
document_chunks   # Chunked text with embeddings
chat_sessions     # User chat sessions
chat_messages     # Conversation history
bill_analyses     # Medical bill analysis results
rag_queries       # RAG query logs and statistics
```

### Developer Tools

- **Debug Endpoints** - Comprehensive debugging tools for development
- **Admin Tools** - Database management and cleanup utilities
- **Logging** - Detailed logging throughout the application
- **Type Safety** - Pydantic models for all data structures

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `ENVIRONMENT` | Application environment | `development` |

### RAG Configuration

- **Chunk Size**: 750 words
- **Chunk Overlap**: 2 sentences
- **Embedding Dimensions**: 768
- **Top-k Results**: Configurable (default: 5)
- **Similarity Threshold**: Configurable (default: 0.3)

---

## 📚 Use Cases

1. **Policy Understanding** - Upload your insurance card and instantly understand your coverage
2. **Bill Verification** - Check if your medical bills match your insurance coverage
3. **Cost Planning** - Ask questions about coverage for specific procedures
4. **Dispute Resolution** - Generate professional dispute emails for billing errors
5. **Coverage Comparison** - Understand what services are covered vs. not covered

---

## 🛠️ Development

### Running Tests

```bash
cd backend
python test_backend.py
```

### Debug Mode

Access debug endpoints for development:
- `/debug/upload-process/{id}` - Debug upload flow
- `/debug/latest-upload` - Latest document information
- `/debug/embeddings/stats` - Embedding statistics
- `/debug/chat/context` - Test chat context building

---

## 🚧 Known Limitations

- Requires active internet connection for AI features
- OCR accuracy depends on image quality
- Response time varies based on document size
- SQLite limits concurrent write operations

---

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with healthcare provider APIs
- [ ] Advanced analytics dashboard
- [ ] Export to PDF/Excel functionality
- [ ] Multi-user support with authentication
- [ ] Cloud database integration
- [ ] Real-time collaboration features

---

## 📄 License

This project was developed for the Devlabs Hackathon. All rights reserved.

---

## 👨‍💻 Author

**Naman Ranka**

- GitHub: [@naman-ranka](https://github.com/naman-ranka)

---

## 🏆 Acknowledgments

- **2nd Place** at Devlabs Hackathon
- Built with Google Gemini AI
- Inspired by Genkit AI framework patterns
- Special thanks to the Devlabs community

---

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

<div align="center">

**Made with ❤️ for the Devlabs Hackathon**

*Empowering patients through AI-powered healthcare transparency*

</div>
