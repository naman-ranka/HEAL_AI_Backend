# HEAL.AI - Healthcare Expenses Analyzer & Logger

<div align="center">

**🏆 2nd Place Winner - Devlabs Hackathon**

*An AI-powered healthcare financial assistant that empowers patients to understand their medical bills and insurance policies with unprecedented clarity*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat&logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite)](https://vitejs.dev/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5-4285F4?style=flat&logo=google)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-06B6D4?style=flat&logo=tailwindcss)](https://tailwindcss.com/)

</div>

---

## 🎯 Overview

HEAL.AI is a comprehensive healthcare financial assistant that leverages cutting-edge AI technology to help patients navigate the complex world of medical insurance. Built during the Devlabs Hackathon where it secured **2nd place**, this platform combines advanced document analysis, RAG-powered conversational AI, and intelligent bill verification to give users unprecedented clarity and control over their healthcare expenses.

### 🌟 Problem Statement

**80% of medical bills contain errors** - Patients struggle to understand:
- What part is covered by insurance?
- What's truly owed out-of-pocket?
- Are there errors or duplicate charges?
- How to dispute billing discrepancies?

Navigating medical billing and health insurance is confusing and stressful for most patients, leading to overpayment and financial stress.

### 🌟 Key Features

- **📄 AI-Powered Insurance Analysis** - Upload insurance cards or policy documents to automatically extract deductibles, out-of-pocket maximums, copays, and comprehensive coverage details
- **💰 Smart Medical Bill Checker** - Analyze medical bills against your insurance policy to identify billing errors, coverage discrepancies, and potential disputes with detailed financial breakdowns
- **🤖 RAG-Powered Conversational AI** - Ask natural language questions about your insurance policy and get accurate, source-attributed answers through an intuitive chat interface
- **📧 Automated Dispute Generation** - Generate FDCPA-compliant dispute emails for billing errors with professional templates
- **🔍 Advanced Semantic Search** - Vector-based search across your policy documents with confidence scoring
- **📊 Comprehensive Dashboard** - Modern, responsive interface with insurance summaries, bill history, and emergency QR codes
- **💬 Persistent Chat Sessions** - Full conversation history with context-aware responses
- **🏥 Emergency QR Code** - Instant access to critical medical information for emergency situations

---

## 🏗️ Architecture

### Tech Stack

#### Frontend (Modern React SPA)
- **React 18.3** - Latest React with concurrent features and improved performance
- **TypeScript 5.8** - Full type safety and enhanced developer experience
- **Vite 5.4** - Lightning-fast build tool and development server
- **Tailwind CSS 3.4** - Utility-first CSS framework with custom design system
- **shadcn/ui** - Modern, accessible component library with 50+ components
- **React Router 6** - Client-side routing with nested routes
- **TanStack Query** - Powerful data fetching and caching
- **React Hook Form** - Performant forms with validation
- **Lucide React** - Beautiful, customizable icons
- **React Markdown** - Rich markdown rendering with syntax highlighting

#### Backend (Python FastAPI)
- **FastAPI 0.109** - Modern, high-performance Python web framework
- **Google Gemini 2.5** - State-of-the-art multimodal AI (Flash & Pro models)
- **SQLite** - Efficient local database with comprehensive schema
- **scikit-learn** - Vector similarity and machine learning operations
- **pytesseract** - OCR for image processing
- **PyMuPDF** - PDF text extraction and processing
- **NLTK** - Natural language processing and tokenization
- **Pydantic** - Data validation and serialization

#### AI/ML Pipeline
- **Retrieval-Augmented Generation (RAG)** - Context-aware document question answering
- **Vector Embeddings** - Google's text-embedding-004 (768 dimensions)
- **Semantic Search** - Cosine similarity-based retrieval with confidence scoring
- **Document Chunking** - Intelligent text segmentation with 2-sentence overlap
- **Multimodal Analysis** - Simultaneous processing of images and text

---

## 📁 Project Structure

```
HEAL/
├── backend/                             # Python FastAPI Backend
│   ├── main.py                          # Main FastAPI application (2000+ lines)
│   ├── langchain_main.py                # LangChain integration server
│   ├── genkit_api.py                    # Genkit-style API patterns
│   ├── ai/                              # AI Processing Layer
│   │   ├── flows/
│   │   │   ├── policy_analysis.py       # Insurance policy extraction flows
│   │   │   └── chatbot.py               # RAG chatbot implementation
│   │   ├── embedder.py                  # Gemini embeddings with fallback
│   │   ├── genkit_config.py             # AI service configuration
│   │   └── schemas.py                   # Pydantic data models
│   ├── rag/                             # RAG System Components
│   │   ├── document_processor.py        # Text extraction & intelligent chunking
│   │   ├── retriever.py                 # Vector similarity search engine
│   │   └── chatbot.py                   # Context-aware RAG chatbot
│   ├── database/                        # Database Layer
│   │   └── schema.py                    # SQLite schema & connection management
│   ├── services/                        # Business Logic Services
│   │   ├── gemini_service.py            # Gemini API wrapper & utilities
│   │   └── bill_analysis_service.py     # Medical bill analysis engine
│   ├── uploads/                         # File storage directory
│   ├── requirements.txt                 # Python dependencies
│   └── venv/                            # Virtual environment
├── frontend/ (Git Submodule)            # Modern React TypeScript Frontend
│   ├── src/
│   │   ├── App.tsx                      # Main application component
│   │   ├── main.tsx                     # Vite entry point
│   │   ├── components/                  # Reusable UI Components
│   │   │   ├── ui/                      # shadcn/ui component library (50+ components)
│   │   │   ├── layout/                  # Layout components (Header, Layout)
│   │   │   ├── sections/                # Page sections (Hero, Features, Upload)
│   │   │   ├── BillSummaryCard.tsx      # Bill analysis display (374 lines)
│   │   │   ├── PolicySummary.tsx        # Insurance policy visualization
│   │   │   ├── DisputeEmailModal.tsx    # Dispute generation interface
│   │   │   ├── BillAnalysisLoader.tsx   # Loading states for analysis
│   │   │   └── MarkdownMessage.tsx      # Rich text message rendering
│   │   ├── pages/                       # Application Pages
│   │   │   ├── Index.tsx                # Landing page with hero section
│   │   │   ├── Dashboard.tsx            # Main dashboard with tabs
│   │   │   ├── Chat.tsx                 # AI chat interface
│   │   │   ├── BillSummary.tsx          # Detailed bill analysis view
│   │   │   ├── Admin.tsx                # Admin panel for data management
│   │   │   └── NotFound.tsx             # 404 error page
│   │   ├── contexts/                    # React Context Providers
│   │   │   └── AppContext.tsx           # Global application state
│   │   ├── services/                    # API Integration Layer
│   │   │   └── api.ts                   # Comprehensive API service (420+ lines)
│   │   ├── hooks/                       # Custom React Hooks
│   │   │   ├── use-toast.ts             # Toast notification system
│   │   │   └── use-mobile.tsx           # Mobile responsive utilities
│   │   └── lib/                         # Utility Functions
│   │       └── utils.ts                 # Common utilities and helpers
│   ├── public/                          # Static Assets
│   │   ├── favicon.ico                  # Application favicon
│   │   └── placeholder.svg              # Placeholder images
│   ├── dist/                            # Production build output
│   ├── package.json                     # Node.js dependencies & scripts
│   ├── tailwind.config.ts               # Tailwind CSS configuration
│   ├── vite.config.ts                   # Vite build configuration
│   ├── tsconfig.json                    # TypeScript configuration
│   └── components.json                  # shadcn/ui component configuration
├── docs/                                # Comprehensive Documentation
│   ├── RAG_IMPLEMENTATION_GUIDE.md      # RAG system implementation details
│   ├── BACKEND_SUCCESS_SUMMARY.md       # Backend development summary
│   ├── IMPLEMENTATION_SUMMARY.md        # Overall project implementation
│   ├── GEMINI_SETUP.md                  # Google Gemini API setup guide
│   ├── GENKIT_MIGRATION_GUIDE.md        # Genkit pattern migration
│   └── CHAT_IMPROVEMENTS_SUMMARY.md     # Chat system enhancements
├── .gitmodules                          # Git submodule configuration
└── README.md                            # This comprehensive guide
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ and npm
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/naman-ranka/HEAL_AI_Backend.git
cd HEAL_AI_Backend
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
# Initialize submodules (if cloning fresh)
git submodule update --init --recursive

cd frontend

# Install dependencies
npm install

# Or using alternative package managers:
# yarn install
# pnpm install
# bun install
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

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Start the FastAPI server
python main.py
# Or using uvicorn directly:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs on `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

#### Start Frontend Development Server

```bash
cd frontend

# Start Vite development server
npm run dev
# Or using alternative package managers:
# yarn dev
# pnpm dev
# bun dev

# For production build:
# npm run build
# npm run preview
```

Frontend runs on `http://localhost:8080` (Vite default)
- Hot module replacement enabled
- TypeScript compilation
- Tailwind CSS processing

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

### Modern Frontend Experience

- **Responsive Design** - Mobile-first approach with adaptive layouts
- **Dark/Light Mode** - System preference detection with manual toggle
- **Progressive Loading** - Skeleton screens and optimistic updates
- **Real-time Updates** - Live chat interface with typing indicators
- **Accessibility** - WCAG 2.1 compliant with keyboard navigation
- **Component Library** - 50+ reusable shadcn/ui components
- **Type Safety** - Full TypeScript coverage with strict mode
- **State Management** - React Context with optimistic updates

### Advanced AI Capabilities

- **Multimodal Analysis** - Processes both images and text simultaneously
- **Vector Embeddings** - Google's latest text-embedding-004 model (768 dimensions)
- **Fallback Handling** - Graceful degradation when AI services are unavailable
- **Confidence Scoring** - All responses include reliability metrics (0-1 scale)
- **Semantic Search** - Context-aware document retrieval with similarity thresholds
- **Conversation Memory** - Maintains context across chat sessions
- **Source Attribution** - Links responses to specific document sections

### Comprehensive Database Schema

```sql
-- Core Tables
policies              # Insurance policy metadata and analysis results
documents             # Uploaded documents with file information
document_chunks       # Text chunks with vector embeddings (768-dim)
chat_sessions         # User chat sessions with document context
chat_messages         # Complete conversation history
bill_analyses         # Medical bill analysis results and recommendations
rag_queries           # RAG query logs with performance metrics

-- Analytics Tables
upload_logs           # File upload tracking and error logs
embedding_stats       # Vector embedding performance metrics
user_interactions     # User behavior analytics
```

### Developer Experience

- **Hot Reload** - Instant updates during development (Vite HMR)
- **Debug Endpoints** - Comprehensive debugging tools for development
- **Admin Panel** - Database management and cleanup utilities in frontend
- **Comprehensive Logging** - Structured logging throughout the application
- **Type Safety** - Pydantic models (backend) + TypeScript interfaces (frontend)
- **API Documentation** - Auto-generated OpenAPI/Swagger docs
- **Error Boundaries** - Graceful error handling with user-friendly messages

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

## 📚 Use Cases & User Journey

### 1. **New User Onboarding**
- Upload insurance card or policy document via drag-and-drop interface
- AI extracts key information (deductible, copay, out-of-pocket max)
- View comprehensive policy summary in dashboard
- Access emergency QR code for critical medical information

### 2. **Medical Bill Analysis**
- Upload medical bill (PDF, image, or photo)
- AI analyzes bill against insurance policy
- Receive detailed financial breakdown with coverage analysis
- Identify potential billing errors or discrepancies
- Generate professional dispute emails if needed

### 3. **Interactive Policy Consultation**
- Ask natural language questions about coverage
- Get instant, source-attributed answers from AI chatbot
- Explore coverage scenarios for planned procedures
- Understand network restrictions and referral requirements

### 4. **Ongoing Healthcare Management**
- Track bill analysis history with searchable interface
- Monitor healthcare spending patterns
- Access emergency medical information via QR code
- Maintain conversation history for reference

### 5. **Administrative Tasks**
- Generate FDCPA-compliant dispute letters
- Export analysis results for record-keeping
- Manage multiple insurance policies (family coverage)
- Reset and clean data through admin panel

---

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
python test_backend.py
python test_langchain_complete.py
python test_genkit_system.py

# Frontend tests (if configured)
cd frontend
npm run test
npm run test:coverage
```

### Development Tools

#### Backend Debug Endpoints
- `/debug/upload-process/{id}` - Debug upload flow
- `/debug/latest-upload` - Latest document information  
- `/debug/embeddings/stats` - Embedding statistics
- `/debug/chat/context` - Test chat context building
- `/admin/database-info` - Database statistics
- `/admin/reset-all` - Reset all data (development only)

#### Frontend Development
```bash
cd frontend

# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

### Git Submodule Management

```bash
# Update frontend submodule to latest
git submodule update --remote frontend

# Pull latest changes including submodules
git pull --recurse-submodules

# Clone with submodules
git clone --recurse-submodules <repository-url>
```

---

## 🚧 Known Limitations

- Requires active internet connection for AI features
- OCR accuracy depends on image quality
- Response time varies based on document size
- SQLite limits concurrent write operations

---

## 🔮 Future Enhancements

### Frontend Improvements
- [ ] Progressive Web App (PWA) support
- [ ] Offline functionality with service workers
- [ ] Advanced data visualization with charts
- [ ] Multi-language internationalization (i18n)
- [ ] Enhanced accessibility features
- [ ] Mobile-optimized touch interactions

### Backend & AI Enhancements  
- [ ] Multi-user authentication and authorization
- [ ] Cloud database integration (PostgreSQL/MongoDB)
- [ ] Integration with healthcare provider APIs
- [ ] Advanced analytics and reporting dashboard
- [ ] Export functionality (PDF/Excel/CSV)
- [ ] Real-time collaboration features
- [ ] Enhanced OCR with multiple AI models
- [ ] Voice input and audio responses

### System Architecture
- [ ] Microservices architecture
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline automation
- [ ] Performance monitoring and alerting
- [ ] Automated testing suite expansion

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
