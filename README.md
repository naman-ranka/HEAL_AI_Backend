# Insurance Policy Analyzer

A minimal web application that allows users to upload images of their insurance policies and get AI-powered summaries of key information like deductible, out-of-pocket maximum, and copay.

## Features

- Upload insurance card images
- AI-powered analysis using Google Gemini Pro Vision
- Extract key policy information (deductible, out-of-pocket max, copay)
- Simple and clean React frontend
- FastAPI backend with SQLite database

## Project Structure

```
HEAL/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── env.example         # Environment variables template
│   └── heal.db             # SQLite database (created automatically)
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js          # Main React component
    │   ├── App.css         # Styling
    │   ├── index.js        # React entry point
    │   └── index.css       # Global styles
    └── package.json        # Node.js dependencies
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```powershell
   # Windows PowerShell
   venv\Scripts\Activate.ps1
   
   # Windows Command Prompt
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Copy `env.example` to `.env`:
     ```powershell
     # Windows PowerShell
     Copy-Item env.example .env
     
     # Windows Command Prompt
     copy env.example .env
     ```
   - Add your Google Gemini API key to the `.env` file:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

6. Run the backend server:
   ```bash
   python main.py
   ```

The backend will start on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will start on `http://localhost:3000`

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key and add it to your `.env` file

## Usage

1. Make sure both backend and frontend servers are running
2. Open your browser to `http://localhost:3000`
3. Click "Upload Insurance Card"
4. Select an image file of your insurance policy
5. Wait for the AI analysis to complete
6. View the extracted information (deductible, out-of-pocket max, copay)

## API Endpoints

- `GET /` - Health check endpoint
- `POST /upload` - Upload insurance card image for analysis

## Database

The application uses SQLite with a single table:

```sql
CREATE TABLE policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_json TEXT NOT NULL
);
```

## Troubleshooting

### 500 Internal Server Error

If you get a 500 error when uploading files, check the following:

1. **Missing API Key**: Make sure you have created a `.env` file in the `backend` directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Dependencies**: Ensure all Python dependencies are installed:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Test Backend**: Run the test script to verify setup:
   ```bash
   cd backend
   python test_backend.py
   ```

### Quick Setup

Use the setup script for automated configuration:
```bash
python setup_backend.py
```

### PDF Support

Currently, PDF processing requires additional setup with Gemini 1.5 Pro. For now, use image files (JPEG, PNG) of your insurance documents.

## Notes

- The application accepts common image formats (JPEG, PNG, etc.)
- PDF support is planned but requires Gemini 1.5 Pro configuration
- AI analysis results are stored in the database for potential future use
- The frontend includes loading states and error handling
- CORS is configured to allow requests from the React development server
- Mock responses are provided when API key is not configured (for testing)
