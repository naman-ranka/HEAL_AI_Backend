# 🎬 HEAL.AI Demo Preparation Guide

## 🚀 Quick Demo Reset

### 1. **Access Admin Panel**
- Navigate to: `http://localhost:5173/admin` (development only)
- The Admin link appears in the header when `NODE_ENV=development`

### 2. **Reset Database**
- Click **"Reset All Data"** button
- Confirm twice (safety measure)
- Wait for completion notification
- Verify stats show all zeros

### 3. **Demo Flow**
1. **Upload Insurance** → Shows real AI analysis
2. **View Dashboard** → See parsed deductible, copay, etc.
3. **Start Chat** → Ask questions about coverage
4. **Upload Medical Bill** → Get detailed analysis & dispute detection

## 🔧 API Endpoints (for scripts)

### Reset Everything
```bash
curl -X DELETE "http://localhost:8000/admin/reset-all?confirm=CONFIRM_RESET"
```

### Get Database Stats
```bash
curl "http://localhost:8000/admin/database-info"
```

### Cleanup Embeddings Only
```bash
curl -X POST "http://localhost:8000/admin/cleanup-embeddings"
```

## 📊 What Gets Reset

✅ **All Documents** - Insurance policies and medical bills  
✅ **All Chunks** - Text chunks and embeddings  
✅ **All Chat Sessions** - Conversation history  
✅ **All Analyses** - Policy and bill analysis results  
✅ **All Files** - Uploaded files in `/uploads` directory  
✅ **Auto-increment IDs** - Reset to start from 1  

## 🔒 Security Features

- **Environment Check** - Only works in development
- **Double Confirmation** - Requires explicit user confirmation
- **Production Block** - Automatically disabled in production
- **Detailed Logging** - All reset operations are logged

## 📈 Admin Dashboard Features

### **Database Statistics**
- Real-time counts of all data types
- File storage usage
- Embedding dimension analysis

### **Cleanup Tools**
- Remove mismatched embeddings
- Fix dimension inconsistencies
- Optimize database performance

### **System Information**
- Environment detection
- Storage usage monitoring
- Connection status

## 🎯 Demo Script

### **Opening (Clean Slate)**
1. Show Admin dashboard with zero counts
2. Explain AI-powered insurance analysis

### **Upload Insurance**
1. Upload sample insurance document
2. Show real Gemini 2.5 Pro analysis
3. Navigate to Dashboard → see parsed data

### **Chat Interaction**
1. Click "Start AI Chat Session"
2. Ask: "What's my deductible?"
3. Ask: "What does my policy cover?"
4. Show RAG-powered responses with sources

### **Bill Analysis**
1. Upload medical bill in chat
2. Show automatic analysis
3. Highlight dispute detection
4. Explain financial breakdown

### **Reset for Next Demo**
1. Go to Admin panel
2. Reset all data
3. Ready for next demonstration

## 🛠️ Environment Variables

```bash
# Backend (.env)
ENVIRONMENT=development  # Enables admin endpoints
GEMINI_API_KEY=your_key  # For real AI analysis

# Frontend (automatically detected)
NODE_ENV=development     # Shows admin link in header
```

## 📝 Quick Reset Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
- [ ] Navigate to `/admin`
- [ ] Click "Reset All Data"
- [ ] Confirm twice
- [ ] Verify zero counts
- [ ] Ready for demo!

The admin system provides a clean, safe way to reset everything for demos while protecting against accidental data loss in production environments.


