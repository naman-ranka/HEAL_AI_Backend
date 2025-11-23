# 🚀 HEAL.AI Deployment Guide

## Quick Deploy Options (Choose One)

### 1. 🚂 Railway (Recommended - Easiest)

**One-command deployment:**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Steps:**
1. Push your code to GitHub
2. Connect Railway to your GitHub repo
3. Set `GEMINI_API_KEY` environment variable
4. Railway automatically builds and deploys!

**URL**: Your app will be available at `https://your-app.railway.app`

---

### 2. ☁️ Render (Also Super Easy)

**Steps:**
1. Push code to GitHub
2. Connect Render to your repo
3. Render auto-detects the `render.yaml` config
4. Set `GEMINI_API_KEY` in environment variables
5. Deploy!

**URL**: Your app will be available at `https://your-app.onrender.com`

---

### 3. 🐳 Docker (Local/VPS)

**For local testing:**
```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

**For VPS deployment:**
```bash
# On your server
git clone your-repo
cd HEAL
docker-compose up -d --build
```

---

### 4. 🌐 Vercel + Railway Split

**Frontend (Vercel):**
```bash
cd frontend
npx vercel --prod
```

**Backend (Railway):**
```bash
cd backend
railway init
railway up
```

Then update frontend API URL to point to Railway backend.

---

## Environment Variables Needed

For any deployment, set these environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=production
```

---

## Post-Deployment Checklist

- [ ] Test health endpoint: `https://your-app.com/health`
- [ ] Test API docs: `https://your-app.com/docs`
- [ ] Upload test insurance document
- [ ] Try chat functionality
- [ ] Test bill analysis feature

---

## Troubleshooting

**Common Issues:**

1. **Build fails**: Check that all dependencies are in requirements.txt
2. **Frontend not loading**: Ensure static files are served correctly
3. **API errors**: Verify GEMINI_API_KEY is set correctly
4. **Database issues**: Check file permissions for SQLite

**Logs:**
- Railway: `railway logs`
- Render: Check dashboard logs
- Docker: `docker-compose logs`

---

## Scaling Options

**Free Tiers:**
- Railway: 500 hours/month
- Render: 750 hours/month
- Vercel: Unlimited static hosting

**Paid Upgrades:**
- Railway: $5/month for always-on
- Render: $7/month for always-on
- VPS: $5-20/month depending on specs

---

## Security Notes

- ✅ HTTPS automatically enabled on Railway/Render
- ✅ Environment variables encrypted
- ✅ Database files persistent
- ⚠️ Consider adding authentication for production use
- ⚠️ Set up proper backup strategy for database

---

## Performance Tips

- Use Railway/Render for simplicity
- Consider CDN for static assets in production
- Monitor usage and upgrade plans as needed
- Set up health monitoring alerts
