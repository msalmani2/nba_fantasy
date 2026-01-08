# 🚀 Deployment Guide - NBA Fantasy Optimizer

## Web App Deployment to Streamlit Cloud

### Step 1: Push to GitHub ✅

Your code is already on GitHub at: `https://github.com/msalmani2/nba_fantasy`

### Step 2: Setup Kaggle API Credentials

#### Option A: Local Development
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json`

#### Option B: Streamlit Cloud Deployment
1. When deploying on Streamlit Cloud, you'll add secrets in the dashboard
2. Format for secrets:

```toml
[kaggle]
username = "your_kaggle_username"
key = "your_kaggle_api_key"
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure:**
   - Repository: `msalmani2/nba_fantasy`
   - Branch: `main`
   - Main file path: `app.py`

5. **Add Secrets** (Click "Advanced settings"):
   ```toml
   [kaggle]
   username = "your_kaggle_username"
   key = "your_kaggle_api_key"
   ```

6. **Click "Deploy"**

7. **Wait 2-3 minutes** for deployment to complete

8. **Your app will be live at:** `https://your-app-name.streamlit.app`

### Step 4: Using the Web App

#### First Time Setup:
1. Click "🔄 Update from Kaggle" in the sidebar to download the latest player database
2. This downloads 1.6M+ player records and may take 30-60 seconds
3. Data is cached for 1 hour

#### Daily Usage:
1. Upload your FanDuel player list CSV
2. The app will use ML predictions if Kaggle data is loaded
3. Generate optimal lineups
4. Download FanDuel-ready CSV files
5. Upload directly to FanDuel!

## Features

### 🤖 ML Predictions
- Uses trained CatBoost model (2.2MB)
- Predicts fantasy scores based on 60+ features
- Falls back to FPPG if model not available

### 📊 Kaggle Integration
- One-click data updates
- Cached for 1 hour (prevents rate limiting)
- Downloads latest historical player data

### 🎯 ILP Optimization
- Mathematically optimal lineups
- Multi-lineup generation with diversity
- Fast: optimizes 200+ players in <2 seconds

### 📤 FanDuel Export
- Direct upload format (`ID:PlayerName`)
- Detailed CSV for analysis
- One-click download

## Troubleshooting

### Issue: "Kaggle credentials not found"
**Solution:** Add Kaggle credentials to Streamlit secrets (see Step 2)

### Issue: "Model file not found"
**Solution:** The model files need to be added to the repository. Run locally:
```bash
git add models/saved/catboost.pkl
git commit -m "Add CatBoost model for deployment"
git push
```

### Issue: "Memory error during optimization"
**Solution:** Reduce number of lineups or use "Greedy (Fast)" method

### Issue: "Kaggle download timeout"
**Solution:** 
- Click "Update from Kaggle" again
- Data is cached, so subsequent loads are instant
- Wait during off-peak hours

## Alternative Deployment Options

### Railway
- Similar to Streamlit Cloud
- More resources available
- Requires `Procfile`

### Heroku
- Free tier available
- Requires `Procfile` and `runtime.txt`
- Good for production

### Render
- Modern platform
- Free tier
- Easy deployment

### ❌ Not Recommended: Vercel
Vercel is optimized for static sites and serverless functions, not persistent servers like Streamlit requires.

## Performance Tips

1. **Cache Management:**
   - Model loads once and stays in memory
   - Kaggle data cached for 1 hour
   - Predictions cached per session

2. **Optimization:**
   - Use ILP for best results (2-3 seconds)
   - Use Greedy for faster results (0.5 seconds)

3. **Data Updates:**
   - Update from Kaggle once per day
   - CSV uploads are instant
   - No need to reload on every optimization

## Cost Considerations

### Free Tier Limits:
- **Streamlit Cloud:** Free for public apps
- **Kaggle API:** Free, unlimited downloads
- **GitHub:** Free for public repos

### Paid Upgrades (Optional):
- **Streamlit Cloud Pro:** $250/month for private apps
- **Railway:** $5/month for more resources
- **Heroku:** $7/month for persistent dynos

## Updates and Maintenance

### Updating the App:
```bash
git add .
git commit -m "Update description"
git push
```

Streamlit Cloud auto-deploys on push!

### Retraining Models:
```bash
python scripts/automated_daily_update.py --full
git add models/saved/catboost.pkl
git commit -m "Update model"
git push
```

### Updating Dependencies:
Edit `requirements.txt` and push:
```bash
git add requirements.txt
git commit -m "Update dependencies"
git push
```

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Review GitHub Actions (if set up)
3. Test locally first: `streamlit run app.py`

## Next Steps

After deployment:
1. ✅ Test the app with sample FanDuel CSV
2. ✅ Verify Kaggle data updates work
3. ✅ Generate test lineups
4. ✅ Download and verify FanDuel upload format
5. ✅ Share your app URL!

---

**Your app will be live at:** `https://[your-app-name].streamlit.app`

Enjoy! 🏀🎉

