# ⚡ Quick Start Guide

## 🌐 Deploy to Streamlit Cloud (5 minutes)

Your code is ready at: **https://github.com/msalmani2/nba_fantasy**

### Step 1: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - Repository: `msalmani2/nba_fantasy`
   - Branch: `main`
   - Main file: `app.py`
5. Click **"Deploy"**

### Step 2: Add Kaggle Credentials (Optional but Recommended)

Click "Advanced settings" → "Secrets" and add:

```toml
[kaggle]
username = "your_kaggle_username"
key = "your_kaggle_api_key"
```

**Get your Kaggle API key:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Copy the username and key from the JSON file

### Step 3: Use the App!

Your app will be live at: `https://[your-app-name].streamlit.app`

**First time:**
1. Click "🔄 Update from Kaggle" (if you added credentials)
2. Upload FanDuel player CSV
3. Generate optimal lineups
4. Download FanDuel-ready CSV
5. Upload to FanDuel!

## 💻 Run Locally

```bash
# Clone repository
git clone https://github.com/msalmani2/nba_fantasy.git
cd nba_fantasy

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle credentials (optional)
# Place kaggle.json in ~/.kaggle/

# Run app
streamlit run app.py
```

Open http://localhost:8501

## 📱 Using the Web App

### Upload Players
1. Download player list from FanDuel
2. Click "Upload FanDuel CSV" in sidebar
3. Review players and stats

### Generate Lineups
1. Adjust settings:
   - Number of lineups (1-10)
   - Salary cap ($60,000 default)
   - Optimization method (ILP recommended)
2. Click "🚀 Generate Optimal Lineups"
3. Wait 2-3 seconds

### Download & Upload
1. Click "📤 FanDuel Upload" button
2. Save the CSV file
3. Go to FanDuel contest
4. Upload the CSV directly
5. Done! 🎉

## 🤖 ML Features

### Automatic Updates
- Click "🔄 Update from Kaggle" in sidebar
- Downloads latest 1.6M+ player records
- Cached for 1 hour
- Uses ML model for predictions

### ML Predictions vs FPPG
- **With Kaggle data:** Uses trained CatBoost model
- **Without Kaggle data:** Uses FanDuel FPPG (still works great!)
- Model trained on 60+ features

## 🎯 Optimization Methods

### ILP (Integer Linear Programming)
- ✅ Mathematically optimal
- ✅ Proven best solution
- ⏱️ 2-3 seconds
- 🏆 Recommended

### Greedy Algorithm
- ⚡ Very fast
- ⏱️ 0.5 seconds
- 📊 Good but not optimal
- 🎯 Use for quick iterations

## 📊 Output Files

### FanDuel Upload CSV
- Format: `ID:PlayerName`
- Direct upload to FanDuel
- Positions: PG, PG, SG, SG, SF, SF, PF, PF, C

### Details CSV
- Full player information
- Salary, projections, value
- Team, opponent, position
- For your analysis

## ⚠️ Important Notes

### Vercel Won't Work
Vercel is for static sites, not Streamlit apps. Use **Streamlit Cloud** instead (it's free!).

### Model Files
The app includes a 2.2MB CatBoost model. If deployment fails due to size, remove model files and app will use FPPG.

### Kaggle Rate Limits
- Update once per day
- Data is cached for 1 hour
- Don't spam the update button

## 🆘 Troubleshooting

### "Kaggle credentials not found"
Add credentials to Streamlit secrets (see Step 2 above)

### "Model file not found"
The model is included in the repo. Try redeploying or use FPPG mode.

### Lineups don't match salary cap
Check your salary cap setting in sidebar. Default is $60,000 for FanDuel.

### Players show as injured
The app filters injured players automatically. Toggle "Show Injured Players" to see them.

## 📚 More Info

- **Full Deployment Guide:** See `DEPLOYMENT_GUIDE.md`
- **Web App Usage:** See `WEB_APP_USAGE.md`
- **FanDuel Upload:** See `documentation/FANDUEL_UPLOAD_GUIDE.md`
- **Project Documentation:** See `documentation/` folder

## 🎉 You're Ready!

That's it! Your NBA Fantasy Optimizer is ready to deploy and use.

**Deploy now:** https://share.streamlit.io

Good luck with your fantasy lineups! 🏀💰

