# 🏠 Run NBA Fantasy Optimizer Locally

## Why Run Locally?

✅ **No Timeouts**: Unlimited processing time  
✅ **Full Resources**: Use your full CPU/RAM  
✅ **Faster**: Direct file access, no cloud delays  
✅ **Privacy**: All data stays on your machine  
✅ **No Limits**: Process as much data as you want  

---

## 🚀 Quick Start

### 1. Open Terminal

Navigate to your project:
```bash
cd /Users/Mohammad/Library/CloudStorage/OneDrive-UniversityatBuffalo/nba_fantasy
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Open Browser

Your browser should automatically open to:
```
http://localhost:8501
```

If not, manually go to that URL.

---

## 🔑 Setup Kaggle Credentials (One Time)

### Method 1: Using kaggle.json (Recommended)

1. **Get your API key:**
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Place the file:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Done!** The app will automatically find it.

### Method 2: Environment Variables

Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Then:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

---

## 🎮 Using the App Locally

### First Time:
1. App opens in browser at `http://localhost:8501`
2. Click "🔄 Update from Kaggle" (takes 30-60 seconds)
3. Upload your FanDuel CSV
4. Generate lineups
5. Download FanDuel-ready CSV

### Daily Use:
1. Open terminal in project folder
2. Run: `streamlit run app.py`
3. Upload FanDuel CSV (Kaggle data cached for 1 hour)
4. Generate lineups
5. Download and use!

---

## 🛠️ Troubleshooting

### "streamlit: command not found"
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Verify streamlit is installed
pip list | grep streamlit

# If not installed:
pip install -r requirements.txt
```

### "Kaggle credentials not found"
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If not, follow Method 1 above
```

### Port 8501 already in use
```bash
# Kill existing streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run app.py --server.port 8502
```

### Virtual environment issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚡ Performance Tips

### Local is MUCH faster:
- **Kaggle download:** 15-30 seconds (vs 60-90 on cloud)
- **Data processing:** 10-20 seconds (vs timeout on cloud)
- **Optimization:** 1-2 seconds (same)
- **Total:** Under 1 minute to full workflow!

### Caching:
- ML model: Loads once, cached forever (in session)
- Kaggle data: Cached for 1 hour
- No need to reload between lineups!

---

## 🔄 Daily Workflow

### Morning Routine (2 minutes):
```bash
# 1. Start app
cd /Users/Mohammad/Library/CloudStorage/OneDrive-UniversityatBuffalo/nba_fantasy
source venv/bin/activate
streamlit run app.py

# 2. In browser (opens automatically):
# - Click "Update from Kaggle" (once per day)
# - Upload FanDuel CSV
# - Generate 3 lineups
# - Download FanDuel-ready CSVs
# - Upload to FanDuel
# - Win! 💰
```

---

## 🎯 Advantages Over Cloud

| Feature | Cloud | Local |
|---------|-------|-------|
| **Speed** | Slow (health checks) | Fast (full resources) |
| **Kaggle** | Times out | Works perfectly |
| **Data Size** | 1GB limit | Unlimited |
| **Privacy** | Data uploaded | Stays local |
| **Cost** | Free (limited) | Free (unlimited) |
| **Setup** | Complex secrets | Simple kaggle.json |

---

## 🚀 Advanced: Create Launcher Script

Want one-click launch? Create this file:

**File: `launch_app.sh`**
```bash
#!/bin/bash
cd /Users/Mohammad/Library/CloudStorage/OneDrive-UniversityatBuffalo/nba_fantasy
source venv/bin/activate
streamlit run app.py
```

Make it executable:
```bash
chmod +x launch_app.sh
```

Now just double-click or run:
```bash
./launch_app.sh
```

---

## 📱 Access from Phone/Tablet (Optional)

Want to use the app from your phone while it runs on your laptop?

1. **Find your local IP:**
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```
   
2. **Run with network access:**
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

3. **Access from phone:**
   ```
   http://YOUR_IP:8501
   ```
   Example: `http://192.168.1.100:8501`

**Note:** Only works on same WiFi network!

---

## 🛑 Stopping the App

Press `Ctrl+C` in the terminal where streamlit is running.

Or:
```bash
pkill -f streamlit
```

---

## 💡 Pro Tips

1. **Keep terminal open** while using the app
2. **Update Kaggle once per day** (data cached for 1 hour)
3. **Generate multiple lineups** in one session (no need to restart)
4. **Bookmark** `http://localhost:8501` for quick access
5. **Leave app running** - uses minimal resources when idle

---

## 🎉 You're All Set!

Your NBA Fantasy Optimizer is now running locally with:
- ✅ Full ML model support
- ✅ Fast Kaggle data updates
- ✅ No timeouts or limits
- ✅ Complete privacy
- ✅ Unlimited usage

Enjoy! 🏀💰

