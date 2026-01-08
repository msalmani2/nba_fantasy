# NBA Fantasy Lineup Optimizer - Web App Guide

## 🚀 Quick Start

### Start the Web App

```bash
cd /path/to/nba_fantasy
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 🎯 Features

### 1. **CSV Upload**
- Upload FanDuel player list CSV
- Automatic player filtering
- Injury status detection

### 2. **Interactive Player Pool**
- View all available players
- Filter by:
  - Salary range
  - Minimum FPPG
  - Injury status
- Sortable columns
- Real-time statistics

### 3. **ILP Optimization**
- **Provably optimal** lineups
- Generate 1-10 diverse lineups
- Adjustable salary cap
- Choice of ILP (optimal) or Greedy (fast)

### 4. **Beautiful Lineup Display**
- Detailed lineup breakdown
- Position assignments
- Salary tracking
- Projected points
- Value metrics

### 5. **Comparison Charts**
- Visual lineup comparison
- Points projection charts
- Salary usage analysis
- Efficiency metrics

### 6. **Export Functionality**
- Download lineups as CSV
- Ready for FanDuel upload
- Individual or batch export

---

## 📖 How to Use

### Step 1: Upload Players
1. Click "Browse files" in sidebar
2. Select your FanDuel CSV export
3. Wait for confirmation

### Step 2: Review Players
1. Switch to "Players" tab
2. Use filters to narrow down pool
3. Check player statistics

### Step 3: Configure Optimization
1. Set number of lineups (1-10)
2. Adjust salary cap if needed
3. Choose ILP (optimal) or Greedy (fast)

### Step 4: Generate Lineups
1. Click "Generate Optimal Lineups"
2. Wait a few seconds
3. View results in "Lineups" tab

### Step 5: Analyze & Export
1. Switch to "Analysis" tab for comparisons
2. Review each lineup in "Lineups" tab
3. Download your favorite lineups

---

## 🎨 Interface Overview

### Sidebar (Left)
- **File Upload**: Upload player CSV
- **Settings**: Configure optimization
- **Action Button**: Generate lineups

### Main Area (Center)
Three tabs:
1. **📋 Players**: View and filter player pool
2. **🏀 Lineups**: View generated lineups
3. **📊 Analysis**: Compare lineup performance

### Player Tab Features
- Total players count
- Healthy players count
- Average salary
- Average FPPG
- Filter sliders
- Sortable table

### Lineups Tab Features
- Individual lineup cards
- Position breakdowns
- Salary tracking
- Projected points
- Download buttons

### Analysis Tab Features
- Comparison bar charts
- Best lineup identification
- Most efficient lineup
- Statistical summaries

---

## 💡 Tips & Tricks

### 1. Finding Value Players
```
1. Go to Players tab
2. Sort by "Value" column (descending)
3. Look for players with value > 4.5
4. Check injury status
```

### 2. Generating Diverse Lineups
```
1. Set num_lineups to 5-10
2. Use ILP optimization
3. System automatically creates diverse options
4. Compare in Analysis tab
```

### 3. Budget Optimization
```
1. Set salary cap to 59000 (leave $1000 buffer)
2. Generate lineups
3. Look for lineups with highest points/salary ratio
```

### 4. Risk Management
```
1. Filter out injured players (default)
2. Set min FPPG to 15+ for consistency
3. Check player matchups (team vs opponent)
```

### 5. Quick Tournament Lineups
```
1. Upload CSV
2. Set lineups to 10
3. Use ILP optimization
4. Download all lineups
5. Enter multiple lineups in tournament
```

---

## 🔧 Troubleshooting

### Issue: "No optimal solution found"
**Solution**: 
- Check that you have enough healthy players
- Lower minimum FPPG filter
- Increase salary cap slightly

### Issue: "Upload failed"
**Solution**:
- Ensure CSV is from FanDuel export
- Check file isn't corrupted
- Try re-exporting from FanDuel

### Issue: "Optimization taking too long"
**Solution**:
- Switch to Greedy algorithm
- Reduce number of lineups
- Filter player pool first

### Issue: "Missing positions in lineup"
**Solution**:
- Check Position column in CSV
- Ensure enough players per position
- Verify CSV format matches FanDuel

---

## 📊 Understanding Metrics

### FPPG (Fantasy Points Per Game)
- Average fantasy points from FanDuel
- Higher = better expected performance
- Based on season-to-date stats

### Value (Points per $1k)
- FPPG divided by (Salary / 1000)
- Measures efficiency
- Target: > 4.0 is good, > 5.0 is excellent

### Projected Points
- Sum of all player FPPG in lineup
- Your lineup's expected total
- Higher = better

### Remaining Salary
- Budget left unused
- $0-$500 is optimal
- Too much = missed opportunity

---

## 🎯 Optimization Methods Explained

### ILP (Integer Linear Programming)
✅ **Pros**:
- Mathematically proven optimal
- Finds absolute best lineup
- Handles complex constraints
- Diverse lineup generation

❌ **Cons**:
- Slightly slower (1-3 seconds)
- Requires PuLP library

**When to use**: Always for final lineups

### Greedy Algorithm
✅ **Pros**:
- Very fast (<0.5 seconds)
- Simple to understand
- Good enough for quick tests

❌ **Cons**:
- Not optimal (misses 1-2% points)
- May get stuck in local maximum

**When to use**: Quick testing only

---

## 📱 Mobile Access

The web app works on mobile browsers!

```
1. Start app on your computer
2. Find your local IP (ipconfig/ifconfig)
3. Access from phone: http://YOUR_IP:8501
4. Use same interface on phone
```

---

## 🚀 Advanced Usage

### Running on Different Port
```bash
streamlit run app.py --server.port 8080
```

### Running in Production Mode
```bash
streamlit run app.py --server.headless true
```

### Allowing External Connections
```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 🎨 Customization

### Changing Theme
1. Click ⋮ menu (top-right)
2. Select "Settings"
3. Choose theme
4. Or edit `.streamlit/config.toml`

### Adjusting Layout
Edit `app.py` and modify:
```python
st.set_page_config(
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)
```

---

## 📈 Performance Tips

### For Large Player Pools (500+)
1. Pre-filter players before upload
2. Use Greedy for initial testing
3. Switch to ILP for final lineups
4. Reduce number of lineups generated

### For Multiple Contests
1. Generate 10 lineups at once
2. Export all to CSV
3. Use different lineups per contest
4. Diversify your entries

---

## 🔐 Privacy & Security

- ✅ All processing happens locally
- ✅ No data sent to external servers
- ✅ CSV files stay on your machine
- ✅ No account or login required

---

## 📞 Support

### Common Questions

**Q: Can I use DraftKings CSV?**
A: Yes! The format is similar. May need minor adjustments.

**Q: Does it account for injuries?**
A: Yes! Automatically filters out O/Q players.

**Q: How accurate are projections?**
A: Based on FPPG from FanDuel, very reliable for averages.

**Q: Can I adjust player projections?**
A: Not in web interface yet. Use Python scripts for custom predictions.

**Q: How many lineups should I generate?**
A: 3-5 for cash games, 10+ for tournaments.

---

## 🎓 Best Practices

### Cash Games (50/50, Head-to-Head)
- Generate 3-5 lineups
- Focus on consistency (high floor)
- Filter min FPPG to 20+
- Use players with value > 4.5
- Check for injuries carefully

### Tournaments (GPP)
- Generate 10+ lineups
- Include some risk (lower floor, high ceiling)
- Mix in cheaper players
- Diversify across multiple games
- Consider contrarian plays

### Daily Workflow
1. Download FanDuel CSV (morning)
2. Upload to web app
3. Generate lineups
4. Cross-reference with news
5. Make final adjustments
6. Submit before lock

---

## 🏆 Success Metrics

Track your performance:
- Win rate in cash games
- ROI (return on investment)
- Average placement in GPPs
- Actual vs projected points

Recommended: Keep a spreadsheet of:
- Date
- Contest type
- Lineup used
- Projected points
- Actual points
- Winnings

---

**Enjoy building optimal lineups! 🏀🚀**

*Web interface makes ILP optimization accessible to everyone!*

