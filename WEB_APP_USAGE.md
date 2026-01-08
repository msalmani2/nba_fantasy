# NBA Fantasy Web App - Usage Guide

## 🚀 Quick Start

### Starting the Web App

```bash
cd /Users/Mohammad/Library/CloudStorage/OneDrive-UniversityatBuffalo/nba_fantasy
source venv/bin/activate
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Step-by-Step Guide

### 1. Upload Your FanDuel CSV

1. Download your FanDuel player list from FanDuel.com
2. Click **"Upload FanDuel CSV"** in the sidebar
3. Select your downloaded CSV file
4. The app will automatically:
   - Parse the player data
   - Filter out injured players (O, Q status)
   - Calculate player values

### 2. Review Players

Switch to the **"📋 Players"** tab to:
- View all available players
- Filter by salary range
- Filter by minimum FPPG
- Toggle injured players on/off
- Sort and search players

### 3. Configure Optimization

In the sidebar, set:
- **Number of Lineups**: 1-10 (default: 3)
- **Salary Cap**: Usually $60,000 for FanDuel
- **Optimization Method**: 
  - **ILP (Optimal)** - Mathematically guaranteed best lineups ✅ Recommended
  - **Greedy (Fast)** - Faster but suboptimal

### 4. Generate Lineups

1. Click **"🚀 Generate Optimal Lineups"**
2. Wait for optimization (usually < 2 seconds)
3. View generated lineups in the **"🏀 Lineups"** tab

### 5. Download Lineups

For each lineup, you get **TWO download options**:

#### Option 1: FanDuel Upload Format (📤)
**File**: `fanduel_upload_lineup_1.csv`

**Format**:
```csv
PG,PG,SG,SG,SF,SF,PF,PF,C
125122-84669:Luka Doncic,125122-84690:Jalen Brunson,125122-171672:AJ Green,...
```

✅ **Use this file to upload directly to FanDuel!**

**How to upload**:
1. Go to your FanDuel contest
2. Click "Upload Lineups"
3. Select the `fanduel_upload_lineup_X.csv` file
4. Click "Upload"
5. Done!

#### Option 2: Details Format (📥)
**File**: `lineup_1_details.csv`

**Format**:
```csv
roster_position,player_name,team,opponent,salary,predicted_fantasy_score,value
PG,Luka Doncic,LAL,MEM,12100,58.32,4.82
PF,Giannis Antetokounmpo,MIL,CHA,10700,51.42,4.81
...
```

✅ **Use this for your own analysis and reference**

### 6. Compare Lineups

Switch to the **"📊 Analysis"** tab to:
- View side-by-side comparison charts
- See projected points for each lineup
- See salary usage
- Identify the best and most efficient lineups

## 🎯 Key Features

### FanDuel Upload Format ⭐ NEW!
The app now generates CSV files in the **exact format FanDuel expects**:
- Format: `PlayerID:PlayerName` (e.g., `125122-84669:Luka Doncic`)
- Position headers: `PG,PG,SG,SG,SF,SF,PF,PF,C`
- **No need to edit** - upload directly to FanDuel!

### Player Name Fix ⭐ NEW!
- Fixed duplicate names (was: "Luka Doncic Doncic", now: "Luka Doncic")
- Uses the proper Nickname field from FanDuel CSV

### ILP Optimization
- **Mathematically optimal** lineups
- Considers all 200+ players simultaneously
- Respects:
  - Salary cap ($60,000)
  - Position requirements (2 PG, 2 SG, 2 SF, 2 PF, 1 C)
  - Player availability (filters injured)

### Lineup Diversity
- Each generated lineup uses different players
- Automatic diversity penalty (10%) for repeated players
- Gives you multiple options for different contests

## 💡 Tips for Success

### 1. Generate Multiple Lineups
- Generate 3-5 lineups for different contest entries
- Diversifies risk across different player combinations
- Better chance of hitting the optimal combination

### 2. Review Player Values
- Look for players with high "Value" (FPPG per $1K salary)
- High-value players often outperform expensive stars
- Balance stars with value plays

### 3. Check Matchups
- Review opponent teams in the details CSV
- Look for favorable matchups (weak defenses)
- Consider pace factors (fast-paced games = more opportunities)

### 4. Monitor Injury Status
- The app automatically filters out O and Q players
- Re-run optimization if injury news breaks before contest start
- Keep the FanDuel CSV up-to-date

### 5. Use Both Download Options
- **FanDuel Upload**: For quick submission
- **Details CSV**: For deeper analysis and planning

## 📊 Understanding the Metrics

### Salary
- Total cost of all 9 players
- Must be ≤ $60,000 for FanDuel
- Remaining salary is "wasted" (can't be used)

### Projected Points
- Sum of all players' FPPG (Fantasy Points Per Game)
- Based on FanDuel's historical averages
- Higher = better expected performance

### Average Value
- Mean of (FPPG / Salary * 1000) for all players
- Measures salary efficiency
- Higher = better "bang for your buck"

### Remaining Salary
- Salary cap minus total lineup salary
- Goal: Get as close to $0 as possible
- $200-500 remaining is typical for optimal lineups

## 🔧 Troubleshooting

### "Not enough players" Error
**Problem**: Too few healthy players for valid lineup

**Solutions**:
- Download fresh FanDuel CSV (more players)
- Reduce number of lineups to generate
- Check if many players are marked injured

### CSV Upload Error
**Problem**: App can't parse the CSV

**Solutions**:
- Ensure you downloaded from FanDuel (official format)
- Try re-downloading the CSV from FanDuel
- Check file isn't corrupted

### Slow Optimization
**Problem**: Takes longer than 2 seconds

**Solutions**:
- Reduce number of lineups (e.g., 3 instead of 10)
- Switch to "Greedy (Fast)" method
- Close other applications to free up CPU

### No Players Showing
**Problem**: Player table is empty

**Solutions**:
- Check filter settings (min/max salary, min FPPG)
- Toggle "Show Injured Players" on
- Reset filters to defaults

## 🏆 Best Practices

### Daily Routine
1. **Morning**: Download latest FanDuel CSV (captures overnight news)
2. **Mid-Day**: Run optimizer, generate 3-5 lineups
3. **Evening**: Re-check for injury updates before contest lock
4. **Pre-Lock**: Final optimization with latest information

### Contest Strategy
- **Cash Games** (50/50, Double-Ups): Use lineup #1 (highest projected)
- **GPP/Tournaments**: Use lineups #2-5 (more variance, higher upside)
- **Multiple Entries**: Upload 3-5 different lineups

### Analysis Workflow
1. Generate lineups in web app
2. Download both FanDuel upload + details CSVs
3. Review details CSV for player analysis
4. Upload FanDuel CSV directly to contest
5. Track results and iterate

## 🎨 UI Overview

### Sidebar
- File upload
- Optimization settings
- Generate button
- Quick access to all controls

### Players Tab
- Complete player pool
- Advanced filters
- Sortable columns
- Injury status indicators

### Lineups Tab
- Generated lineup display
- Player breakdown by position
- Salary and projections
- **Two download buttons per lineup**

### Analysis Tab
- Comparison charts
- Best lineup identification
- Efficiency metrics
- Overall statistics

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify FanDuel CSV is up-to-date
3. Try refreshing the web app (Ctrl+R or Cmd+R)
4. Restart the Streamlit app if needed

---

**Enjoy optimizing your NBA fantasy lineups!** 🏀🎉

