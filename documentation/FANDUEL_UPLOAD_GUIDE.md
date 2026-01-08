# FanDuel Lineup Upload Guide

## Overview

This guide explains how to generate and upload optimal NBA fantasy lineups to FanDuel using our prediction system.

## Step 1: Download Your FanDuel Player List

1. Go to FanDuel.com and navigate to your NBA contest
2. Click on "Download Player List" or "Upload Lineups"
3. Download the CSV template file (e.g., `FanDuel-NBA-2026-01-02-125122-lineup-upload-template.csv`)

## Step 2: Generate Optimal Lineups

Run the lineup optimizer script with your downloaded CSV:

```bash
./venv/bin/python scripts/modeling/optimize_fanduel_csv.py \
    --csv "/path/to/FanDuel-NBA-2026-01-02-125122-lineup-upload-template.csv" \
    --num-lineups 3
```

### Command Options

- `--csv`: Path to your FanDuel CSV file (required)
- `--num-lineups`: Number of lineups to generate (default: 3)
- `--use-greedy`: Use greedy algorithm instead of ILP (faster but less optimal)
- `--output`: Custom output directory (default: `models/lineups/`)

## Step 3: Review Generated Lineups

The script generates two types of files for each lineup:

### 1. Upload Format (Ready for FanDuel)
**File**: `fanduel_upload_lineup_1.csv`, `fanduel_upload_lineup_2.csv`, etc.

**Format**:
```csv
PG,PG,SG,SG,SF,SF,PF,PF,C
125122-84669:Luka Doncic,125122-84690:Jalen Brunson,125122-171672:AJ Green,...
```

**Important**: The format is `PlayerID:PlayerName` (e.g., `125122-84669:Luka Doncic`). This matches FanDuel's exact requirements and can be **directly uploaded** to FanDuel without any modifications.

### 2. Details Format (For Your Reference)
**File**: `fanduel_lineup_1_details.csv`, `fanduel_lineup_2_details.csv`, etc.

**Contains**:
- Player names
- Teams and opponents
- Salaries
- Predicted fantasy points (FPPG)
- Value (points per $1000)

## Step 4: Upload to FanDuel

1. Go to your FanDuel contest page
2. Click "Upload Lineups"
3. Select one or more `fanduel_upload_lineup_X.csv` files
4. Click "Upload"
5. Review and confirm your lineups

## Understanding the Output

### Console Output

The script displays:
```
LINEUP #1
Position                              Player Team Opponent  Salary  FPPG Value
      PG                  Luka Doncic Doncic  LAL      MEM $12,100 58.32  4.82
      PF Giannis Antetokounmpo Antetokounmpo  MIL      CHA $10,700 51.42  4.81
      ...

Total Salary: $60,000
Total Predicted Points: 274.04
Average Value: 4.43 pts/$1k
```

### Key Metrics

- **Total Salary**: Should be at or near the $60,000 cap
- **Total Predicted Points**: Higher is better
- **Average Value**: Points per $1000 salary (higher = better value)

## Lineup Requirements (FanDuel)

- 2 Point Guards (PG)
- 2 Shooting Guards (SG)
- 2 Small Forwards (SF)
- 2 Power Forwards (PF)
- 1 Center (C)
- **Total**: 9 players
- **Salary Cap**: $60,000

## Optimization Features

### Integer Linear Programming (ILP)

Our optimizer uses ILP to guarantee mathematically optimal lineups:
- **Maximizes** predicted fantasy points
- **Respects** salary cap constraints
- **Ensures** position requirements are met
- **Generates** diverse lineups (each lineup uses different players)

### Player Filtering

The script automatically filters out:
- Players marked as "Out" (O)
- Players marked as "Questionable" (Q)
- Players with $0 salary
- Players with invalid positions

### Multiple Lineups

Generating multiple lineups helps you:
- Diversify risk across different player combinations
- Enter multiple contests with different lineups
- Find value plays you might have missed

## Troubleshooting

### "Not enough players" Error

**Cause**: Too many players are injured or the CSV doesn't have enough valid players.

**Solution**: 
- Download a fresh player list from FanDuel
- Reduce `--num-lineups` to 1 or 2

### "Could not find optimal lineup" Error

**Cause**: ILP solver couldn't find a valid combination within constraints.

**Solution**:
- Check that the CSV has players for all positions
- Try using `--use-greedy` flag as a fallback

### CSV Parsing Error

**Cause**: Unexpected CSV format.

**Solution**:
- Ensure you're using an official FanDuel CSV download
- Check that the file isn't corrupted

## Advanced Usage

### Using Model Predictions

By default, the script uses the FPPG (Fantasy Points Per Game) values from your FanDuel CSV. To use our trained model's predictions instead:

```bash
./venv/bin/python scripts/modeling/optimize_fanduel_csv.py \
    --csv "your_file.csv" \
    --use-model
```

**Note**: This requires matching players to our historical dataset and may take longer.

### Custom Output Directory

```bash
./venv/bin/python scripts/modeling/optimize_fanduel_csv.py \
    --csv "your_file.csv" \
    --output "/path/to/custom/directory"
```

### Greedy Algorithm (Faster)

For quick lineup generation:

```bash
./venv/bin/python scripts/modeling/optimize_fanduel_csv.py \
    --csv "your_file.csv" \
    --use-greedy
```

**Trade-off**: Greedy is faster but may not find the absolute optimal lineup.

## Tips for Success

1. **Generate Multiple Lineups**: Don't put all your eggs in one basket
2. **Review Matchups**: Check the opponent teams in the details file
3. **Monitor Injury Reports**: Re-run the optimizer if injury news breaks
4. **Check Value Players**: Look for high "Value" scores in the details file
5. **Diversify**: Use different lineups for different contests

## Example Workflow

```bash
# 1. Download FanDuel CSV
# (from FanDuel website)

# 2. Generate 5 optimal lineups
cd /path/to/nba_fantasy
./venv/bin/python scripts/modeling/optimize_fanduel_csv.py \
    --csv ~/Downloads/FanDuel-NBA-2026-01-02-125122-lineup-upload-template.csv \
    --num-lineups 5

# 3. Review the lineups
cat models/lineups/fanduel_lineup_*_details.csv

# 4. Upload to FanDuel
# (use fanduel_upload_lineup_1.csv, fanduel_upload_lineup_2.csv, etc.)
```

## Support

If you encounter issues:
1. Check that your CSV is from FanDuel
2. Verify Python environment is activated: `source venv/bin/activate`
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check the console output for specific error messages

## Files Generated

For each lineup (N = 1, 2, 3...):

- `fanduel_upload_lineup_N.csv`: Ready to upload to FanDuel
- `fanduel_lineup_N_details.csv`: Detailed player information for your review

Both files are saved in: `models/lineups/`

---

**Good luck with your FanDuel contests!** 🏀🎉

