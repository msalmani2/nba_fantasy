# 🏀 NBA Fantasy Lineup Optimizer

An advanced machine learning system that predicts NBA players' FanDuel fantasy scores and generates mathematically optimal lineups using Integer Linear Programming (ILP).

## ✨ Features

- 🎯 **ILP Optimization**: Provably optimal lineup generation (not just greedy)
- 📊 **ML Predictions**: Ensemble models (Random Forest, XGBoost, LightGBM, CatBoost)
- 🌐 **Web Interface**: Beautiful Streamlit app with drag-and-drop CSV upload
- 📤 **FanDuel Ready**: Direct CSV export in FanDuel upload format
- 🤕 **Smart Filtering**: Automatic injury detection and filtering
- 📈 **Analytics**: Interactive charts and lineup comparisons

## 🚀 Quick Start

### Web App (Recommended)

```bash
# Clone the repository
git clone https://github.com/msalmani2/nba_fantasy.git
cd nba_fantasy

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser!

### Command Line

```bash
# Generate 3 optimal lineups from FanDuel CSV
python scripts/modeling/optimize_fanduel_csv.py \
    --csv path/to/fanduel_players.csv \
    --num-lineups 3
```

## Project Overview

This project combines machine learning predictions with mathematical optimization to create optimal FanDuel fantasy basketball lineups. It uses historical player statistics from Kaggle and implements multiple advanced models combined through ensemble techniques.

## Project Structure

```
nba_fantasy/
├── data/
│   ├── raw/              # Original downloaded datasets
│   ├── processed/        # Cleaned and feature-engineered data
│   └── external/         # Additional external datasets if needed
├── scripts/
│   ├── temp/             # Temporary/experimental scripts
│   ├── data_processing/  # Data loading, cleaning, feature engineering
│   ├── modeling/         # Model training and evaluation scripts
│   └── utils/            # Utility functions
├── notebooks/
│   ├── exploration/      # EDA notebooks
│   └── modeling/         # Model development notebooks
├── models/
│   ├── saved/            # Trained model files
│   └── predictions/      # Model predictions
├── documentation/
│   ├── project_overview.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   └── results/
├── readmes/
│   └── step_by_step/     # Generated readme files for each phase
├── config/
│   └── config.yaml       # Configuration file for hyperparameters
└── tests/
    └── unit/              # Unit tests
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your configuration values if needed.

## Usage

### Data Loading

```python
from scripts.data_processing.load_data import load_player_statistics

df = load_player_statistics()
```

### Model Training

```python
from scripts.modeling.train_models import train_all_models

models = train_all_models()
```

### Making Predictions

**For all players:**
```python
from scripts.modeling.predict import predict_fantasy_scores

predictions = predict_fantasy_scores(new_data)
```

**For specific teams:**
```bash
# Command line
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors Celtics --top 20

# Interactive
python scripts/modeling/predict_by_teams_simple.py
```

**For next game predictions:**
```bash
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors --next-game
```

**Evaluate on recent games:**
```bash
python scripts/modeling/evaluate_recent_games.py --days 7
```

See `documentation/team_predictions_guide.md` for detailed usage.

### Fantasy Salary Data

The NBA API does not provide fantasy salaries. See `documentation/fantasy_salary_sources.md` for:
- Free salary data sources (RickRunGood)
- Paid API options (SportsDataIO, Sportradar, etc.)
- How to integrate salary data for lineup optimization

## Dataset

The project uses the Kaggle dataset: **eoinamoore/historical-nba-data-and-player-box-scores**

Main dataset file: `PlayerStatistics.csv`

## Methodology

1. **Data Exploration**: Comprehensive analysis of player statistics
2. **FanDuel Scoring**: Research and implementation of FanDuel scoring system
3. **Feature Engineering**: Creation of temporal, player-specific, and game context features
4. **Model Development**: Implementation of baseline and advanced models (Random Forest, XGBoost, LightGBM, CatBoost)
5. **Ensemble Methods**: Combining models through averaging, stacking, and blending
6. **Evaluation**: Comprehensive model evaluation and error analysis

## Success Metrics

- Target MAE < 5 fantasy points
- Target RMSE < 7 fantasy points
- Ensemble model outperforms individual models

## 🌐 Deployment

### Streamlit Cloud (Recommended - Free!)

1. Push your code to GitHub (see below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `msalmani2/nba_fantasy`
6. Main file path: `app.py`
7. Click "Deploy"!

Your app will be live at: `https://[your-app-name].streamlit.app`

### Alternative Platforms

- **Heroku**: Good for production deployments
- **Railway**: Modern platform with free tier
- **Render**: Simple deployment with free tier

**Note**: Vercel is not recommended for Streamlit apps (requires persistent server).

## 📚 Documentation

See the `documentation/` folder for detailed documentation:
- `project_overview.md`: Project overview and goals
- `data_dictionary.md`: Dataset schema and column descriptions
- `fanduel_scoring.md`: FanDuel scoring system documentation
- `methodology.md`: Detailed methodology and approach
- `features.md`: Feature engineering documentation
- `WEB_APP_USAGE.md`: Complete guide for using the web app
- `FANDUEL_UPLOAD_GUIDE.md`: How to generate and upload lineups

## 🎯 Key Components

### Machine Learning Pipeline
- Ensemble models (RF, XGBoost, LightGBM, CatBoost)
- Advanced feature engineering (60+ features)
- Temporal, player-specific, and game context features
- Confidence intervals and risk assessment

### Optimization
- Integer Linear Programming (ILP) using PuLP
- Guaranteed optimal lineups (not greedy)
- Multi-lineup generation with diversity
- Position and salary cap constraints

### Web Interface
- Interactive player filtering
- Real-time optimization
- Lineup comparison charts
- FanDuel-ready CSV export

## 📄 License

MIT License

## 👤 Author

Mohammad Salmani
- GitHub: [@msalmani2](https://github.com/msalmani2)

