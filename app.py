"""
NBA Fantasy Lineup Optimizer - Enhanced Web Interface with ML Models

Features:
- ML model predictions (CatBoost ensemble)
- Kaggle dataset updates
- ILP-based lineup optimization
- FanDuel CSV upload/export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from io import StringIO
import pickle
import os
from datetime import datetime
import kagglehub

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.modeling.optimize_fanduel_csv import parse_positions
from scripts.modeling.ilp_optimizer import optimize_lineup_ilp_fanduel
from scripts.utils.fantasy_scoring import add_fantasy_score_column

# Page config
st.set_page_config(
    page_title="NBA Fantasy Optimizer - ML Enhanced",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .lineup-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_model():
    """Load the ML model (cached for performance)."""
    try:
        model_path = project_root / 'models' / 'saved' / 'catboost.pkl'
        if not model_path.exists():
            return None, "Model file not found. Please train models first."
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_kaggle_data():
    """Load the latest Kaggle dataset (cached for 1 hour)."""
    try:
        # Download latest dataset
        download_path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
        
        # Load player statistics with proper dtype handling
        csv_path = Path(download_path) / "PlayerStatistics.csv"
        if not csv_path.exists():
            return None, "PlayerStatistics.csv not found in downloaded data"
        
        # Load with low_memory=False to handle mixed dtypes
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Add fantasy scores
        df = add_fantasy_score_column(df)
        
        return df, None
    except Exception as e:
        return None, f"Error loading Kaggle data: {str(e)}"


def get_kaggle_credentials():
    """Get Kaggle credentials from kaggle.json, environment, or Streamlit secrets."""
    # Try kaggle.json file first (local development)
    try:
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            import json
            with open(kaggle_json) as f:
                creds = json.load(f)
            return creds
    except Exception as e:
        pass  # Continue to next method
    
    # Try environment variables
    try:
        if 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
            return {
                'username': os.environ['KAGGLE_USERNAME'],
                'key': os.environ['KAGGLE_KEY']
            }
    except Exception:
        pass  # Continue to next method
    
    # Try Streamlit secrets (cloud deployment)
    try:
        if hasattr(st, 'secrets'):
            # This will throw an exception if secrets.toml doesn't exist
            secrets = st.secrets
            if 'kaggle' in secrets:
                return {
                    'username': secrets['kaggle']['username'],
                    'key': secrets['kaggle']['key']
                }
    except Exception:
        pass  # No secrets file, that's okay
    
    return None


def setup_kaggle_credentials():
    """Setup Kaggle credentials in environment."""
    creds = get_kaggle_credentials()
    if creds:
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        return True
    return False


def predict_with_ml(df, model, historical_data):
    """Make predictions using ML model."""
    try:
        # This is a simplified version - you'd need to:
        # 1. Match players in df with historical_data
        # 2. Calculate features for each player
        # 3. Run model predictions
        
        # For now, return FPPG as fallback
        st.warning("⚠️ ML predictions not fully implemented yet. Using FPPG from CSV.")
        return df
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return df


def initialize_session_state():
    """Initialize session state variables."""
    if 'players_df' not in st.session_state:
        st.session_state.players_df = None
    if 'lineups' not in st.session_state:
        st.session_state.lineups = []
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = None
    if 'kaggle_data' not in st.session_state:
        st.session_state.kaggle_data = None
    if 'data_last_updated' not in st.session_state:
        st.session_state.data_last_updated = None


def load_data_from_upload(uploaded_file):
    """Load and process uploaded CSV file."""
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # Try to detect FanDuel upload template format
        try:
            first_row = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")), nrows=1)
            if 'PG' in first_row.columns and 'Instructions' in first_row.columns:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio, skiprows=6)
            else:
                df = pd.read_csv(stringio)
        except:
            df = pd.read_csv(stringio)
        
        df = process_fanduel_data(df)
        return df, None
    except Exception as e:
        return None, str(e)


def process_fanduel_data(df):
    """Process FanDuel CSV data."""
    # Filter out injured players
    if 'Injury Indicator' in df.columns:
        df['is_injured'] = df['Injury Indicator'].apply(
            lambda x: str(x).upper() in ['O', 'Q'] if pd.notna(x) else False
        )
    else:
        df['is_injured'] = False
    
    # Parse positions
    if 'Position' in df.columns:
        df['positions'] = df['Position'].apply(parse_positions)
    elif 'Roster Position' in df.columns:
        df['positions'] = df['Roster Position'].apply(parse_positions)
    else:
        df['positions'] = [[] for _ in range(len(df))]
    
    # Parse salary
    if 'Salary' in df.columns:
        df['salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0)
    else:
        df['salary'] = 0
    
    # Get FPPG
    if 'FPPG' in df.columns:
        df['fppg'] = pd.to_numeric(df['FPPG'], errors='coerce').fillna(0)
        df['predicted_fantasy_score'] = df['fppg']
    else:
        df['fppg'] = 0
        df['predicted_fantasy_score'] = 0
    
    # Create player name
    if 'Nickname' in df.columns:
        df['player_name'] = df['Nickname']
    elif 'First Name' in df.columns and 'Last Name' in df.columns:
        df['player_name'] = df['First Name'] + ' ' + df['Last Name']
    else:
        df['player_name'] = 'Unknown'
    
    # Get team and opponent
    df['team'] = df['Team'] if 'Team' in df.columns else ''
    df['opponent'] = df['Opponent'] if 'Opponent' in df.columns else ''
    
    # Calculate value
    df['value'] = df['predicted_fantasy_score'] / (df['salary'] / 1000)
    
    return df


def display_player_stats(df):
    """Display player statistics and filters."""
    st.subheader("📊 Player Pool Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    
    with col2:
        healthy = (~df['is_injured']).sum() if 'is_injured' in df.columns else len(df)
        st.metric("Healthy Players", healthy)
    
    with col3:
        avg_salary = df['salary'].mean()
        st.metric("Avg Salary", f"${avg_salary:,.0f}")
    
    with col4:
        avg_fppg = df['fppg'].mean()
        st.metric("Avg FPPG", f"{avg_fppg:.1f}")
    
    # Filters
    st.subheader("🔍 Filter Players")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_salary = st.slider("Min Salary", 3500, 15000, 3500, step=500)
    
    with col2:
        max_salary = st.slider("Max Salary", 3500, 15000, 15000, step=500)
    
    with col3:
        min_fppg = st.slider("Min FPPG", 0.0, 70.0, 0.0, step=5.0)
    
    # Apply filters
    filtered_df = df[
        (df['salary'] >= min_salary) &
        (df['salary'] <= max_salary) &
        (df['fppg'] >= min_fppg)
    ]
    
    if 'is_injured' in filtered_df.columns:
        show_injured = st.checkbox("Show Injured Players", value=False)
        if not show_injured:
            filtered_df = filtered_df[~filtered_df['is_injured']]
    
    st.info(f"📋 Showing {len(filtered_df)} players")
    
    return filtered_df


def display_player_table(df):
    """Display interactive player table."""
    st.subheader("📋 Player List")
    
    display_df = df[[
        'player_name', 'positions', 'team', 'opponent', 
        'salary', 'fppg', 'value'
    ]].copy()
    
    if 'is_injured' in df.columns:
        display_df['status'] = df['is_injured'].apply(lambda x: '🤕' if x else '✅')
    
    # Format columns
    display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:,}")
    display_df['fppg'] = display_df['fppg'].apply(lambda x: f"{x:.1f}")
    display_df['value'] = display_df['value'].apply(lambda x: f"{x:.2f}")
    display_df['positions'] = display_df['positions'].apply(lambda x: '/'.join(x) if isinstance(x, list) else x)
    
    # Rename columns
    cols = ['Player', 'Positions', 'Team', 'Opp', 'Salary', 'FPPG', 'Value']
    if 'status' in display_df.columns:
        cols.append('Status')
    display_df.columns = cols
    
    st.dataframe(display_df, use_container_width=True, height=400)


def optimize_lineups(df, num_lineups, salary_cap, use_ilp):
    """Generate optimal lineups."""
    try:
        if 'is_injured' in df.columns:
            df = df[~df['is_injured']].copy()
        
        if use_ilp:
            st.info("🔄 Running ILP optimization (guaranteed optimal)...")
            lineups = optimize_lineup_ilp_fanduel(
                df,
                salary_cap=salary_cap,
                num_lineups=num_lineups,
                diversity_penalty=0.90
            )
        else:
            st.info("🔄 Running greedy optimization (fast but suboptimal)...")
            from scripts.modeling.optimize_fanduel_csv import optimize_fanduel_lineup
            lineups = optimize_fanduel_lineup(
                df,
                salary_cap=salary_cap,
                num_lineups=num_lineups,
                use_model_predictions=False
            )
        
        return lineups, None
    except Exception as e:
        return None, str(e)


def convert_to_fanduel_format(lineup_df):
    """Convert lineup to FanDuel upload format."""
    header = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
    lineup_row = [''] * 9
    
    position_slot_map = {
        'PG': [0, 1],
        'SG': [2, 3],
        'SF': [4, 5],
        'PF': [6, 7],
        'C': [8]
    }
    
    position_counts = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    
    for _, player in lineup_df.iterrows():
        roster_pos = player.get('roster_position', '')
        if roster_pos not in position_slot_map:
            continue
        
        slot_indices = position_slot_map[roster_pos]
        slot_count = position_counts[roster_pos]
        
        if slot_count >= len(slot_indices):
            continue
        
        slot_index = slot_indices[slot_count]
        position_counts[roster_pos] += 1
        
        player_id = player.get('Id', '')
        player_name = player.get('player_name', '')
        lineup_row[slot_index] = f"{player_id}:{player_name}"
    
    lineup_upload_df = pd.DataFrame([lineup_row], columns=header)
    return lineup_upload_df.to_csv(index=False)


def display_lineup(lineup_df, lineup_num):
    """Display a single lineup beautifully."""
    st.markdown(f"""
    <div class="lineup-card">
        <h3>🏀 Lineup #{lineup_num}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate lineup stats
    total_salary = lineup_df['salary'].sum()
    total_points = lineup_df['predicted_fantasy_score'].sum()
    avg_value = lineup_df['value'].mean()
    remaining = 60000 - total_salary
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Salary", f"${total_salary:,}")
    
    with col2:
        st.metric("Projected Points", f"{total_points:.2f}")
    
    with col3:
        st.metric("Avg Value", f"{avg_value:.2f}")
    
    with col4:
        st.metric("Remaining", f"${remaining:,}")
    
    # Display players
    st.markdown("**Players:**")
    
    for idx, row in lineup_df.iterrows():
        pos = row.get('roster_position', '/'.join(row['positions']) if isinstance(row['positions'], list) else 'UTIL')
        name = row['player_name']
        team = row['team']
        opp = row['opponent']
        salary = row['salary']
        fppg = row['predicted_fantasy_score']
        value = row['value']
        
        st.markdown(f"""
        <div class="player-row">
            <b>{pos}</b>: {name} ({team} vs {opp}) - 
            ${salary:,} | {fppg:.1f} pts | {value:.2f} val
        </div>
        """, unsafe_allow_html=True)
    
    # Export buttons
    col1, col2 = st.columns(2)
    
    with col1:
        fanduel_csv = convert_to_fanduel_format(lineup_df)
        st.download_button(
            label=f"📤 FanDuel Upload (Lineup #{lineup_num})",
            data=fanduel_csv,
            file_name=f"fanduel_upload_lineup_{lineup_num}.csv",
            mime="text/csv",
            help="Upload this file directly to FanDuel"
        )
    
    with col2:
        detailed_csv = lineup_df[['roster_position', 'player_name', 'team', 'opponent', 'salary', 'predicted_fantasy_score', 'value']].to_csv(index=False)
        st.download_button(
            label=f"📥 Details (Lineup #{lineup_num})",
            data=detailed_csv,
            file_name=f"lineup_{lineup_num}_details.csv",
            mime="text/csv",
            help="Detailed breakdown for your reference"
        )
    
    st.markdown("---")


def display_lineup_comparison(lineups):
    """Display comparison chart of multiple lineups."""
    if len(lineups) == 0:
        return
    
    st.subheader("📊 Lineup Comparison")
    
    comparison_data = []
    for i, lineup in enumerate(lineups, 1):
        comparison_data.append({
            'Lineup': f'Lineup {i}',
            'Projected Points': lineup['predicted_fantasy_score'].sum(),
            'Total Salary': lineup['salary'].sum(),
            'Avg Value': lineup['value'].mean(),
            'Remaining Salary': 60000 - lineup['salary'].sum()
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            comp_df,
            x='Lineup',
            y='Projected Points',
            title='Projected Points by Lineup',
            color='Projected Points',
            color_continuous_scale='Blues'
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            comp_df,
            x='Lineup',
            y='Total Salary',
            title='Salary Usage by Lineup',
            color='Total Salary',
            color_continuous_scale='Greens'
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def main():
    """Main app function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Fantasy Optimizer - ML Enhanced</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Machine Learning & Integer Linear Programming!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # ML Model Status
        st.subheader("🤖 ML Model")
        if st.session_state.ml_model is None:
            with st.spinner("Loading ML model..."):
                model, error = load_ml_model()
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.ml_model = model
                    st.success("✅ Model loaded!")
        else:
            st.success("✅ Model ready")
        
        # Kaggle Data Update
        st.subheader("📊 Kaggle Data")
        if st.session_state.data_last_updated:
            st.info(f"Last updated: {st.session_state.data_last_updated}")
        
        if st.button("🔄 Update from Kaggle", help="Download latest player data from Kaggle (takes 60-90 seconds)"):
            if not setup_kaggle_credentials():
                st.error("❌ Kaggle credentials not found! Please set up credentials.")
                st.markdown("""
                **Setup Instructions:**
                1. Go to https://www.kaggle.com/settings/account
                2. Click "Create New API Token"
                3. Add to Streamlit Settings → Secrets:
                ```
                [kaggle]
                username = "your_username"
                key = "your_key"
                ```
                4. Save and restart app
                """)
            else:
                st.info("⏳ Downloading 52MB dataset... This takes 60-90 seconds. Please wait...")
                try:
                    kaggle_data, error = load_kaggle_data()
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.kaggle_data = kaggle_data
                        st.session_state.data_last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.success(f"✅ Loaded {len(kaggle_data):,} player records!")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.warning("💡 You can still use the app without Kaggle data. Just upload your FanDuel CSV!")
        
        st.markdown("---")
        
        # File upload
        st.subheader("📁 Upload Players")
        uploaded_file = st.file_uploader(
            "Upload FanDuel CSV",
            type=['csv'],
            help="Upload the FanDuel player list CSV file"
        )
        
        if uploaded_file is not None:
            df, error = load_data_from_upload(uploaded_file)
            if error:
                st.error(f"Error loading file: {error}")
            else:
                # Use ML model if available
                if st.session_state.ml_model and st.session_state.kaggle_data is not None:
                    df = predict_with_ml(df, st.session_state.ml_model, st.session_state.kaggle_data)
                
                st.session_state.players_df = df
                st.success(f"✅ Loaded {len(df)} players!")
        
        st.markdown("---")
        
        # Optimization settings
        st.subheader("🎯 Optimization")
        
        num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of diverse lineups to generate"
        )
        
        salary_cap = st.number_input(
            "Salary Cap",
            min_value=50000,
            max_value=70000,
            value=60000,
            step=1000,
            help="Maximum total salary for lineup"
        )
        
        use_ilp = st.radio(
            "Optimization Method",
            ["ILP (Optimal)", "Greedy (Fast)"],
            index=0,
            help="ILP finds provably optimal lineups"
        ) == "ILP (Optimal)"
        
        st.markdown("---")
        
        # Optimize button
        if st.button("🚀 Generate Optimal Lineups", type="primary"):
            if st.session_state.players_df is None:
                st.error("Please upload a player list first!")
            else:
                with st.spinner("Optimizing lineups..."):
                    lineups, error = optimize_lineups(
                        st.session_state.players_df,
                        num_lineups,
                        salary_cap,
                        use_ilp
                    )
                    
                    if error:
                        st.error(f"Optimization failed: {error}")
                    else:
                        st.session_state.lineups = lineups
                        st.session_state.optimization_done = True
                        st.success(f"✅ Generated {len(lineups)} optimal lineups!")
    
    # Main content
    if st.session_state.players_df is None:
        # Welcome screen
        st.info("👈 Upload a FanDuel CSV file to get started!")
        
        st.markdown("""
        ### 🚀 Quick Start (No Setup Required!):
        
        1. **Upload** your FanDuel player list CSV  
        2. **Generate** optimal lineups using ILP  
        3. **Download** FanDuel-ready CSV  
        4. **Upload** to FanDuel and win! 💰
        
        ### ✨ Features:
        
        - **🎯 ILP Optimization**: Mathematically optimal lineups (guaranteed best!)
        - **⚡ Fast**: Optimizes 200+ players in 2-3 seconds
        - **📤 FanDuel Ready**: Direct CSV export in correct format
        - **🤕 Smart**: Automatically filters injured players
        - **🤖 ML Model**: CatBoost predictions (uses FPPG from your CSV)
        
        ### 📊 Optional: Kaggle Integration
        
        Want ML predictions on historical data?  
        1. Get API key: https://www.kaggle.com/settings/account  
        2. Add to Settings → Secrets:
        ```
        [kaggle]
        username = "your_username"
        key = "your_api_key"
        ```
        3. Click "🔄 Update from Kaggle" (takes 60-90 seconds)
        
        **Note:** Kaggle is optional - the app works perfectly without it!
        """)
        
    else:
        # Display tabs
        tab1, tab2, tab3 = st.tabs(["📋 Players", "🏀 Lineups", "📊 Analysis"])
        
        with tab1:
            filtered_df = display_player_stats(st.session_state.players_df)
            display_player_table(filtered_df)
        
        with tab2:
            if st.session_state.optimization_done and len(st.session_state.lineups) > 0:
                st.markdown('<div class="success-box">✅ Optimization Complete!</div>', unsafe_allow_html=True)
                st.markdown("")
                
                for i, lineup in enumerate(st.session_state.lineups, 1):
                    display_lineup(lineup, i)
            else:
                st.info("👈 Click 'Generate Optimal Lineups' in the sidebar to create lineups!")
        
        with tab3:
            if st.session_state.optimization_done and len(st.session_state.lineups) > 0:
                display_lineup_comparison(st.session_state.lineups)
                
                st.subheader("📈 Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Best Lineup:**")
                    best_idx = max(range(len(st.session_state.lineups)), 
                                  key=lambda i: st.session_state.lineups[i]['predicted_fantasy_score'].sum())
                    best_points = st.session_state.lineups[best_idx]['predicted_fantasy_score'].sum()
                    st.metric("Lineup", f"#{best_idx + 1}", f"{best_points:.2f} pts")
                
                with col2:
                    st.markdown("**Most Efficient:**")
                    efficient_idx = max(range(len(st.session_state.lineups)),
                                      key=lambda i: st.session_state.lineups[i]['value'].mean())
                    efficient_val = st.session_state.lineups[efficient_idx]['value'].mean()
                    st.metric("Lineup", f"#{efficient_idx + 1}", f"{efficient_val:.2f} val")
                    
            else:
                st.info("Generate lineups first to see analysis!")


if __name__ == "__main__":
    main()

