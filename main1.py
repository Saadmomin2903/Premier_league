import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
from joblib import load
import catboost
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'pipe.joblib'
DB_PARAMS = {
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "Qx9nHGmey4fX",
    "host": "ep-muddy-hill-a1wfdgdb.ap-southeast-1.aws.neon.tech",
    "port": "5432",
    "sslmode": "require"
}
WINDOW_SIZE = 5

# Load the pipeline
@st.cache_resource
def load_model(model_path: str):
    try:
        return load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

pipe = load_model(MODEL_PATH)

# Set up the Streamlit page
st.set_page_config(page_title="Premier League PREDICTOR", page_icon="⚽")
st.markdown(
    """
    <style>
    body {
        background-image: url("https://wallpapercave.com/wp/wp4059913.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Premier League PREDICTOR")

# Define teams
H_team = sorted([
    'Fulham', 'West Ham United', 'Aston Villa', 'Liverpool', 'Everton',
    'Manchester United', 'Southampton', 'Crystal Palace', 'Chelsea',
    'Newcastle United', 'Tottenham', 'Manchester City', 'Arsenal',
    'Leicester City', 'Bournemouth', 'Brighton',
    'Wolverhampton Wanderers', 'Leeds United', 'Brentford',
    'Nottingham Forest'
])
A_team = sorted([
    'Sunderland', 'Cardiff City', 'Arsenal', 'Stoke City',
    'Norwich City', 'Swansea City', 'West Brom', 'Tottenham',
    'Hull City', 'Manchester City', 'Chelsea', 'Aston Villa',
    'Newcastle Utd', 'Manchester Utd', 'Liverpool', 'Crystal Palace',
    'Everton', 'Southampton', 'West Ham', 'Fulham', 'Leicester City',
    'Burnley', 'Other', 'Bournemouth', 'Watford', 'Brighton',
    'Huddersfield', 'Wolves', 'Sheffield Utd', 'Leeds United',
    'Brentford'
])

# User inputs for team selection
col1, col2 = st.columns(2)
with col1:
    HomeTeam = st.selectbox("Select Home Team", H_team)
with col2:
    AwayTeam = st.selectbox("Select Away Team", A_team)

# Database functions
def connect_to_db(params: Dict[str, Any]):
    try:
        return psycopg2.connect(**params)
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        st.error(f"Database connection error: {str(e)}")
        return None

def fetch_data_from_db(conn) -> pd.DataFrame:
    query = """
    SELECT "GF", "GA", "home_team", "Opponent", "Result", "Date"
    FROM preprocessed_stats
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Data processing functions
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['home_team', 'Date'])
    
    for stat in ['GF', 'GA']:
        df[f'Rolling_{stat}_Home'] = df.groupby('home_team')[stat].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
        df[f'Rolling_{stat}_Away'] = df.groupby('Opponent')[stat].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
    
    for result, column in zip([1, 0, -1], ['Wins', 'Draws', 'Losses']):
        df[f'Home_{column}_Form'] = df.groupby('home_team')['Result'].transform(lambda x: calculate_form(x, result))
        df[f'Away_{column}_Form'] = df.groupby('Opponent')['Result'].transform(lambda x: calculate_form(x, result))
    
    return df

def calculate_form(series: pd.Series, result: int) -> pd.Series:
    return series.rolling(window=WINDOW_SIZE, min_periods=1).apply(lambda x: (x == result).sum(), raw=False)

def get_statistic_for_team(df: pd.DataFrame, team_name: str, stat_name: str):
    if stat_name not in df.columns:
        logger.error(f"Statistic '{stat_name}' is not a valid column in the DataFrame.")
        return None

    team_column = 'home_team' if 'Home' in stat_name else 'Opponent'
    team_stats = df[df[team_column] == team_name]
    return team_stats[stat_name].iloc[-1] if not team_stats.empty else None

# Prediction function
def predict_match(pipe, home_team: str, away_team: str, df: pd.DataFrame) -> Dict[str, float]:
    features = [
        "Rolling_GF_Home", "Rolling_GA_Home", "Rolling_GF_Away", "Rolling_GA_Away",
        "Home_Wins_Form", "Home_Draws_Form", "Home_Losses_Form",
        "Away_Wins_Form", "Away_Draws_Form", "Away_Losses_Form"
    ]
    
    final = pd.DataFrame({
        "home_team": [home_team],
        "Opponent": [away_team],
        **{feat: [get_statistic_for_team(df, home_team if 'Home' in feat else away_team, feat)] for feat in features}
    })

    if final.isnull().values.any():
        logger.error("Error: Some statistics are missing for the selected teams.")
        return {}

    try:
        result = pipe.predict_proba(final)
        return {
            "home_win": result[0, 2],
            "draw": result[0, 1],
            "away_win": result[0, 0]
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {}

# Main execution
if st.button('Predict Probability'):
    with st.spinner('Fetching and processing data...'):
        conn = connect_to_db(DB_PARAMS)
        if conn:
            df = fetch_data_from_db(conn)
            conn.close()
            if not df.empty:
                df = process_data(df)
                probabilities = predict_match(pipe, HomeTeam, AwayTeam, df)
                if probabilities:
                    st.success("Prediction successful!")
                    st.text(f"{HomeTeam} Win Probability: {round(probabilities['home_win'] * 100)}%")
                    st.text(f"Draw Probability: {round(probabilities['draw'] * 100)}%")
                    st.text(f"{AwayTeam} Win Probability: {round(probabilities['away_win'] * 100)}%")
                else:
                    st.error("Unable to make prediction. Please try again.")
            else:
                st.error("No data available for prediction.")
        else:
            st.error("Unable to connect to the database. Please try again later.")
