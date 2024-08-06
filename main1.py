import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from joblib import load
import catboost


# Load the pipeline
pipe = load('pipe.joblib')

# Your NeonDB URL
db_url = "postgresql://neondb_owner:Qx9nHGmey4fX@ep-muddy-hill-a1wfdgdb.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

# Create an engine to connect to the PostgreSQL database
engine = create_engine(db_url)

# Set up the Streamlit page
page_by_img = '''
<style>
body {
    background-image: url("https://wallpapercave.com/wp/wp4059913.jpg");
    background-size: cover;
}
</style>
'''
st.markdown(page_by_img, unsafe_allow_html=True)
st.title("Premire League PREDICTOR")

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

# Load data from the database
query = """
SELECT "GF", "GA", "home_team", "Opponent", "Result", "Date"
FROM preprocessed_stats
"""

df = pd.read_sql(query, engine)

df = pd.read_sql(query, engine)
df['Date'] = pd.to_datetime(df['Date'])

# Process data
df = df.sort_values(by=['home_team', 'Date'])
df['Rolling_GF_Home'] = df.groupby('home_team')['GF'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['Rolling_GA_Home'] = df.groupby('home_team')['GA'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['Rolling_GF_Away'] = df.groupby('Opponent')['GF'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['Rolling_GA_Away'] = df.groupby('Opponent')['GA'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

window_size = 5

def calculate_form(series, result):
    return series.rolling(window=window_size, min_periods=1).apply(lambda x: (x == result).sum(), raw=False)

df['Home_Wins_Form'] = df.groupby('home_team')['Result'].transform(lambda x: calculate_form(x, 1))
df['Home_Draws_Form'] = df.groupby('home_team')['Result'].transform(lambda x: calculate_form(x, 0))
df['Home_Losses_Form'] = df.groupby('home_team')['Result'].transform(lambda x: calculate_form(x, -1))
df['Away_Wins_Form'] = df.groupby('Opponent')['Result'].transform(lambda x: calculate_form(x, 1))
df['Away_Draws_Form'] = df.groupby('Opponent')['Result'].transform(lambda x: calculate_form(x, 0))
df['Away_Losses_Form'] = df.groupby('Opponent')['Result'].transform(lambda x: calculate_form(x, -1))

def get_statistic_for_team(team_name, stat_name):
    if stat_name not in df.columns:
        st.error(f"Statistic '{stat_name}' is not a valid column in the DataFrame.")
        return None

    # Filter the DataFrame for the specific team and get the latest statistic
    if stat_name in ['Rolling_GF_Home', 'Rolling_GA_Home', 'Home_Wins_Form', 'Home_Draws_Form', 'Home_Losses_Form']:
        team_stats = df[df['home_team'] == team_name]
    else:
        team_stats = df[df['Opponent'] == team_name]

    latest_stat = team_stats[stat_name].iloc[-1] if not team_stats.empty else None
    return latest_stat

if st.button('Predict Probability'):
    final = pd.DataFrame({
        "home_team": [HomeTeam],  # Changed from "HomeTeam"
        "Opponent": [AwayTeam],   # Changed from "AwayTeam"
        "Rolling_GF_Home": [get_statistic_for_team(HomeTeam, "Rolling_GF_Home")],
        "Rolling_GA_Home": [get_statistic_for_team(HomeTeam, "Rolling_GA_Home")],
        "Rolling_GF_Away": [get_statistic_for_team(AwayTeam, "Rolling_GF_Away")],
        "Rolling_GA_Away": [get_statistic_for_team(AwayTeam, "Rolling_GA_Away")],
        "Home_Wins_Form": [get_statistic_for_team(HomeTeam, "Home_Wins_Form")],
        "Home_Draws_Form": [get_statistic_for_team(HomeTeam, "Home_Draws_Form")],
        "Home_Losses_Form": [get_statistic_for_team(HomeTeam, "Home_Losses_Form")],
        "Away_Wins_Form": [get_statistic_for_team(AwayTeam, "Away_Wins_Form")],
        "Away_Draws_Form": [get_statistic_for_team(AwayTeam, "Away_Draws_Form")],
        "Away_Losses_Form": [get_statistic_for_team(AwayTeam, "Away_Losses_Form")]
    })

    if final.isnull().values.any():
        st.error("Error: Some statistics are missing for the selected teams.")
    else:
        result = pipe.predict_proba(final)
        home_win_proba = result[0, 2]  # Probability of Home Win (1)
        draw_proba = result[0, 1]  # Probability of Draw (0)
        away_win_proba = result[0, 0]  # Probability of Away Win (-1)

        st.text(f"{HomeTeam} Win Probability: {round(home_win_proba * 100)}%")
        st.text(f"Draw Probability: {round(draw_proba * 100)}%")
        st.text(f"{AwayTeam} Win Probability: {round(away_win_proba * 100)}%")