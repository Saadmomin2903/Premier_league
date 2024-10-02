# Premier League Match Predictor

## Overview

This repository contains a Premier League match prediction model that utilizes historical match data to predict the outcomes of future matches. The model is built using the CatBoost Classifier and includes data preprocessing steps like handling missing values, calculating rolling averages, and form statistics for both home and away teams.

### Features:

- **Data Preprocessing**: Handling missing values, removing duplicates, filling categorical and numerical values.
- **Rolling Averages**: Calculate rolling averages for goals and other stats for the last five matches for both home and away teams.
- **Match Prediction**: Predict match results using CatBoost Classifier.
- **Streamlit Dashboard**: User-friendly dashboard to predict match results based on teams.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/premier-league-predictor.git
cd premier-league-predictor
```

### 2. Install Dependencies

Install the necessary dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Set Up the Database

Set up a PostgreSQL database and import the preprocessed Premier League stats:

```python
from sqlalchemy import create_engine
import pandas as pd

# Database connection string
db_url = "postgresql://<your_db_username>:<your_db_password>@<your_db_host>/<your_db_name>?sslmode=require"

# Load the cleaned data and store it into the database
df_cleaned = pd.read_csv('final11.csv')
engine = create_engine(db_url)
df_cleaned.to_sql('preprocessed_stats', engine, if_exists='append', index=False)
```

### 4. Run the Streamlit Dashboard

You can launch the Streamlit app to interact with the prediction model:

```bash
streamlit run main1.py
```

## Data Preprocessing

The dataset is loaded, cleaned, and preprocessed using the following steps:

1. **Load Data**: Read the merged Premier League dataset (`merged_premier_league_data.csv`).
2. **Filter Premier League Matches**: Filter the data to include only Premier League matches.
3. **Remove Duplicates**: Clean the data by removing duplicates based on `Date` and `Referee`.
4. **Handle Missing Values**:
   - Numeric columns are filled with the median value.
   - Categorical columns are filled with the most frequent value.
5. **Encode Results**: Map match results to integers: `W` (Win) = 1, `D` (Draw) = 0, `L` (Loss) = -1.
6. **Replace Low-Match Teams**: Replace teams with fewer than 40 matches with 'Other'.
7. **Rolling Averages and Forms**:
   - Calculate rolling averages for `GF` (Goals For) and `GA` (Goals Against) for the last 5 matches for both home and away teams.
   - Calculate rolling forms for home and away teams (wins, draws, losses).

## Model Training

The match outcome prediction model uses a **CatBoost Classifier** with hyperparameters tuned using grid search. The pipeline includes one-hot encoding for the team names.

### Features:

- `home_team`: The name of the home team.
- `Opponent`: The name of the away team.
- `Rolling_GF_Home`, `Rolling_GA_Home`: Rolling averages of goals scored and conceded by the home team.
- `Rolling_GF_Away`, `Rolling_GA_Away`: Rolling averages of goals scored and conceded by the away team.
- `Home_Wins_Form`, `Home_Draws_Form`, `Home_Losses_Form`: Form statistics for the home team (wins, draws, losses).
- `Away_Wins_Form`, `Away_Draws_Form`, `Away_Losses_Form`: Form statistics for the away team (wins, draws, losses).

The model achieves an accuracy of **72.3%** on the test set.

## Usage

### 1. Train the Model

Run the following code to train the model on the preprocessed data:

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier

X = df[feature_columns]
y = df[target_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding for categorical columns
trf = ColumnTransformer([("trf", OneHotEncoder(sparse_output=False, drop="first"), ['home_team', 'Opponent'])], remainder="passthrough")

# Define the CatBoost model
model = CatBoostClassifier(
    colsample_bylevel=0.9,
    depth=8,
    l2_leaf_reg=1,
    learning_rate=0.01,
    n_estimators=300
)

# Pipeline
pipe = Pipeline(steps=[("step1", trf), ("step2", model)])
pipe.fit(X_train, Y_train)

# Evaluate the model
Y_pred = pipe.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")
```

### 2. Predict Match Results

Using the trained model, you can predict the results for specific matches using the Streamlit dashboard or by calling the prediction functions in Python.

## License

This project is licensed under the MIT License. Feel free to modify and distribute it as needed.

## Contributing

Contributions are welcome! Please create a pull request if you want to improve or add new features to this repository.

