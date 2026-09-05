# IPL Win Prediction

A machine-learning project that predicts whether the team batting second will win an Indian Premier League (IPL) match. The model learns from ball-by-ball match data and uses the live match situation to estimate the result.

## Features used for prediction

- Batting and bowling teams
- First-innings target score
- Current score
- Runs remaining
- Balls remaining
- Wickets remaining
- Current run rate
- Required run rate

## Project files

```text
ipl prediction/
├── main.py.py          # Main training and prediction script
├── matches.csv         # IPL match-level data
├── deliveries.csv      # Ball-by-ball IPL data
└── README.md
```

## Dataset

The project requires both CSV files in the same folder as `main.py.py`:

- `matches.csv` contains match information, including teams, winners, venues, and seasons.
- `deliveries.csv` contains ball-by-ball match information, including scores, batters, bowlers, and dismissals.

> **Note:** `deliveries.csv` is larger than GitHub's 25 MB website-upload limit. If you upload the project through the GitHub website, leave this file out and provide its download source separately, or push it using GitHub Desktop/Git instead.

## Installation

Install Python 3.10 or later, then install the packages below:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Run the project

Open a terminal in the project folder and run:

```bash
python "main.py.py"
```

The program will:

1. Load the IPL match and delivery datasets.
2. Prepare match-state features for the second innings.
3. Encode the batting and bowling teams.
4. Train an XGBoost classification model.
5. Print the test accuracy and the most important prediction features.

## Model

The project uses `XGBClassifier` with 200 estimators and a maximum tree depth of 10. Data is split by match ID so deliveries from the same match are not included in both training and testing data.

## Important notes

- Keep `main.py.py`, `matches.csv`, and `deliveries.csv` in the same folder.
- The current script standardizes older team names, such as `Delhi Daredevils` to `Delhi Capitals`.
- For a clearer filename, you may rename `main.py.py` to `main.py`; if you do, run it with `python main.py`.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn

