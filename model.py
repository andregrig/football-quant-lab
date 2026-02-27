import pandas as pd
import numpy as np
import math

CSV_PATH = "I1.csv"

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

df['HomeTeam_lower'] = df['HomeTeam'].str.lower()
df['AwayTeam_lower'] = df['AwayTeam'].str.lower()

TEAM_ALIASES = {
    "juve": "juventus",
    "ac milan": "milan",
    "internazionale": "inter",
    "roma fc": "roma"
}

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

def calculate_lambdas(data, home_team, away_team):

    home_team = home_team.strip().lower()
    away_team = away_team.strip().lower()

    home_team = TEAM_ALIASES.get(home_team, home_team)
    away_team = TEAM_ALIASES.get(away_team, away_team)

    avg_home_goals = data['FTHG'].mean()
    avg_away_goals = data['FTAG'].mean()

    home_stats = data[data['HomeTeam_lower'] == home_team]
    away_stats = data[data['AwayTeam_lower'] == away_team]

    if len(home_stats) < 2 or len(away_stats) < 2:
        return None, None

    home_attack = home_stats['FTHG'].mean()
    home_defense = home_stats['FTAG'].mean()

    away_attack = away_stats['FTAG'].mean()
    away_defense = away_stats['FTHG'].mean()

    lambda_home = (home_attack * away_defense) / avg_home_goals
    lambda_away = (away_attack * home_defense) / avg_away_goals

    return lambda_home, lambda_away

def predict_match(home_team, away_team, max_goals=5):

    lambda_home, lambda_away = calculate_lambdas(df, home_team, away_team)

    if lambda_home is None:
        return None

    home_win = 0
    draw = 0
    away_win = 0
    over_2_5 = 0
    score_probs = []

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):

            prob = poisson_prob(lambda_home, i) * poisson_prob(lambda_away, j)

            score_probs.append({
                "score": f"{i}-{j}",
                "prob": prob
            })

            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob

            if i + j > 2:
                over_2_5 += prob

    under_2_5 = 1 - over_2_5

    score_probs = sorted(score_probs, key=lambda x: x["prob"], reverse=True)

    top_scores = [
        {
            "score": s["score"],
            "probability": round(s["prob"] * 100, 2)
        }
        for s in score_probs[:5]
    ]

    return {
        "home_win": round(home_win * 100, 2),
        "draw": round(draw * 100, 2),
        "away_win": round(away_win * 100, 2),
        "over_2_5": round(over_2_5 * 100, 2),
        "under_2_5": round(under_2_5 * 100, 2),
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "top_scores": top_scores
    }