import pandas as pd
import numpy as np
import math

# =========================
# CARICAMENTO DATASET
# =========================

CSV_PATH = "I1.csv"  # Deve essere nella stessa cartella

df = pd.read_csv(CSV_PATH)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

print("Dataset caricato correttamente.")


# =========================
# FUNZIONE POISSON BASE
# =========================

def poisson_prob(lmbda, k):
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)


# =========================
# MODELLO POISSON 1X2
# =========================

def poisson_model(df, home_team, away_team, max_goals=5):

    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()

    home_stats = df[df['HomeTeam'] == home_team]
    away_stats = df[df['AwayTeam'] == away_team]

    if len(home_stats) < 5 or len(away_stats) < 5:
        return None

    home_attack = home_stats['FTHG'].mean()
    home_defense = home_stats['FTAG'].mean()

    away_attack = away_stats['FTAG'].mean()
    away_defense = away_stats['FTHG'].mean()

    lambda_home = (home_attack * away_defense) / avg_home_goals
    lambda_away = (away_attack * home_defense) / avg_away_goals

    home_win = 0
    draw = 0
    away_win = 0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):

            prob = poisson_prob(lambda_home, i) * poisson_prob(lambda_away, j)

            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob

    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win
    }


# =========================
# PROBABILITÀ IMPLICITE
# =========================

def implied_probabilities_from_odds(home_odds, draw_odds, away_odds):

    inv = np.array([1/home_odds, 1/draw_odds, 1/away_odds])
    inv_sum = inv.sum()
    norm = inv / inv_sum

    return {
        'home_win': norm[0],
        'draw': norm[1],
        'away_win': norm[2]
    }


# =========================
# CONTROLLO VALUE
# =========================

def check_value(model_probs, implied_probs, threshold=0.05):

    results = {}

    for k in ['home_win','draw','away_win']:
        diff = model_probs[k] - implied_probs[k]
        results[k] = diff >= threshold

    return results


# =========================
# BACKTEST COMPLETO
# =========================

def backtest_poisson(df, threshold=0.05):

    profit = 0
    bets = 0
    wins = 0

    for i in range(50, len(df)):

        historical = df.iloc[:i]
        match = df.iloc[i]

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        if pd.isna(match["B365H"]) or pd.isna(match["B365D"]) or pd.isna(match["B365A"]):
            continue

        model_probs = poisson_model(historical, home_team, away_team)

        if model_probs is None:
            continue

        implied_probs = implied_probabilities_from_odds(
            match["B365H"],
            match["B365D"],
            match["B365A"]
        )

        values = check_value(model_probs, implied_probs, threshold)

        for outcome in ["home_win", "draw", "away_win"]:

            if values[outcome]:

                bets += 1

                if outcome == "home_win" and match["FTHG"] > match["FTAG"]:
                    profit += match["B365H"] - 1
                    wins += 1

                elif outcome == "draw" and match["FTHG"] == match["FTAG"]:
                    profit += match["B365D"] - 1
                    wins += 1

                elif outcome == "away_win" and match["FTHG"] < match["FTAG"]:
                    profit += match["B365A"] - 1
                    wins += 1

                else:
                    profit -= 1

    roi = (profit / bets) * 100 if bets > 0 else 0

    return profit, bets, wins, roi


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("\n=== BACKTEST POISSON MODEL ===")

    profit, bets, wins, roi = backtest_poisson(df, threshold=0.05)

    print(f"\nNumero scommesse: {bets}")
    print(f"Numero vinte: {wins}")
    print(f"Profitto totale: {profit:.2f} unità")
    print(f"ROI: {roi:.2f}%")