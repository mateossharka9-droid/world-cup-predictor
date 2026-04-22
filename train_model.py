"""
train_model.py
Trains a World Cup match outcome predictor and saves the model artifacts.
Run this once before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = "data/results.csv"
MODEL_DIR = "model"
LOOKBACK_MATCHES = 10          # rolling window for form features
MIN_MATCHES_PLAYED = 5         # teams need at least this many matches


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add target column: 0=away_win, 1=draw, 2=home_win"""
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    choices = [2, 1, 0]  # home_win, draw, away_win
    df["outcome"] = np.select(conditions, choices)
    return df


def build_team_stats(df: pd.DataFrame) -> dict:
    """
    For every team, maintain a rolling record of:
    goals_for, goals_against, wins, draws, losses per match.
    Returns a dict: team -> list of dicts ordered by date.
    """
    stats: dict[str, list] = {}

    for _, row in df.iterrows():
        for side, opp_side in [("home", "away"), ("away", "home")]:
            team = row[f"{side}_team"]
            gf = row[f"{side}_score"]
            ga = row[f"{opp_side}_score"]
            win = int(gf > ga)
            draw = int(gf == ga)
            loss = int(gf < ga)

            if team not in stats:
                stats[team] = []
            stats[team].append(
                {"date": row["date"], "gf": gf, "ga": ga,
                 "win": win, "draw": draw, "loss": loss}
            )
    return stats


def rolling_features(team_history: list, before_date, n: int = LOOKBACK_MATCHES):
    """Return rolling averages over the last n matches before a given date."""
    past = [m for m in team_history if m["date"] < before_date][-n:]
    if not past:
        return {"avg_gf": 0, "avg_ga": 0, "win_rate": 0,
                "draw_rate": 0, "loss_rate": 0, "matches_played": 0}
    gf = np.mean([m["gf"] for m in past])
    ga = np.mean([m["ga"] for m in past])
    wr = np.mean([m["win"] for m in past])
    dr = np.mean([m["draw"] for m in past])
    lr = np.mean([m["loss"] for m in past])
    return {"avg_gf": gf, "avg_ga": ga, "win_rate": wr,
            "draw_rate": dr, "loss_rate": lr, "matches_played": len(past)}


def build_feature_matrix(df: pd.DataFrame, team_stats: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        date = row["date"]

        hf = rolling_features(team_stats.get(ht, []), date)
        af = rolling_features(team_stats.get(at, []), date)

        # skip rows where either team lacks history
        if (hf["matches_played"] < MIN_MATCHES_PLAYED or
                af["matches_played"] < MIN_MATCHES_PLAYED):
            continue

        feature_row = {
            # home team features
            "h_avg_gf":     hf["avg_gf"],
            "h_avg_ga":     hf["avg_ga"],
            "h_win_rate":   hf["win_rate"],
            "h_draw_rate":  hf["draw_rate"],
            "h_loss_rate":  hf["loss_rate"],
            # away team features
            "a_avg_gf":     af["avg_gf"],
            "a_avg_ga":     af["avg_ga"],
            "a_win_rate":   af["win_rate"],
            "a_draw_rate":  af["draw_rate"],
            "a_loss_rate":  af["loss_rate"],
            # head-to-head differential
            "gf_diff":      hf["avg_gf"] - af["avg_gf"],
            "ga_diff":      hf["avg_ga"] - af["avg_ga"],
            "wr_diff":      hf["win_rate"] - af["win_rate"],
            # context
            "is_neutral":   int(row["neutral"]),
            # target
            "outcome":      row["outcome"],
        }
        rows.append(feature_row)

    return pd.DataFrame(rows)


def train(df_features: pd.DataFrame):
    X = df_features.drop(columns=["outcome"])
    y = df_features["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Away Win", "Draw", "Home Win"]))
    return clf, X.columns.tolist()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data …")
    df = load_and_clean(DATA_PATH)
    df = add_outcome_label(df)

    print("Building team history …")
    team_stats = build_team_stats(df)

    print("Engineering features …")
    df_feat = build_feature_matrix(df, team_stats)
    print(f"Feature matrix shape: {df_feat.shape}")

    print("Training Random Forest …")
    model, feature_names = train(df_feat)

    # save artifacts
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(team_stats, f"{MODEL_DIR}/team_stats.pkl")
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

    # save sorted team list for the UI dropdowns
    teams = sorted(team_stats.keys())
    joblib.dump(teams, f"{MODEL_DIR}/teams.pkl")

    print(f"\nArtifacts saved to ./{MODEL_DIR}/")
    print(f"Total teams in database: {len(teams)}")


if __name__ == "__main__":
    main()
