"""
app.py
Streamlit frontend for the World Cup match outcome predictor.
Run: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ World Cup Match Predictor",
    page_icon="⚽",
    layout="centered",
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
MODEL_DIR = "model"

@st.cache_resource
def load_artifacts():
    model        = joblib.load(f"{MODEL_DIR}/model.pkl")
    team_stats   = joblib.load(f"{MODEL_DIR}/team_stats.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    teams        = joblib.load(f"{MODEL_DIR}/teams.pkl")
    return model, team_stats, feature_names, teams


LOOKBACK_MATCHES = 10


def rolling_features(team_history: list, n: int = LOOKBACK_MATCHES):
    """Return rolling averages over the last n matches (no date filter — use all history)."""
    past = team_history[-n:]
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


def predict_outcome(model, team_stats, feature_names,
                    home_team: str, away_team: str, neutral: bool):
    hf = rolling_features(team_stats.get(home_team, []))
    af = rolling_features(team_stats.get(away_team, []))

    features = {
        "h_avg_gf":     hf["avg_gf"],
        "h_avg_ga":     hf["avg_ga"],
        "h_win_rate":   hf["win_rate"],
        "h_draw_rate":  hf["draw_rate"],
        "h_loss_rate":  hf["loss_rate"],
        "a_avg_gf":     af["avg_gf"],
        "a_avg_ga":     af["avg_ga"],
        "a_win_rate":   af["win_rate"],
        "a_draw_rate":  af["draw_rate"],
        "a_loss_rate":  af["loss_rate"],
        "gf_diff":      hf["avg_gf"] - af["avg_gf"],
        "ga_diff":      hf["avg_ga"] - af["avg_ga"],
        "wr_diff":      hf["win_rate"] - af["win_rate"],
        "is_neutral":   int(neutral),
    }

    X = np.array([[features[f] for f in feature_names]])
    proba = model.predict_proba(X)[0]   # [away_win, draw, home_win]
    pred  = model.predict(X)[0]
    label = {0: "Away Win", 1: "Draw", 2: "Home Win"}[pred]
    return label, proba, hf, af


def team_form_bar(label: str, stats: dict):
    """Render a compact form summary."""
    mp = stats["matches_played"]
    if mp == 0:
        st.write(f"**{label}** – no data available")
        return
    cols = st.columns(5)
    cols[0].metric("Avg Goals For",     f"{stats['avg_gf']:.2f}")
    cols[1].metric("Avg Goals Against", f"{stats['avg_ga']:.2f}")
    cols[2].metric("Win Rate",          f"{stats['win_rate']*100:.0f}%")
    cols[3].metric("Draw Rate",         f"{stats['draw_rate']*100:.0f}%")
    cols[4].metric("Loss Rate",         f"{stats['loss_rate']*100:.0f}%")


# ─── App layout ───────────────────────────────────────────────────────────────
st.title("⚽ World Cup Match Outcome Predictor")
st.markdown(
    "Select two national teams to get a data-driven prediction based on "
    "historical FIFA match results (**49 000+ games** dating back to 1872)."
)

if not os.path.exists(f"{MODEL_DIR}/model.pkl"):
    st.error(
        "Model not found. Please run `python train_model.py` first to train "
        "and save the model artifacts."
    )
    st.stop()

model, team_stats, feature_names, teams = load_artifacts()

st.divider()

# ─── Team selection ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0, key="home")

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select away team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1, key="away")

neutral = st.checkbox("Neutral venue (e.g. World Cup group stage)", value=True)

st.divider()

# ─── Predict ──────────────────────────────────────────────────────────────────
if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
    label, proba, hf, af = predict_outcome(
        model, team_stats, feature_names, home_team, away_team, neutral
    )

    away_p, draw_p, home_p = proba

    # ── Result banner ──────────────────────────────────────────────────────
    result_emoji = {"Home Win": "🟢", "Draw": "🟡", "Away Win": "🔴"}
    st.markdown(f"## {result_emoji[label]} Predicted Result: **{label}**")
    if label == "Home Win":
        st.success(f"**{home_team}** is predicted to win this match.")
    elif label == "Away Win":
        st.error(f"**{away_team}** is predicted to win this match.")
    else:
        st.info("The model predicts this match will end in a **Draw**.")

    # ── Probability bars ───────────────────────────────────────────────────
    st.markdown("### Probabilities")
    prob_df = pd.DataFrame({
        "Outcome":     [f"🏠 {home_team} Win", "🤝 Draw", f"✈️ {away_team} Win"],
        "Probability": [home_p, draw_p, away_p],
    })

    col_h, col_d, col_a = st.columns(3)
    col_h.metric(f"{home_team} Win", f"{home_p*100:.1f}%")
    col_d.metric("Draw",            f"{draw_p*100:.1f}%")
    col_a.metric(f"{away_team} Win", f"{away_p*100:.1f}%")

    st.bar_chart(
        prob_df.set_index("Outcome"),
        color=["#2ecc71"],
        height=220,
    )

    # ── Team form ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📊 {home_team} — Last {LOOKBACK_MATCHES} Matches")
    team_form_bar(home_team, hf)

    st.markdown(f"### 📊 {away_team} — Last {LOOKBACK_MATCHES} Matches")
    team_form_bar(away_team, af)

    st.divider()
    st.caption(
        "Model: Random Forest (300 trees) trained on 49 000+ historical "
        "international matches. Predictions are probabilistic and for "
        "entertainment/educational purposes only."
    )
