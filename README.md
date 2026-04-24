# ⚽ World Cup Match Outcome Predictor

A machine-learning project that predicts the outcome of international football matches — **Home Win**, **Draw**, or **Away Win** — using 150+ years of FIFA historical data.

Built with **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**.

---

## 🗂️ Project Structure

```
world_cup_predictor/
│
├── data/
│   └── results.csv          # Historical match results (Kaggle dataset)
│
├── model/                   # Auto-generated after training
│   ├── model.pkl
│   ├── team_stats.pkl
│   ├── feature_names.pkl
│   └── teams.pkl
│
├── train_model.py           # Feature engineering + model training
├── app.py                   # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

[International Football Results (1872–2026)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — Kaggle

Place the downloaded `results.csv` inside a `data/` folder.

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/mateossharka9-droid/world-cup-predictor.git
cd world-cup-predictor
pip install -r requirements.txt
```

### 2. Add the dataset

```bash
mkdir data
# Put results.csv in the data/ folder
```

### 3. Train the model

```bash
python train_model.py
```

This will print test accuracy and a classification report, then save four
artifacts to `model/`.

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

### Feature Engineering

For each match, the model looks back at the **last 10 matches** each team played
before that game and computes rolling statistics:

| Feature | Description |
|---|---|
| `h/a_avg_gf` | Average goals scored |
| `h/a_avg_ga` | Average goals conceded |
| `h/a_win_rate` | Win percentage |
| `h/a_draw_rate` | Draw percentage |
| `h/a_loss_rate` | Loss percentage |
| `gf_diff` | Home minus away avg goals scored |
| `ga_diff` | Home minus away avg goals conceded |
| `wr_diff` | Home minus away win rate |
| `is_neutral` | 1 if played at a neutral venue |

### Model

A **Random Forest Classifier** (300 trees, `max_depth=12`) trained on ~38 000
samples with a stratified 80/20 train/test split.

```
Test Accuracy: ~48%
```

> Predicting football is inherently noisy — a naive baseline (always predict
> the most common class, Home Win) gives ~49%, so the model roughly matches
> that while producing calibrated probabilities across all 3 outcomes.
> It significantly outperforms random chance (~33%) on home wins and away wins.

### Target Classes

| Label | Meaning |
|---|---|
| 0 | Away Win |
| 1 | Draw |
| 2 | Home Win |

---

## 📈 Results

```
              precision    recall  f1-score
   Away Win       0.46      0.51      0.48
       Draw       0.27      0.32      0.29
   Home Win       0.64      0.54      0.59
```

Home wins are the easiest to predict; draws remain the hardest — consistent
with the broader sports analytics literature.

---

## 🛠️ Possible Improvements

- Add FIFA/ELO rankings as features
- Include head-to-head historical record
- Weight recent matches more heavily (exponential decay)
- Fine-tune on World Cup matches only
- Try XGBoost or LightGBM

---

## 📄 License

MIT
