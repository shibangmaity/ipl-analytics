# 🏏 IPL Analytics & Match Predictor

> **Predict. Analyze. Win.**
>
> 🚀 **[Live Demo → shibang-ipl-analytics.streamlit.app](https://shibang-ipl-analytics.streamlit.app)**

An end-to-end **Data Analytics + Machine Learning** project analyzing **18 seasons of IPL** (2008–2026), covering **1,100+ matches** and **288,000+ ball-by-ball deliveries**. Built a Random Forest + Gradient Boosting Ensemble to predict match outcomes with a full interactive analytics dashboard — publicly deployed on Streamlit Cloud.

---

## 📸 Screenshots

### ⚡ Match Predictor
![Predict](home.png)

### 👥 Teams
![Teams](Teams.png)

### 🏏 Player Stats
![Player Stats](Players.png)

### 🎯 Style Analytics
![Style](Style.png)

### 🔬 Deep Analytics
![Deep](Deep.png)

### 📊 Analytics Charts
![Analytics](Charts.png)

### ⚔️ Head to Head
![H2H](H2H.png)

### ℹ️ About
![About](About.png)

---

## ✨ Features

| Page | Description |
|------|-------------|
| ⚡ Predict Match | Predict winner with win probabilities using ML |
| 👥 Teams | All-time win rankings, sixes, win rate per team |
| 🏏 Player Stats | Top batters, bowlers, Orange & Purple cap history, milestones |
| 🎯 Style Analytics | Left vs Right bat · Pace vs Spin · Full matchup matrix |
| 🔬 Deep Analytics | Phase splits · Dismissals · Bowl styles · Allrounders · Death bowlers · PP batters |
| 📊 Charts | Season run trends, toss decisions, team wins, top batters/bowlers |
| ⚔️ Head to Head | Full rivalry breakdown between any two teams |
| ℹ️ About | Model methodology, dataset breakdown, key findings, tech stack |

---

## 🧠 ML Model

- **Algorithm:** Random Forest + Gradient Boosting → **Ensemble (RF + GB)**
- **Augmentation:** Flip-team data augmentation to double training size and remove team-order bias
- **Train/Test split:** 80% / 20% (time-based — no data leakage)
- **Features:** 25 engineered features (all pre-match, no leakage)
- **Accuracy:** ~54–55% — strong benchmark for pre-match T20 prediction
- **Explainability:** SHAP summary plots for feature importance
- **Deployed:** Publicly on Streamlit Cloud ✔

**Features used:**
- Team win rate differential (strongest predictor — confirmed by SHAP)
- Rolling form — last 5, 10 & 15 matches
- Head-to-head historical win rate (rolling, no leakage)
- Within-season win rate
- Venue-specific win rate differential
- Toss outcome & decision
- Player style features — left-hand batter ratio & pace bowling ratio
- Season number

**Feature Engineering:**
```
• Rolling team form       →  Last 5, 10, 15 match win rates (3 windows)
• Venue performance       →  Team-specific win rate at each ground
• Historical H2H          →  Head-to-head win rate (pre-match only)
• Season win rate         →  Current season form (no leakage)
• Toss influence          →  Toss winner + bat/field decision encoded
• Team strength metrics   →  Overall win rate differential between teams
• Player style            →  Left-hand batter ratio & pace bowling ratio
```

**Model Comparison:**
```
Logistic Regression   →  Excluded (linear model, below 50% accuracy)
Random Forest         →  ~53–54%
Gradient Boosting     →  ~54–55%
Ensemble (RF + GB)    →  ~54–55% ← Best
```

> 💡 Even professional betting models achieve only 55–58% on T20 cricket. The model's key value is in **feature importance insights** (SHAP) and **trend discovery**, not just raw accuracy.

---

## 📊 Key Findings

```
✅ Win rate differential      →  Strongest predictor of match outcome (SHAP #1)
✅ Toss outcome               →  Near-zero impact (~50.8% win rate)
✅ Field-first preference     →  Grew 70%+ post-2013; now dominant
✅ Average match scores       →  Rose 27%+ from 2009 to 2026
✅ Left-handers vs Pace       →  Highest strike rate (131.7) — angle advantage
✅ Left arm Pace vs RHB       →  Highest wicket rate (5.41%) — hardest matchup
✅ Death over run rate        →  ~9.0 vs powerplay ~7.3 — most explosive phase
✅ Pace takes 2.2× wickets   →  But spin has lower economy in middle overs
✅ Ensemble accuracy          →  ~54–55% — strong T20 ML benchmark
✔  Publicly deployed          →  Live on Streamlit Cloud
```

---

## 🎯 Style & Matchup Analytics

One of the app's unique features is a deep **batter vs bowler matchup matrix**:

| Matchup | Strike Rate | Wicket % |
|---------|-------------|----------|
| Left bat vs Left arm Spin | 137.6 | 4.8% |
| Left bat vs Pace | 131.7 | 5.1% |
| Right bat vs Left arm Pace | 128.4 | 5.41% ← highest |
| Right bat vs Left arm Spin | 119.0 | 4.6% ← most restrictive |

---

## 🔬 Deep Analytics Pages

- **Phase Analysis** — Powerplay / Middle / Death over run rates & wicket rates
- **Dismissal Breakdown** — Caught, bowled, LBW, run out distribution
- **Bowl Style Wicket Rate** — Wrist spin vs orthodox vs pace styles
- **Allrounders** — Players with 500+ runs AND 30+ wickets
- **City Win Rates** — Team performance by city across all venues
- **Death Bowlers** — Best overs 15–19 specialists (wickets + economy)
- **Powerplay Batters** — Best overs 0–5 scorers with strike rates
- **Caught Leaders** — Bowlers who generate the most caught dismissals

---

## 🗃️ Data Pipeline

Raw data was sourced from **Cricsheet** (JSON) and **Kaggle**, then parsed and cleaned into 3 CSVs:

```
Cricsheet JSON (ipl_male_json.zip)     Kaggle
        ↓                                 ↓
  Parse all match JSONs            players_clean.csv
        ↓
  matches_cricsheet.csv
  deliveries_cricsheet.csv
        ↓
  Fix season labels (2020/2021 overlap)
  Standardize team names
        ↓
  Upload all 3 CSVs to Hugging Face
        ↓
  app.py loads from Hugging Face at runtime
```

**Datasets:**

| File | Description | Rows |
|------|-------------|------|
| `matches_cricsheet.csv` | Match-level data (2008–2026) | 1,100+ |
| `deliveries_cricsheet.csv` | Ball-by-ball data | 288,000+ |
| `players_clean.csv` | Player profiles (bat hand, bowl style, bowl arm) | 799 |

> ⚠️ All 3 datasets hosted on **Hugging Face** (sourced from Kaggle & Cricsheet) due to GitHub's file size limit. Loaded automatically on first run — no manual download needed.

---

## 🛠️ Tech Stack

`Python 3.11` `Pandas` `NumPy` `Scikit-learn` `Streamlit` `Matplotlib` `Seaborn` `SHAP` `SQLite` `HuggingFace Datasets`

---

## 📁 Project Structure

```
ipl-analytics/
│
├── app.py                          # Streamlit web app (all 8 pages)
├── IPL_Analytics_2008_2026.ipynb   # Full EDA + ML notebook (17 sections)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/shibangmaity/ipl-analytics.git
cd ipl-analytics

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Datasets are loaded automatically from Hugging Face on first run — no manual download needed.

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🌐 Live App | [shibang-ipl-analytics.streamlit.app](https://shibang-ipl-analytics.streamlit.app) |
| 📦 Hugging Face *(all datasets)* | [shibangmaity/ipl-analytics-data](https://huggingface.co/datasets/shibangmaity/ipl-analytics-data) |
| 📊 Kaggle | [IPL Complete Dataset 2008–2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) |
| 🏏 Cricsheet | [cricsheet.org](https://cricsheet.org) |

---

## 👤 Author

**Shibang Maity** · Computer Science, KIIT University

- 📧 shibangmaity@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/shibang-maity-865ba4304)
- 🐙 [GitHub](https://github.com/shibangmaity)

---

*Built with ❤️ and cricket data · IPL 2008–2026 · Data: Kaggle + Cricsheet · Random Forest + Gradient Boosting + SHAP · KIIT University 2026*
