# 🏏 IPL Analytics & Match Predictor

> **Predict. Analyze. Win.**
> 
> 🚀 **[Live Demo → shibang-ipl-analytics.streamlit.app](https://shibang-ipl-analytics.streamlit.app)**

An end-to-end **Data Analytics + Machine Learning** project analyzing **1,090 IPL matches** and **260,920 ball-by-ball deliveries** from 2008–2024. Built a Random Forest classifier to predict match outcomes with a full interactive analytics dashboard. The application is deployed publicly on Streamlit Cloud for interactive use.

---

## 📸 Screenshots

### Predict Match
![Predict](images/home.png)

### Player Stats
![Player Stats](images/player.png)

### Analytics Charts
![Analytics](images/analytics.png)

### Head to Head
![H2H](images/h2h.png)

---

## ✨ Features

| Page | Description |
|------|-------------|
| ⚡ Predict Match | Predict winner with win probabilities using ML |
| 👥 Teams | All-time win rankings, sixes, win rate per team |
| 🏏 Player Stats | Top batters, bowlers, Orange & Purple cap history |
| 📊 Analytics Charts | Season run trends, toss decisions, boundary leaders |
| ⚔️ Head to Head | Full rivalry breakdown between any two teams |
| ℹ️ About | Model methodology, key findings, tech stack |

---

## 🧠 ML Model

- **Algorithm:** Random Forest Classifier (1000 estimators)
- **Train/Test split:** 75% / 25% (time-based, no data leakage)
- **Accuracy:** ~52% — consistent with domain benchmarks for pre-match cricket prediction
- **Deployed:** Publicly on Streamlit Cloud ✔

**Features used:**
- Team win rate differential (strongest predictor)
- Rolling form — last 15 matches & last 5 matches
- Head-to-head historical win rate
- Venue-specific win rate differential
- Toss outcome & decision
- Season number & match stage

**Feature Engineering:**
```
• Rolling team form       →  Last 5 and last 15 match win rates
• Venue performance       →  Team-specific win rate at each ground
• Historical H2H          →  Head-to-head win rate (rolling, no leakage)
• Toss influence          →  Toss winner + bat/field decision encoded
• Team strength metrics   →  Overall win rate differential between teams
```

> 💡 Even professional betting models achieve only 55–58% on cricket. The model's key value is in **feature importance insights**, not just accuracy.

---

## 📊 Key Findings

```
✅ Win rate differential  →  Strongest predictor of match outcome
✅ Toss outcome           →  Near-zero impact (~50.8% win rate)
✅ Field-first preference →  Grew from 55% (2008) to 83% (2018)
✅ Average match scores   →  Rose 27% from 2009 to 2024
✅ V. Kohli               →  All-time runs leader (8,014 runs)
✅ YS Chahal              →  All-time wicket leader (205 wickets)
✔  Publicly deployed      →  Live on Streamlit Cloud
```

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Streamlit` `Matplotlib` `Seaborn` `SHAP`

---

## 📁 Project Structure

```
ipl-analytics/
│
├── app.py              # Streamlit web app (all pages)
├── ipl.ipynb           # EDA + ML notebook (Jupyter)
├── matches.csv         # Match-level data (1,090 matches)
├── requirements.txt    # Python dependencies
├── images/             # Screenshots for README
└── README.md
```

> ⚠️ `deliveries.csv` (260,920 rows) is hosted on Hugging Face due to GitHub's 25MB file size limit.
> Download → [HuggingFace Dataset](https://huggingface.co/datasets/shibangmaity/ipl-analytics-data)

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

> For the full EDA notebook, download `deliveries.csv` from Hugging Face and place it in the root folder.

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🌐 Live App | [shibang-ipl-analytics.streamlit.app](https://shibang-ipl-analytics.streamlit.app) |
| 📦 Dataset | [Hugging Face](https://huggingface.co/datasets/shibangmaity/ipl-analytics-data) |
| 📓 Kaggle Source | [IPL Complete Dataset 2008–2024](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) |

---

## 👤 Author

**Shibang Maity** · Computer Science, KIIT University

- 📧 shibangmaity@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/shibang-maity-865ba4304)
- 🐙 [GitHub](https://github.com/shibangmaity)

---

*Built with ❤️ and cricket data · IPL 2008–2024 · Random Forest Classifier*
