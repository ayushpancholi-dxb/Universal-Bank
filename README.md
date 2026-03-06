# Universal Bank Personal Loan Analytics Dashboard

> **Built by Ayush Pancholi | MBA Candidate in Finance – SP Jain Global School of Management**

A comprehensive **4-layer analytics dashboard** (Descriptive → Diagnostic → Predictive → Prescriptive) that helps Universal Bank identify which customers are most likely to accept a personal loan offer.

---

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📊 Dashboard Features

### Tab 1 – Descriptive Analytics
- Loan acceptance donut chart (9.6% acceptance rate)
- Age and income distribution histograms by loan status
- Family size and education level breakdowns
- KPI summary cards (total customers, avg income, avg CCAvg, avg mortgage)
- Average statistics table: Accepted vs Rejected

### Tab 2 – Diagnostic Analytics
- Violin plots: Income & CCAvg for accepted vs rejected
- Acceptance rate by education level and income group
- Income vs CCAvg scatter plot (bubble = mortgage)
- Banking service ownership analysis (CD Account, Securities, Online, CreditCard)
- Full correlation heatmap
- Radar chart: normalised feature profiles
- **Interactive Sunburst drill-down**: Income Group → Education → Loan Status
- **Interactive Treemap**: Income Group → Family Size → Loan Status

### Tab 3 – Predictive Analytics
- Choose from 4 classifiers: Random Forest, Gradient Boosting, Logistic Regression, Decision Tree
- Metrics: Accuracy, Precision, Recall, F1, ROC AUC
- Feature importance / coefficient chart
- ROC Curve with AUC score
- Confusion matrix
- Probability distribution histogram
- Customer-level prediction explorer with adjustable threshold

### Tab 4 – Prescriptive Analytics
- Lead funnel by probability tier
- 6 actionable customer segments with targeting strategy
- **Personalised Loan Offer Engine**: generates loan amount, interest rate, tenure, and pitch line per customer
- Downloadable CSV of all personalised offers
- Campaign roadmap (4-phase strategy)

---

## 🗂️ Project Structure

```
universal_bank_dashboard/
│
├── app.py               # Main Streamlit dashboard
├── data_utils.py        # Data loading, cleaning, feature engineering
├── model_utils.py       # ML models, prediction, offer generation
├── charts.py            # All Plotly chart factory functions
├── requirements.txt     # Python dependencies
│
└── data/
    └── UniversalBank.xlsx   # ← Place your dataset here
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your dataset
# Place UniversalBank.xlsx inside the data/ folder

# 5. Run the dashboard
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. **Push this repository to GitHub** (include the `data/` folder with `UniversalBank.xlsx`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

> **Note:** If your dataset is large or sensitive, you can also add a file uploader (already built-in to the sidebar) and **exclude** the `data/` folder from GitHub using `.gitignore`.

---

## 📦 Dataset

| Column | Type | Description |
|---|---|---|
| ID | int | Customer ID |
| Age | int | Age in years |
| Experience | int | Years of professional experience |
| Income | int | Annual income ($000) |
| ZIP Code | int | Home ZIP code |
| Family | int | Family size (1–4) |
| CCAvg | float | Avg monthly credit card spend ($000) |
| Education | int | 1=Undergrad, 2=Graduate, 3=Advanced/Prof |
| Mortgage | int | Mortgage value ($000) |
| **Personal Loan** | int | **Target: 1=Accepted, 0=Rejected** |
| Securities Account | int | Has securities account (0/1) |
| CD Account | int | Has certificate of deposit account (0/1) |
| Online | int | Uses internet banking (0/1) |
| CreditCard | int | Has UniversalBank credit card (0/1) |

**5,000 customers | 9.6% loan acceptance rate | No missing values**

---

## 🔑 Key Findings

| Driver | Insight |
|---|---|
| Income | **2.2× income gap** – acceptors earn ~$144k vs ~$66k (rejectors) |
| CCAvg | Acceptors spend **$3.9k/mo** on CC vs $1.7k/mo |
| CD Account | CD holders accept at **46.4%** – 5× the base rate |
| Education | Graduate/Advanced customers accept at **3× the Undergrad rate** |
| Income Group | Customers earning **>$160k** show a **52.5% acceptance rate** |

---

## 📄 License
MIT License – free to use and modify.

---

*Built with ❤️ using Python, Streamlit, Plotly, and scikit-learn*
