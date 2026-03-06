"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSAL BANK – PERSONAL LOAN ANALYTICS DASHBOARD                  ║
║        Single-file Streamlit Application                                    ║
║        Author : Ayush Pancholi | SP Jain Global School of Management        ║
║                                                                              ║
║  Run locally :  streamlit run app.py                                         ║
║  Deploy      :  push folder to GitHub → connect on share.streamlit.io        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

# ── stdlib ────────────────────────────────────────────────────────────────────
import hashlib

# ── data ──────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# ── viz ───────────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# ── streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

ACCEPT_CLR  = "#2ECC71"
REJECT_CLR  = "#E74C3C"
NEUTRAL_CLR = "#3498DB"
WARM_CLR    = "#F39C12"
PURPLE_CLR  = "#9B59B6"

CHART_FONT  = dict(family="Inter, sans-serif", size=13, color="#2C3E50")
LAYOUT_BASE = dict(
    font=CHART_FONT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

EDU_MAP       = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
LOAN_MAP      = {0: "Rejected",  1: "Accepted"}
INCOME_BINS   = [0, 40, 80, 120, 160, 225]
INCOME_LABELS = ["<$40k", "$40–80k", "$80–120k", "$120–160k", ">$160k"]
CCAVG_BINS    = [0, 1, 2, 4, 10.01]
CCAVG_LABELS  = ["<$1k", "$1–2k", "$2–4k", ">$4k"]
AGE_BINS      = [22, 30, 40, 50, 60, 68]
AGE_LABELS    = ["23–30", "31–40", "41–50", "51–60", "61–67"]

FEATURE_COLS = [
    "Age", "Experience", "Income", "Family",
    "CCAvg", "Education", "Mortgage",
    "Securities Account", "CD Account", "Online", "CreditCard"
]
TARGET_COL = "Personal Loan"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, clip negatives, and add all derived columns."""
    df = df.dropna(how="all")
    df = df[df["ID"] != "ID"]
    for col in df.columns:
        if col != "ZIP Code":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL]    = df[TARGET_COL].astype(int)
    df["Experience"]  = df["Experience"].clip(lower=0)
    df["Edu_Label"]   = df["Education"].map(EDU_MAP)
    df["Loan_Status"] = df[TARGET_COL].map(LOAN_MAP)
    df["Income_Group"] = pd.cut(df["Income"], bins=INCOME_BINS,
                                labels=INCOME_LABELS, right=True)
    df["CCAvg_Group"]  = pd.cut(df["CCAvg"],  bins=CCAVG_BINS,
                                labels=CCAVG_LABELS, right=True)
    df["Age_Group"]    = pd.cut(df["Age"],    bins=AGE_BINS,
                                labels=AGE_LABELS, right=True)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_data(path: str = "data/UniversalBank.xlsx") -> pd.DataFrame:
    try:
        df = pd.read_excel(path, header=3)
    except FileNotFoundError:
        st.error("❌  `data/UniversalBank.xlsx` not found. Upload via the sidebar.")
        st.stop()
    return _clean(df)


def load_uploaded(f) -> pd.DataFrame:
    df = pd.read_excel(f, header=3)
    return _clean(df)


def apply_filters(df, income_range, edu_levels, family_sizes, loan_status):
    mask = (
        df["Income"].between(income_range[0], income_range[1]) &
        df["Education"].isin(edu_levels) &
        df["Family"].isin(family_sizes) &
        df[TARGET_COL].isin(loan_status)
    )
    return df[mask].copy()


def summary_stats(df):
    return (
        df.groupby("Loan_Status")[["Income", "CCAvg", "Mortgage", "Age"]]
        .mean().round(2).reset_index()
    )


def banking_service_rates(df):
    rows = []
    for svc in ["Securities Account", "CD Account", "Online", "CreditCard"]:
        for val in [0, 1]:
            sub  = df[df[svc] == val]
            rate = sub[TARGET_COL].mean() * 100
            rows.append({"Service": svc,
                         "Has Service": "Yes" if val else "No",
                         "Acceptance Rate (%)": round(rate, 2),
                         "Count": len(sub)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – MODEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model(df_hash: int, df: pd.DataFrame, model_name: str = "Random Forest"):
    """Train classifier → return metrics, curves, feature importances."""
    df_c = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df_c[FEATURE_COLS].copy()
    y = df_c[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()

    _models = {
        "Random Forest":      RandomForestClassifier(n_estimators=300, max_depth=10,
                                                      class_weight="balanced",
                                                      random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                          learning_rate=0.05,
                                                          random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   class_weight="balanced",
                                                   random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8,
                                                       class_weight="balanced",
                                                       random_state=42),
    }
    clf = _models[model_name]

    if model_name == "Logistic Regression":
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        clf.fit(Xtr, y_train)
        y_pred  = clf.predict(Xte)
        y_proba = clf.predict_proba(Xte)[:, 1]
    else:
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "ROC AUC":   round(roc_auc_score(y_test, y_proba) * 100, 2),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    if hasattr(clf, "feature_importances_"):
        fi = pd.DataFrame({"Feature": FEATURE_COLS,
                           "Importance": clf.feature_importances_})
    else:
        fi = pd.DataFrame({"Feature": FEATURE_COLS,
                           "Importance": np.abs(clf.coef_[0])})
    fi = fi.sort_values("Importance", ascending=False)

    return dict(model=clf, scaler=scaler, metrics=metrics,
                fpr=fpr, tpr=tpr, cm=cm, fi=fi,
                X_test=X_test, y_test=y_test,
                y_pred=y_pred, y_proba=y_proba,
                model_name=model_name)


def predict_all(result, df):
    clf, scaler = result["model"], result["scaler"]
    X_all = df[FEATURE_COLS].fillna(0)
    if result["model_name"] == "Logistic Regression":
        proba = clf.predict_proba(scaler.transform(X_all))[:, 1]
    else:
        proba = clf.predict_proba(X_all)[:, 1]
    out = df.copy()
    out["Loan_Probability"] = (proba * 100).round(2)
    out["Prediction"]       = (proba >= 0.5).astype(int)
    return out


def generate_offer(row):
    prob     = row.get("Loan_Probability", 50)
    income   = row.get("Income",  60)
    ccavg    = row.get("CCAvg",   1.5)
    mortgage = row.get("Mortgage", 0)
    edu      = row.get("Education", 1)
    cd       = row.get("CD Account", 0)
    age      = row.get("Age",    40)
    family   = row.get("Family", 2)
    online   = row.get("Online", 0)

    if prob >= 75:
        tier, rate_band, amult = "🔥 Hot Lead – Priority Offer",     (8.5, 10.5), 5.5
    elif prob >= 55:
        tier, rate_band, amult = "⭐ Warm Lead – Targeted Offer",    (10.5, 12.5), 4.5
    else:
        tier, rate_band, amult = "💡 Potential Lead – Awareness Offer", (12.5, 14.5), 3.5

    base = income * amult
    if cd:       base *= 1.1
    if mortgage: base *= 0.9
    loan_amount = round(min(base, income * 6), 1)

    rate = rate_band[0]
    if edu == 3:   rate -= 0.5
    if ccavg > 3:  rate -= 0.3
    if cd:         rate -= 0.4
    if age < 35:   rate += 0.3
    rate = round(max(rate_band[0], min(rate, rate_band[1])), 2)

    tenure = "3–5 years" if income > 100 else ("5–7 years" if income > 60 else "7–10 years")

    drivers = []
    if income > 120: drivers.append("high income profile")
    if ccavg  > 3:   drivers.append("strong credit card spending")
    if cd:           drivers.append("existing CD account relationship")
    if edu >= 2:     drivers.append("graduate-level education")
    if family >= 3:  drivers.append("growing family needs")
    key_reason = (" & ".join(drivers[:3]) if drivers else "demonstrated banking engagement").title()

    channel = "our UniversalBank app" if online else "your nearest branch"
    pitch   = (f"Exclusively for you: a pre-approved Personal Loan of "
               f"${loan_amount:,.0f}k at just {rate}% p.a. "
               f"Apply now via {channel} in under 5 minutes!")

    return {"Offer Tier": tier, "Loan Amount ($k)": loan_amount,
            "Interest Rate (%)": rate, "Tenure": tenure,
            "Key Driver": key_reason, "Personalised Pitch": pitch}


def build_offer_table(df_pred, min_prob=50.0):
    interested = df_pred[df_pred["Loan_Probability"] >= min_prob].copy()
    rows = []
    for _, row in interested.iterrows():
        offer = generate_offer(row)
        rows.append({
            "ID":           int(row["ID"]),
            "Age":          int(row["Age"]),
            "Income ($k)":  int(row["Income"]),
            "CCAvg ($k)":   round(row["CCAvg"], 2),
            "Education":    {1:"Undergrad",2:"Graduate",3:"Adv/Prof"}.get(int(row["Education"]),"–"),
            "Family":       int(row["Family"]),
            "Loan Prob (%)": row["Loan_Probability"],
            **offer
        })
    return pd.DataFrame(rows).sort_values("Loan Prob (%)", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – CHART FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _apply(fig):
    fig.update_layout(**LAYOUT_BASE)
    fig.update_xaxes(showgrid=True, gridcolor="#ECF0F1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ECF0F1", zeroline=False)
    return fig


# ── Descriptive ───────────────────────────────────────────────────────────────

def donut_loan_acceptance(df):
    counts        = df[TARGET_COL].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Label"] = counts["Status"].map({0: "Rejected", 1: "Accepted"})
    fig = go.Figure(go.Pie(
        labels=counts["Label"], values=counts["Count"], hole=0.60,
        marker_colors=[ACCEPT_CLR, REJECT_CLR],
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"
    ))
    fig.add_annotation(text=f"<b>{counts['Count'].sum():,}</b><br>Customers",
                       x=0.5, y=0.5, font_size=16, showarrow=False)
    fig.update_layout(title="Loan Acceptance Overview", **LAYOUT_BASE)
    return fig


def histogram_age(df):
    return _apply(px.histogram(
        df, x="Age", color="Loan_Status", nbins=30,
        barmode="overlay", opacity=0.75,
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Age Distribution by Loan Status",
        labels={"Age": "Age (years)", "count": "Customers"}
    ))


def histogram_income(df):
    return _apply(px.histogram(
        df, x="Income", color="Loan_Status", nbins=40,
        barmode="overlay", opacity=0.75,
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Income Distribution by Loan Status ($000)",
        labels={"Income": "Annual Income ($000)", "count": "Customers"}
    ))


def bar_family(df):
    g = df.groupby(["Family", "Loan_Status"]).size().reset_index(name="Count")
    return _apply(px.bar(
        g, x="Family", y="Count", color="Loan_Status", barmode="group",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Family Size Distribution by Loan Status",
        labels={"Family": "Family Size", "Count": "Customers"}
    ))


def bar_education(df):
    g = df.groupby(["Edu_Label", "Loan_Status"]).size().reset_index(name="Count")
    return _apply(px.bar(
        g, x="Edu_Label", y="Count", color="Loan_Status", barmode="group",
        category_orders={"Edu_Label": ["Undergrad", "Graduate", "Advanced/Prof"]},
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Education Level Distribution by Loan Status",
        labels={"Edu_Label": "Education Level", "Count": "Customers"}
    ))


def kpi_cards(df):
    accepted = df[TARGET_COL].sum()
    total    = len(df)
    return {
        "Total Customers":   f"{total:,}",
        "Loan Accepted":     f"{accepted:,}",
        "Acceptance Rate":   f"{accepted/total*100:.1f}%",
        "Avg Income ($k)":   f"${df['Income'].mean():.1f}k",
        "Avg CCAvg ($k)":    f"${df['CCAvg'].mean():.2f}k",
        "Avg Mortgage ($k)": f"${df['Mortgage'].mean():.1f}k",
    }


def box_income_group(df):
    return _apply(px.box(
        df, x="Income_Group", y="Income", color="Loan_Status",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Income Distribution Across Groups & Loan Status",
        labels={"Income_Group": "Income Group", "Income": "Annual Income ($000)"}
    ))


# ── Diagnostic ────────────────────────────────────────────────────────────────

def grouped_bar_acceptance_rate(df, col, col_label):
    g = df.groupby(col)[TARGET_COL].agg(["sum", "count"]).reset_index()
    g.columns = [col, "Accepted", "Total"]
    g["Rate"] = (g["Accepted"] / g["Total"] * 100).round(2)
    fig = px.bar(g, x=col, y="Rate",
                 text=g["Rate"].apply(lambda x: f"{x:.1f}%"),
                 color="Rate",
                 color_continuous_scale=["#FEF9E7", WARM_CLR, "#E74C3C"],
                 title=f"Personal Loan Acceptance Rate by {col_label}",
                 labels={col: col_label, "Rate": "Acceptance Rate (%)"})
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig)


def violin_income_vs_loan(df):
    return _apply(px.violin(
        df, x="Loan_Status", y="Income", color="Loan_Status",
        box=True, points="outliers",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Income Distribution: Loan Accepted vs Rejected",
        labels={"Loan_Status": "Loan Status", "Income": "Annual Income ($000)"}
    ))


def violin_ccavg_vs_loan(df):
    return _apply(px.violin(
        df, x="Loan_Status", y="CCAvg", color="Loan_Status",
        box=True, points="outliers",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Credit Card Avg Spend: Loan Accepted vs Rejected",
        labels={"Loan_Status": "Loan Status", "CCAvg": "CC Avg Spend ($000/month)"}
    ))


def scatter_income_ccavg(df):
    sample = df.sample(min(1500, len(df)), random_state=42)
    return _apply(px.scatter(
        sample, x="Income", y="CCAvg", color="Loan_Status",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        size="Mortgage", size_max=18, opacity=0.65,
        hover_data=["Age", "Education", "Family"],
        title="Income vs CC Avg Spend (bubble size = Mortgage)",
        labels={"Income": "Annual Income ($000)", "CCAvg": "CC Avg Spend ($000/month)"}
    ))


def heatmap_correlation(df):
    num_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                "Mortgage","Securities Account","CD Account","Online",
                "CreditCard","Personal Loan"]
    corr = df[num_cols].corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>",
        colorbar=dict(title="r")
    ))
    fig.update_layout(title="Feature Correlation Heatmap",
                      xaxis=dict(tickangle=-35), **LAYOUT_BASE)
    return fig


def stacked_bar_banking_services(df):
    rows = []
    for svc in ["Securities Account", "CD Account", "Online", "CreditCard"]:
        for val, label in [(0,"No"),(1,"Yes")]:
            sub = df[df[svc] == val]
            rows.append({"Service": svc, "Has Service": label,
                         "Acceptance Rate (%)": round(sub[TARGET_COL].mean()*100, 2)})
    g = pd.DataFrame(rows)
    fig = px.bar(g, x="Service", y="Acceptance Rate (%)", color="Has Service",
                 barmode="group",
                 text=g["Acceptance Rate (%)"].apply(lambda x: f"{x:.1f}%"),
                 color_discrete_map={"Yes": ACCEPT_CLR, "No": REJECT_CLR},
                 title="Loan Acceptance Rate by Banking Service Ownership")
    fig.update_traces(textposition="outside")
    return _apply(fig)


def sunburst_drilldown(df):
    g = (df.groupby(["Income_Group","Edu_Label","Loan_Status"])
           .size().reset_index(name="Count"))
    g = g[g["Count"] > 0]
    fig = px.sunburst(
        g, path=["Income_Group","Edu_Label","Loan_Status"],
        values="Count", color="Loan_Status",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR, "(?)": "#BDC3C7"},
        title="🔍 Drill-Down: Income Group → Education → Loan Status",
        branchvalues="total"
    )
    fig.update_traces(
        textinfo="label+percent entry",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percentEntry:.1%} of parent<extra></extra>"
    )
    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def treemap_income_family_loan(df):
    g = (df.groupby(["Income_Group","Family","Loan_Status"])
           .size().reset_index(name="Count"))
    g["Family_Label"] = g["Family"].apply(lambda x: f"Family {x}")
    g = g[g["Count"] > 0]
    fig = px.treemap(
        g, path=["Income_Group","Family_Label","Loan_Status"],
        values="Count", color="Loan_Status",
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR, "(?)": "#BDC3C7"},
        title="Treemap: Income Group → Family Size → Loan Status"
    )
    fig.update_traces(
        textinfo="label+value+percent entry",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>"
    )
    fig.update_layout(margin=dict(l=10,r=10,t=60,b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_income_group_acceptance(df):
    g = df.groupby("Income_Group")[TARGET_COL].agg(["sum","count"]).reset_index()
    g.columns = ["Income_Group","Accepted","Total"]
    g["Rate"] = (g["Accepted"] / g["Total"] * 100).round(2)
    fig = px.bar(g, x="Income_Group", y="Rate",
                 text=g["Rate"].apply(lambda r: f"{r:.1f}%"),
                 color="Rate",
                 color_continuous_scale=["#FDFEFE", WARM_CLR, "#C0392B"],
                 title="Personal Loan Acceptance Rate by Income Group",
                 labels={"Income_Group":"Income Group","Rate":"Acceptance Rate (%)"})
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig)


def radar_avg_comparison(df):
    cols   = ["Income","CCAvg","Mortgage","Age","Experience","Family"]
    groups = df.groupby("Loan_Status")[cols].mean()
    normed = (groups - groups.min()) / (groups.max() - groups.min() + 1e-9)
    fig    = go.Figure()
    colors = {"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR}
    for status in normed.index:
        vals = normed.loc[status].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cols+[cols[0]],
            fill="toself", name=status,
            line_color=colors[status], fillcolor=colors[status], opacity=0.35
        ))
    fig.update_layout(
        title="Normalised Feature Profiles: Accepted vs Rejected",
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        **LAYOUT_BASE
    )
    return fig


# ── Predictive ────────────────────────────────────────────────────────────────

def plot_feature_importance(fi_df):
    top = fi_df.head(11)
    fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                 text=top["Importance"].apply(lambda x: f"{x:.3f}"),
                 color="Importance",
                 color_continuous_scale=["#D6EAF8","#2980B9","#1A5276"],
                 title="Feature Importance / Coefficient Magnitude")
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(autorange="reversed"), **LAYOUT_BASE)
    return fig


def plot_roc_curve(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc_score:.2f}%)",
                             line=dict(color=NEUTRAL_CLR, width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Random Baseline",
                             line=dict(color=REJECT_CLR, dash="dash")))
    fig.update_layout(title="ROC Curve",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", **LAYOUT_BASE)
    return fig


def plot_confusion_matrix(cm):
    labels = ["Rejected","Accepted"]
    z      = cm[::-1]
    annots = []
    for i in range(2):
        for j in range(2):
            annots.append(dict(x=j, y=i, text=str(z[i][j]),
                               font=dict(color="white", size=18, family="Inter"),
                               showarrow=False))
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels[::-1],
        colorscale=[[0,"#ECF0F1"],[1,"#2980B9"]], showscale=False
    ))
    fig.update_layout(title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      annotations=annots, **LAYOUT_BASE)
    return fig


def plot_prob_histogram(df_pred):
    fig = px.histogram(
        df_pred, x="Loan_Probability", color="Loan_Status",
        nbins=40, barmode="overlay", opacity=0.75,
        color_discrete_map={"Accepted": ACCEPT_CLR, "Rejected": REJECT_CLR},
        title="Predicted Loan Probability Distribution",
        labels={"Loan_Probability": "Probability (%)"}
    )
    fig.add_vline(x=50, line_dash="dash", line_color="black",
                  annotation_text="Decision Threshold (50%)")
    return _apply(fig)


# ── Prescriptive ──────────────────────────────────────────────────────────────

def funnel_segments(df_pred):
    thresholds = [50, 60, 70, 80, 90]
    counts     = [len(df_pred[df_pred["Loan_Probability"] >= t]) for t in thresholds]
    labels     = [f"≥{t}% Probability" for t in thresholds]
    fig = go.Figure(go.Funnel(
        y=labels, x=counts,
        textinfo="value+percent initial",
        marker_color=[ACCEPT_CLR,"#27AE60","#1E8449","#196F3D","#145A32"]
    ))
    fig.update_layout(title="Lead Funnel: Customers by Loan Probability Tier",
                      **LAYOUT_BASE)
    return fig


def bar_offer_tiers(df_offers):
    g = df_offers["Offer Tier"].value_counts().reset_index()
    g.columns = ["Tier","Count"]
    cmap = {"🔥 Hot Lead – Priority Offer":      "#E74C3C",
            "⭐ Warm Lead – Targeted Offer":     "#F39C12",
            "💡 Potential Lead – Awareness Offer": "#3498DB"}
    fig = px.bar(g, x="Tier", y="Count", text="Count",
                 color="Tier", color_discrete_map=cmap,
                 title="Distribution of Personalised Offer Tiers")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, **LAYOUT_BASE)
    return fig


def scatter_offers(df_offers):
    return _apply(px.scatter(
        df_offers, x="Income ($k)", y="Loan Amount ($k)",
        color="Offer Tier", size="Loan Prob (%)", size_max=20,
        hover_data=["ID","Age","Education","Interest Rate (%)"],
        title="Personalised Loan Offers: Income vs Loan Amount",
        labels={"Income ($k)":"Annual Income ($k)",
                "Loan Amount ($k)":"Recommended Loan ($k)"}
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Universal Bank | Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.hero{background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
      padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;}
.hero h1{color:#FFF;font-size:2rem;font-weight:700;margin:0;}
.hero p{color:#A9CCE3;font-size:1rem;margin:.4rem 0 0;}

.kpi-card{background:#FFF;border-radius:12px;padding:1.2rem 1.5rem;
          box-shadow:0 2px 12px rgba(0,0,0,.07);border-left:5px solid #3498DB;}
.kpi-card.green{border-left-color:#2ECC71;}
.kpi-card.orange{border-left-color:#F39C12;}
.kpi-card.purple{border-left-color:#9B59B6;}
.kpi-card h3{font-size:1.6rem;font-weight:700;margin:0;color:#2C3E50;}
.kpi-card p{font-size:.82rem;color:#7F8C8D;margin:.2rem 0 0;}

.insight-box{background:linear-gradient(135deg,#EBF5FB,#FDFEFE);
             border-left:4px solid #2980B9;border-radius:8px;
             padding:.9rem 1.2rem;margin:.8rem 0;font-size:.91rem;color:#2C3E50;}
.insight-box.green{background:linear-gradient(135deg,#EAFAF1,#FDFEFE);border-color:#27AE60;}
.insight-box.orange{background:linear-gradient(135deg,#FEF9E7,#FDFEFE);border-color:#F39C12;}
.insight-box.red{background:linear-gradient(135deg,#FDEDEC,#FDFEFE);border-color:#E74C3C;}
.insight-box.purple{background:linear-gradient(135deg,#F4ECF7,#FDFEFE);border-color:#9B59B6;}

.section-header{font-size:1.25rem;font-weight:600;color:#1A1A2E;
                border-bottom:2px solid #3498DB;padding-bottom:.4rem;margin:1.4rem 0 1rem;}

.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;}
.stTabs [data-baseweb="tab"]{background:#EBF5FB;border-radius:8px;
                              padding:8px 18px;font-weight:500;}
.stTabs [aria-selected="true"]{background:#1A5276!important;color:#FFF!important;}

[data-testid="stSidebar"]{background:#1A1A2E;}
[data-testid="stSidebar"] *{color:#ECF0F1!important;}
</style>
""", unsafe_allow_html=True)

# ── sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Dataset (xlsx/csv)",
                                type=["xlsx","csv"],
                                help="Upload UniversalBank.xlsx, or the default file will be used.")

if uploaded is not None:
    try:
        df_full = (pd.read_csv(uploaded).pipe(lambda d: _clean(d))
                   if uploaded.name.endswith(".csv") else load_uploaded(uploaded))
        st.sidebar.success(f"✅ Loaded: **{uploaded.name}**  ({len(df_full):,} rows)")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
        df_full = load_data()
else:
    df_full = load_data()

# ── sidebar: filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Dashboard Filters")
    income_range = st.slider(
        "💰 Income Range ($000)",
        min_value=int(df_full["Income"].min()),
        max_value=int(df_full["Income"].max()),
        value=(int(df_full["Income"].min()), int(df_full["Income"].max())),
        step=5
    )
    edu_opts = {1:"1 – Undergrad", 2:"2 – Graduate", 3:"3 – Advanced/Prof"}
    edu_sel  = st.multiselect("🎓 Education Level",
                               options=list(edu_opts.keys()),
                               format_func=lambda x: edu_opts[x],
                               default=list(edu_opts.keys()))
    fam_sel  = st.multiselect("👨‍👩‍👧 Family Size", options=[1,2,3,4], default=[1,2,3,4])
    loan_sel = st.multiselect("🏷️ Loan Status", options=[0,1],
                               format_func=lambda x: "Accepted" if x else "Rejected",
                               default=[0,1])
    st.markdown("---")
    st.markdown("### 🤖 Predictive Model")
    model_choice     = st.selectbox("Select Classifier",
                                    ["Random Forest","Gradient Boosting",
                                     "Logistic Regression","Decision Tree"])
    predict_threshold = st.slider("Prediction Threshold (%)", 30, 80, 50, 5)
    st.markdown("---")
    st.caption("Dashboard by **Ayush Pancholi** | SP Jain Global")

# ── apply filters ─────────────────────────────────────────────────────────────
edu_sel  = edu_sel  or [1,2,3]
fam_sel  = fam_sel  or [1,2,3,4]
loan_sel = loan_sel or [0,1]

df = apply_filters(df_full, income_range, edu_sel, fam_sel, loan_sel)
if len(df) == 0:
    st.warning("⚠️  No data matches the current filters.")
    st.stop()

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏦 Universal Bank – Personal Loan Analytics Dashboard</h1>
  <p><b>Objective:</b> Understand which customers are most likely to accept a Personal Loan offer
  &nbsp;|&nbsp; 4-Layer Analytics: Descriptive · Diagnostic · Predictive · Prescriptive</p>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
kpis   = kpi_cards(df)
k_cols = st.columns(6)
colors = ["","green","orange","","orange","purple"]
for col, (label, val), color in zip(k_cols, kpis.items(), colors):
    col.markdown(f'<div class="kpi-card {color}"><h3>{val}</h3><p>{label}</p></div>',
                 unsafe_allow_html=True)
st.caption(f"🔎 **Showing {len(df):,} of {len(df_full):,} customers** after filters.")

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Predictive Analytics",
    "💡 Prescriptive Analytics"
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 – DESCRIPTIVE                                        ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-header">📊 Descriptive Analytics – Who are our customers?</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.plotly_chart(donut_loan_acceptance(df), use_container_width=True)
        pct = df[TARGET_COL].mean()*100
        st.markdown(
            f'<div class="insight-box green">✅ <b>{pct:.1f}%</b> of customers accepted '
            f'the loan ({df[TARGET_COL].sum():,} of {len(df):,}).</div>',
            unsafe_allow_html=True)
    with c2:
        st.plotly_chart(histogram_age(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box">📌 Age is broadly spread 23–67 yrs. '
            'No strong age-based skew — income & education are stronger drivers.</div>',
            unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(histogram_income(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box orange">💰 Acceptors are heavily concentrated '
            'in the <b>$80k+</b> income bracket.</div>', unsafe_allow_html=True)
    with c4:
        st.plotly_chart(bar_family(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box purple">👨‍👩‍👧 Families of size <b>3 & 4</b> show '
            'higher relative acceptance — greater financial need.</div>',
            unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(bar_education(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box">🎓 Graduate & Advanced/Prof holders show '
            '<b>~3×</b> higher acceptance vs Undergrads.</div>', unsafe_allow_html=True)
    with c6:
        st.plotly_chart(box_income_group(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box green">📦 Accepted customers sit firmly in '
            '<b>$80k–$160k+</b> income quartiles.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Average Statistics by Loan Status</div>',
                unsafe_allow_html=True)
    stats = summary_stats(df)
    stats_d = stats.rename(columns={
        "Loan_Status":"Loan Status","Income":"Avg Income ($k)",
        "CCAvg":"Avg CC Spend ($k/mo)","Mortgage":"Avg Mortgage ($k)","Age":"Avg Age"
    })
    cl, cr = st.columns([2,1])
    with cl:
        st.dataframe(
            stats_d.set_index("Loan Status")
            .style.background_gradient(cmap="RdYlGn", axis=0)
            .format("{:.2f}", subset=["Avg Income ($k)","Avg CC Spend ($k/mo)",
                                       "Avg Mortgage ($k)","Avg Age"]),
            use_container_width=True)
    with cr:
        acc = stats[stats["Loan_Status"]=="Accepted"]
        rej = stats[stats["Loan_Status"]=="Rejected"]
        if not acc.empty and not rej.empty:
            id_ = acc["Income"].values[0]  - rej["Income"].values[0]
            cd_ = acc["CCAvg"].values[0]   - rej["CCAvg"].values[0]
            md_ = acc["Mortgage"].values[0]- rej["Mortgage"].values[0]
            st.markdown(
                f'<div class="insight-box red">🔑 <b>Key Differentiators:</b><br>'
                f'• Income gap: <b>+${id_:.0f}k</b><br>'
                f'• CC Spend gap: <b>+${cd_:.2f}k/mo</b><br>'
                f'• Mortgage gap: <b>+${md_:.0f}k</b></div>',
                unsafe_allow_html=True)

    with st.expander("🗃️ Raw Data Sample (first 200 rows)"):
        dcols = ["ID","Age","Income","Family","CCAvg","Education",
                 "Mortgage","Personal Loan","Securities Account",
                 "CD Account","Online","CreditCard"]
        st.dataframe(df[dcols].head(200), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 – DIAGNOSTIC                                         ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-header">🔍 Diagnostic Analytics – What drives loan acceptance?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box orange">🔬 <b>Diagnostic Lens:</b> Root causes & key drivers '
        'behind loan acceptance — comparing income, CCAvg, mortgage and education between groups, '
        'and examining how banking service ownership influences outcomes.</div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(violin_income_vs_loan(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box green">📌 Acceptors: median ~<b>$144k</b> vs '
            '~$66k for rejectors — a <b>2.2× income gap</b>.</div>',
            unsafe_allow_html=True)
    with c2:
        st.plotly_chart(violin_ccavg_vs_loan(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box orange">💳 Acceptors spend <b>~$3.9k/mo</b> on CC '
            'vs ~$1.7k/mo — signals financial confidence & credit readiness.</div>',
            unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        edu_df = df.copy()
        edu_df["Education"] = edu_df["Edu_Label"]
        st.plotly_chart(grouped_bar_acceptance_rate(edu_df,"Education","Education Level"),
                        use_container_width=True)
        st.markdown(
            '<div class="insight-box">🎓 Graduate (<b>13.0%</b>) & Advanced/Prof '
            '(<b>13.7%</b>) accept at ~3× the Undergrad rate (<b>4.4%</b>).</div>',
            unsafe_allow_html=True)
    with c4:
        st.plotly_chart(bar_income_group_acceptance(df), use_container_width=True)
        st.markdown(
            '<div class="insight-box red">💡 Income ><b>$160k</b> → <b>52.5% acceptance</b> '
            'vs near-zero for <$40k.</div>', unsafe_allow_html=True)

    st.plotly_chart(scatter_income_ccavg(df), use_container_width=True)
    st.markdown(
        '<div class="insight-box purple">📊 Clear <b>top-right cluster</b> (high income + '
        'high CC spend) aligns almost entirely with loan acceptors.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏦 Banking Service Ownership vs Loan Acceptance</div>',
                unsafe_allow_html=True)
    st.plotly_chart(stacked_bar_banking_services(df), use_container_width=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    msgs = [
        ("CD Account",         "insight-box",        "🥇 <b>CD Account</b> holders accept at <b>46.4%</b> — nearly 5× base rate."),
        ("Securities Account", "insight-box green",  "📈 <b>Securities Account</b> holders: <b>11.5%</b> — modestly above base."),
        ("Online",             "insight-box orange", "💻 <b>Online banking</b> marginally higher (9.75% vs 9.38%)."),
        ("CreditCard",         "insight-box",        "💳 <b>CreditCard</b> ownership: near parity (9.73% vs 9.55%)."),
    ]
    for col, (_, cls, msg) in zip([sc1,sc2,sc3,sc4], msgs):
        col.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔗 Feature Correlation Analysis</div>',
                unsafe_allow_html=True)
    st.plotly_chart(heatmap_correlation(df), use_container_width=True)
    st.markdown(
        '<div class="insight-box green">🔗 <b>Income (r=0.50)</b> & <b>CCAvg (r=0.37)</b> '
        'have the highest correlation with Personal Loan. CD Account (r=0.32) also notable. '
        'Age & Experience are nearly collinear (r≈0.99).</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(radar_avg_comparison(df), use_container_width=True)
    with c6:
        st.markdown('<div class="section-header">🎯 Key Drivers Summary</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box green">
        <b>Top 5 Drivers of Loan Acceptance:</b><br><br>
        1. 💰 <b>Income</b> – Strongest predictor; >$100k customers are prime targets<br>
        2. 💳 <b>CCAvg</b> – High CC spend signals financial capacity<br>
        3. 🏦 <b>CD Account</b> – 46% acceptance; existing relationship drives trust<br>
        4. 🎓 <b>Education</b> – Graduate+ holders 3× more likely to accept<br>
        5. 👨‍👩‍👧 <b>Family Size</b> – Families of 3–4 show elevated need for financing
        </div>
        <br>
        <div class="insight-box orange">
        <b>What does NOT differentiate:</b><br><br>
        • Age & Experience near-identical across both groups<br>
        • Online banking & UniversalBank CreditCard show minimal differential<br>
        • ZIP Code / geography is not a strong discriminator
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔍 Interactive Drill-Down: Income → Education → Loan Status</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">💡 <b>How to use:</b> Click any segment to zoom in. '
        'Click the centre to zoom back out.</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(sunburst_drilldown(df), use_container_width=True)
    with c8:
        st.plotly_chart(treemap_income_family_loan(df), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 – PREDICTIVE                                         ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-header">🤖 Predictive Analytics – Who will accept next?</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-box">🧪 A <b>{model_choice}</b> classifier is trained on 75% '
        f'of the full dataset (stratified). 11 features used.</div>',
        unsafe_allow_html=True)

    with st.spinner(f"⚙️ Training {model_choice}..."):
        df_hash = int(hashlib.md5(
            df_full[["Income","CCAvg",TARGET_COL]].values.tobytes()
        ).hexdigest(), 16) % (10**8)
        result = train_model(df_hash, df_full, model_choice)

    metrics  = result["metrics"]
    m_labels = list(metrics.keys())
    m_vals   = list(metrics.values())
    m_clrs   = ["","green","orange","purple",""]
    mc       = st.columns(5)
    for col, lbl, val, clr in zip(mc, m_labels, m_vals, m_clrs):
        col.markdown(f'<div class="kpi-card {clr}"><h3>{val}%</h3><p>{lbl}</p></div>',
                     unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_feature_importance(result["fi"]), use_container_width=True)
        st.markdown(
            f'<div class="insight-box green">🏆 <b>{result["fi"].iloc[0]["Feature"]}</b> '
            f'is the most important feature.</div>', unsafe_allow_html=True)
    with c2:
        st.plotly_chart(
            plot_roc_curve(result["fpr"], result["tpr"], metrics["ROC AUC"]),
            use_container_width=True)
        st.markdown(
            f'<div class="insight-box orange">📈 ROC AUC of <b>{metrics["ROC AUC"]}%</b> — '
            f'excellent discrimination well above random baseline.</div>',
            unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_confusion_matrix(result["cm"]), use_container_width=True)
    with c4:
        df_pred = predict_all(result, df_full)
        st.plotly_chart(plot_prob_histogram(df_pred), use_container_width=True)
        st.markdown(
            '<div class="insight-box purple">📊 Most customers cluster near 0% probability, '
            'confirming the 9.6% class imbalance. The model concentrates high probabilities '
            'on a tight group of likely acceptors.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔎 Customer-Level Prediction Explorer</div>',
                unsafe_allow_html=True)
    df_pred["Prediction_Custom"] = (df_pred["Loan_Probability"] >= predict_threshold).astype(int)
    n_accept = df_pred["Prediction_Custom"].sum()
    st.markdown(
        f'<div class="insight-box">At the <b>{predict_threshold}%</b> threshold, '
        f'<b>{n_accept:,} customers</b> ({n_accept/len(df_pred)*100:.1f}%) '
        f'are predicted to accept.</div>', unsafe_allow_html=True)

    show_cols = ["ID","Age","Income","CCAvg","Education","Family",
                 "Mortgage",TARGET_COL,"Loan_Probability","Prediction_Custom"]
    filt = st.radio("Filter predictions", ["All","Predicted Accepted","Predicted Rejected"],
                    horizontal=True)
    pds = df_pred[show_cols].copy()
    pds.columns = [*show_cols[:-2], "Actual Loan","Prob (%)","Predicted"]
    pds["Predicted"]   = pds["Predicted"].map({1:"✅ Accept",0:"❌ Reject"})
    pds["Actual Loan"] = pds["Actual Loan"].map({1:"✅ Accept",0:"❌ Reject"})
    if filt == "Predicted Accepted":
        pds = pds[pds["Predicted"]=="✅ Accept"]
    elif filt == "Predicted Rejected":
        pds = pds[pds["Predicted"]=="❌ Reject"]
    st.dataframe(pds.sort_values("Prob (%)", ascending=False).head(300),
                 use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 – PRESCRIPTIVE                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="section-header">💡 Prescriptive Analytics – Who to target and how?</div>',
                unsafe_allow_html=True)

    if "df_pred" not in dir():
        df_pred = predict_all(result, df_full)

    st.plotly_chart(funnel_segments(df_pred), use_container_width=True)

    st.markdown('<div class="section-header">🎯 Strategic Marketing Recommendations</div>',
                unsafe_allow_html=True)
    r1,r2,r3 = st.columns(3)
    with r1:
        st.markdown("""<div class="insight-box green">
        <b>🔥 Segment 1: High-Income Professionals</b><br><br>
        <b>Profile:</b> Income >$100k, Graduate/Adv edu, Family 3–4<br>
        <b>Channel:</b> Relationship manager + personalised email<br>
        <b>Offer:</b> Pre-approved loan, 8.5–10% rate<br>
        <b>Expected ROI:</b> Highest – ~35–52% conversion
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class="insight-box orange">
        <b>⭐ Segment 2: CD Account Holders</b><br><br>
        <b>Profile:</b> Any income, existing CD relationship<br>
        <b>Channel:</b> In-app notification + branch manager<br>
        <b>Offer:</b> Loyalty rate discount (−0.4%)<br>
        <b>Expected ROI:</b> High – 46% base acceptance rate
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class="insight-box purple">
        <b>💳 Segment 3: High CC Spenders</b><br><br>
        <b>Profile:</b> CCAvg >$3k/month<br>
        <b>Channel:</b> CC statement inserts + push notifications<br>
        <b>Offer:</b> Debt consolidation at better rate<br>
        <b>Expected ROI:</b> Medium-High – 40% acceptance
        </div>""", unsafe_allow_html=True)

    r4,r5,r6 = st.columns(3)
    with r4:
        st.markdown("""<div class="insight-box">
        <b>📚 Segment 4: Young Graduates</b><br><br>
        <b>Profile:</b> Age 25–35, Graduate edu, Income $40–80k<br>
        <b>Channel:</b> Digital / mobile-first, social retargeting<br>
        <b>Offer:</b> Starter loan, flexible 7–10 yr tenure<br>
        <b>Expected ROI:</b> Medium – growing segment
        </div>""", unsafe_allow_html=True)
    with r5:
        st.markdown("""<div class="insight-box red">
        <b>🏠 Segment 5: Mortgage Holders</b><br><br>
        <b>Profile:</b> Mortgage >$100k, Income >$80k<br>
        <b>Channel:</b> Dedicated banker outreach<br>
        <b>Offer:</b> Home improvement / top-up loan<br>
        <b>Expected ROI:</b> Medium – already credit-active
        </div>""", unsafe_allow_html=True)
    with r6:
        st.markdown("""<div class="insight-box">
        <b>🚫 Avoid / Nurture: Low-Income Segment</b><br><br>
        <b>Profile:</b> Income <$40k, Undergrad<br>
        <b>Action:</b> Do NOT target with loans — ~0% conversion<br>
        <b>Alternative:</b> Savings products, financial literacy<br>
        <b>Expected ROI:</b> Low for loans
        </div>""", unsafe_allow_html=True)

    # ── Personalised offer engine ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🎁 Personalised Loan Offer Engine</div>',
                unsafe_allow_html=True)
    offer_threshold = st.slider("Minimum Loan Probability for offers (%)",
                                min_value=30, max_value=85, value=50, step=5,
                                key="offer_thresh")

    with st.spinner("🔄 Generating personalised offers..."):
        df_offers = build_offer_table(df_pred, min_prob=float(offer_threshold))

    st.markdown(
        f'<div class="insight-box green">🎯 <b>{len(df_offers):,} customers</b> qualify '
        f'for a personalised offer at ≥{offer_threshold}% probability.</div>',
        unsafe_allow_html=True)

    if len(df_offers) > 0:
        oc1, oc2 = st.columns(2)
        with oc1:
            st.plotly_chart(bar_offer_tiers(df_offers), use_container_width=True)
        with oc2:
            st.plotly_chart(scatter_offers(df_offers), use_container_width=True)

        st.markdown('<div class="section-header">📋 Full Personalised Offer Table</div>',
                    unsafe_allow_html=True)
        tier_filter = st.multiselect("Filter by Offer Tier",
                                      options=df_offers["Offer Tier"].unique().tolist(),
                                      default=df_offers["Offer Tier"].unique().tolist())
        disp_cols = ["ID","Age","Income ($k)","CCAvg ($k)","Education",
                     "Loan Prob (%)","Offer Tier","Loan Amount ($k)",
                     "Interest Rate (%)","Tenure","Key Driver","Personalised Pitch"]
        filtered_offers = df_offers[df_offers["Offer Tier"].isin(tier_filter)]
        st.dataframe(
            filtered_offers[disp_cols].sort_values("Loan Prob (%)", ascending=False)
            .reset_index(drop=True),
            use_container_width=True, height=500
        )
        st.download_button(
            label="⬇️  Download Personalised Offers as CSV",
            data=filtered_offers[disp_cols].to_csv(index=False),
            file_name="universal_bank_personalised_offers.csv",
            mime="text/csv", type="primary"
        )
    else:
        st.warning("No customers qualify at this threshold. Try lowering the slider.")

    st.markdown('<div class="section-header">📌 Campaign Strategy Summary</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box orange">
    <b>🗺️ Recommended Campaign Roadmap:</b><br><br>
    <b>Phase 1 (Month 1–2):</b> Target Hot Leads (≥75% probability) with relationship
    manager outreach and pre-approved offers. Highest ROI.<br><br>
    <b>Phase 2 (Month 3–4):</b> Warm Lead campaigns via email, in-app notifications
    and CC statement inserts. Focus on income >$80k and high CC spenders.<br><br>
    <b>Phase 3 (Month 5–6):</b> Awareness campaigns for Potential Leads.
    Target Graduate/Advanced education customers with income $40–80k.<br><br>
    <b>Phase 4 (Ongoing):</b> Monitor CD Account cross-sell continuously —
    every new CD account holder is a strong personal loan prospect.
    </div>
    """, unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#95A5A6;font-size:.82rem;'>"
    "🏦 Universal Bank Personal Loan Analytics Dashboard &nbsp;|&nbsp; "
    "Built with Streamlit & Plotly &nbsp;|&nbsp; "
    "Ayush Pancholi – SP Jain Global School of Management"
    "</center>",
    unsafe_allow_html=True
)
