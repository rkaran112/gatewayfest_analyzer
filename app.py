import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud

matplotlib.use("Agg")
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


DATA_FILE = "C5-FestDataset - fest_dataset - C5-FestDataset - fest_dataset.csv"

st.set_page_config(
    page_title="FestPulse · GATEWAYS 2025",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }
:root {
  --bg:#0A0E1A; --surface:#111827; --border:#1E2D45;
  --accent:#00D4FF; --accent2:#FF6B35; --ok:#00C896;
  --warn:#FFD166; --fail:#EF476F; --text:#E8EDF5; --muted:#6B7A99;
}
.stApp { background: var(--bg) !important; }
body, p, div, label, span { font-family:'DM Sans',sans-serif !important; color:var(--text) !important; }
h1, h2, h3 { font-family:'Syne',sans-serif !important; letter-spacing:0.12em; text-transform:uppercase; }
[class*='metric'] * { font-family:'JetBrains Mono',monospace !important; }
[data-testid='metric-container'] {
  background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:4px !important; border-left:3px solid var(--accent) !important;
  padding:1rem 1.25rem !important;
}
[data-testid='metric-container']:nth-child(2n) { border-left-color:var(--accent2) !important; }
[data-testid='metric-container']:nth-child(3n) { border-left-color:var(--ok) !important; }
[data-testid='stMetricValue'] { font-size:1.8rem !important; color:var(--accent) !important; }
[data-testid='stMetricLabel'] { font-size:0.68rem !important; color:var(--muted) !important; letter-spacing:0.2em; text-transform:uppercase; }
.stTabs [data-baseweb='tab-list'] { background:transparent; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb='tab'] { font-family:'Syne',sans-serif !important; font-size:0.75rem !important;
  letter-spacing:0.18em; text-transform:uppercase; color:var(--muted) !important;
  padding:0.7rem 1.5rem; border-bottom:2px solid transparent; background:transparent; }
.stTabs [aria-selected='true'] { color:var(--accent) !important; border-bottom-color:var(--accent) !important; }
[data-testid='stSidebar'] { background:#080C16 !important; border-right:1px solid var(--border) !important; }
.stTextInput input { background:var(--surface) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; border-radius:3px; font-family:'JetBrains Mono' !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:4px; }
.stDataFrame th { background:var(--surface) !important; color:var(--accent) !important;
  font-family:'JetBrains Mono' !important; font-size:0.72rem !important;
  letter-spacing:0.1em; text-transform:uppercase; }
.stDataFrame td { font-family:'JetBrains Mono' !important; font-size:0.8rem !important; }
::-webkit-scrollbar { width:4px; background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>""",
    unsafe_allow_html=True,
)

COLORS = ["#00D4FF", "#FF6B35", "#00C896", "#B44FFF", "#FFD166", "#EF476F", "#4FC3F7"]
AXIS = dict(
    gridcolor="#1E2D45",
    linecolor="#1E2D45",
    tickfont=dict(color="#6B7A99", family="JetBrains Mono", size=10),
)


def chart_layout(title: str = "", height: int = 400, **kwargs):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,18,32,0.95)",
        font=dict(family="DM Sans", color="#E8EDF5"),
        title=dict(
            text=title.upper(),
            font=dict(family="Syne", size=13, color="#6B7A99"),
            x=0,
            xanchor="left",
            pad=dict(l=8),
        ),
        xaxis=AXIS,
        yaxis=AXIS,
        colorway=COLORS,
        legend=dict(font=dict(color="#6B7A99", size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=height,
        **kwargs,
    )


def _minmax_100(series: pd.Series) -> pd.Series:
    """Scale a metric to 0-100 with safe handling for flat series."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v = float(numeric.min()) if len(numeric) else 0.0
    max_v = float(numeric.max()) if len(numeric) else 0.0
    if max_v == min_v:
        return pd.Series(50.0, index=numeric.index, dtype=float)
    return ((numeric - min_v) / (max_v - min_v) * 100.0).astype(float)


def build_benchmark_frame(data: pd.DataFrame, dimension: str, weights: dict[str, float]) -> pd.DataFrame:
    if len(data) == 0 or dimension not in data.columns:
        return pd.DataFrame()

    bench = (
        data.groupby(dimension)
        .agg(
            Participants=("Student Name", "count"),
            Avg_Rating=("Rating", "mean"),
            Positive_Rate=("Sentiment", lambda x: (x == "Positive").mean() * 100),
            Revenue=("Amount Paid", "sum"),
        )
        .reset_index()
    )
    if len(bench) == 0:
        return bench

    bench["N_Participants"] = _minmax_100(bench["Participants"])
    bench["N_Rating"] = _minmax_100(bench["Avg_Rating"])
    bench["N_Positive"] = _minmax_100(bench["Positive_Rate"])
    bench["N_Revenue"] = _minmax_100(bench["Revenue"])

    weight_sum = sum(weights.values()) or 1.0
    bench["Benchmark_Score"] = (
        bench["N_Participants"] * weights["participants"]
        + bench["N_Rating"] * weights["rating"]
        + bench["N_Positive"] * weights["sentiment"]
        + bench["N_Revenue"] * weights["revenue"]
    ) / weight_sum
    bench["Benchmark_Score"] = bench["Benchmark_Score"].round(2)

    total_revenue = bench["Revenue"].sum()
    bench["Revenue_Share"] = ((bench["Revenue"] / total_revenue) * 100).fillna(0.0) if total_revenue > 0 else 0.0
    bench["Revenue_Share"] = bench["Revenue_Share"].round(2)

    bench = bench.sort_values(["Benchmark_Score", "Participants"], ascending=[False, False]).reset_index(drop=True)
    bench["Rank"] = range(1, len(bench) + 1)
    return bench


EMOTION_LEXICON = {
    "Joy": {"excellent", "fun", "loved", "engaging", "best", "fantastic", "creative", "stimulating"},
    "Trust": {"organized", "managed", "smooth", "efficient", "helpful", "insightful", "constructive"},
    "Frustration": {"poor", "delay", "late", "crowded", "slow", "issue", "problem"},
    "Anger": {"bad", "worst", "unfair", "waste", "disappointed", "annoyed"},
    "Fear": {"difficult", "hard", "unclear", "confusing", "stress", "stressed"},
}

ROOT_CAUSE_THEMES = {
    "Logistics": {"logistics", "venue", "room", "seat", "crowd", "management"},
    "Mentorship": {"mentor", "mentors", "guidance", "support", "helpful"},
    "Scheduling": {"time", "timing", "schedule", "delay", "late"},
    "Judging": {"judge", "judges", "evaluation", "feedback", "score"},
    "Registration": {"registration", "register", "queue", "process"},
    "Networking": {"network", "opportunities", "interaction", "connect"},
}


def tokenize_feedback(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", str(text).lower())


def detect_emotion(text: str) -> str:
    tokens = set(tokenize_feedback(text))
    if not tokens:
        return "Neutral"
    scores = {k: len(tokens.intersection(v)) for k, v in EMOTION_LEXICON.items()}
    best = max(scores.values()) if scores else 0
    if best == 0:
        return "Neutral"
    return sorted([k for k, v in scores.items() if v == best])[0]


def confidence_score(text: str, polarity: float, sentiment: str) -> float:
    tokens = tokenize_feedback(text)
    richness = min(len(tokens) / 20.0, 1.0)
    polarity_strength = min(abs(float(polarity)) / 0.5, 1.0)
    sentiment_bonus = 0.1 if sentiment in {"Positive", "Negative"} else 0.0
    score = (0.45 * richness + 0.45 * polarity_strength + sentiment_bonus) * 100
    return round(max(0.0, min(score, 100.0)), 2)


def detect_themes(text: str) -> list[str]:
    tokens = set(tokenize_feedback(text))
    hits = [theme for theme, words in ROOT_CAUSE_THEMES.items() if tokens.intersection(words)]
    return hits or ["General"]


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    df["Feedback on Fest"] = df["Feedback on Fest"].fillna("")

    def analyze(text):
        pol = TextBlob(str(text)).sentiment.polarity
        sub = TextBlob(str(text)).sentiment.subjectivity
        label = "Positive" if pol > 0.05 else ("Negative" if pol < -0.05 else "Neutral")
        return pd.Series(
            {"Polarity": round(pol, 4), "Subjectivity": round(sub, 4), "Sentiment": label}
        )

    df[["Polarity", "Subjectivity", "Sentiment"]] = df["Feedback on Fest"].apply(analyze)
    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset file not found: `{DATA_FILE}`")
    st.info("Keep the CSV in the same folder as `app.py` and rerun Streamlit.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

with st.sidebar:
    st.markdown(
        """
    <div style="font-family:'Syne';font-size:1.4rem;font-weight:800;
                color:#00D4FF;letter-spacing:0.2em;padding:1rem 0 0.3rem;">⚡ FESTPULSE</div>
    <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#1E2D45;
                letter-spacing:0.25em;padding-bottom:1rem;border-bottom:1px solid #1E2D45;">
        GATEWAYS · 2025 · CHRIST UNIVERSITY</div>
    """,
        unsafe_allow_html=True,
    )

    sel_states = st.multiselect("State", sorted(df["State"].dropna().unique()), default=[])
    sel_events = st.multiselect("Event", sorted(df["Event Name"].dropna().unique()), default=[])
    sel_type = st.selectbox("Event Type", ["All", "Individual", "Group"])
    sel_rating = st.slider("Min Rating", 1, 5, 1)
    sel_sentiment = st.multiselect(
        "Sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    fdf = df.copy()
    if sel_states:
        fdf = fdf[fdf["State"].isin(sel_states)]
    if sel_events:
        fdf = fdf[fdf["Event Name"].isin(sel_events)]
    if sel_type != "All":
        fdf = fdf[fdf["Event Type"] == sel_type]
    fdf = fdf[fdf["Rating"] >= sel_rating]
    fdf = fdf[fdf["Sentiment"].isin(sel_sentiment)]

    pct = len(fdf) / len(df) * 100 if len(df) > 0 else 0
    st.markdown(
        f"""<div style='font-family:JetBrains Mono;font-size:0.72rem;color:#6B7A99;line-height:1.8;'>
        RECORDS &nbsp;<span style='color:#00D4FF'>{len(fdf):,}</span>/{len(df):,}<br/>
        COVERAGE &nbsp;<span style='color:#FF6B35'>{pct:.1f}%</span></div>""",
        unsafe_allow_html=True,
    )
    st.download_button("⬇ Export CSV", fdf.to_csv(index=False), "festpulse_filtered.csv", "text/csv")

st.markdown(
    """
<div style="padding:1.5rem 0 0.5rem;border-bottom:1px solid #1E2D45;margin-bottom:1.5rem;">
  <div style="font-family:JetBrains Mono;font-size:0.65rem;color:#00D4FF;letter-spacing:0.35em;margin-bottom:0.3rem;">
    CHRIST UNIVERSITY · DEPT. OF COMPUTER SCIENCE · NATIONAL LEVEL FEST</div>
  <div style="font-family:Syne;font-size:3rem;font-weight:800;color:#E8EDF5;line-height:1;">
    GATEWAYS <span style='color:#00D4FF'>2025</span></div>
  <div style="font-family:DM Sans;font-size:0.9rem;color:#6B7A99;margin-top:0.4rem;">
    Participation Intelligence Terminal · FestPulse v1.0</div>
</div>""",
    unsafe_allow_html=True,
)

k = st.columns(6)
top_evt = fdf["Event Name"].value_counts().idxmax() if len(fdf) > 0 else "—"
k[0].metric("⚡ Participants", f"{len(fdf):,}")
k[1].metric("🏛️ Colleges", fdf["College"].nunique())
k[2].metric("🗺️ States", fdf["State"].nunique())
k[3].metric("⭐ Avg Rating", f"{fdf['Rating'].mean():.2f}" if len(fdf) > 0 else "—")
k[4].metric("💰 Revenue", f"₹{fdf['Amount Paid'].sum():,.0f}")
k[5].metric("🏆 Top Event", top_evt)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️  ACT I — CARTOGRAPHY", "🎤  ACT II — THE SIGNAL", "🎛️  ACT III — COMMAND", "📊  ACT IV — BENCHMARK"]
)

with tab1:
    state_agg = (
        fdf.groupby("State")
        .agg(
            Participants=("Student Name", "count"),
            Colleges=("College", "nunique"),
            Revenue=("Amount Paid", "sum"),
        )
        .reset_index()
    )

    INDIA_GEO = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    if len(state_agg) > 0:
        fig_map = px.choropleth(
            state_agg,
            geojson=INDIA_GEO,
            featureidkey="properties.ST_NM",
            locations="State",
            color="Participants",
            hover_data=["Colleges", "Revenue"],
            color_continuous_scale=[[0, "#0A0E1A"], [0.3, "#003D5C"], [0.7, "#005F80"], [1, "#00D4FF"]],
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(**chart_layout("Participant Density — India", height=520))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No records for current filters.")

    c1, c2 = st.columns(2)
    with c1:
        ev = fdf["Event Name"].value_counts().reset_index()
        ev.columns = ["Event", "Count"]
        fig_ev = px.bar(
            ev,
            x="Count",
            y="Event",
            orientation="h",
            color="Count",
            color_continuous_scale=["#003D5C", "#00D4FF"],
        )
        fig_ev.update_layout(**chart_layout("Events — Participation Count", height=380))
        st.plotly_chart(fig_ev, use_container_width=True)
    with c2:
        top_c = fdf["College"].value_counts().head(10).reset_index()
        top_c.columns = ["College", "Count"]
        fig_c = px.bar(top_c, x="Count", y="College", orientation="h", color_discrete_sequence=["#FF6B35"])
        fig_c.update_layout(**chart_layout("Top 10 Colleges", height=380))
        st.plotly_chart(fig_c, use_container_width=True)

    if len(state_agg) > 0:
        fig_tree = px.treemap(
            state_agg,
            path=["State"],
            values="Participants",
            color="Revenue",
            color_continuous_scale=["#111827", "#FF6B35"],
            hover_data=["Revenue", "Colleges"],
        )
        fig_tree.update_layout(**chart_layout("State × Revenue Treemap", height=400))
        st.plotly_chart(fig_tree, use_container_width=True)

with tab2:
    avg_pol = fdf["Polarity"].mean() if len(fdf) > 0 else 0
    avg_sub = fdf["Subjectivity"].mean() if len(fdf) > 0 else 0
    direction = (
        "POSITIVE SIGNAL" if avg_pol > 0.05 else ("NEGATIVE SIGNAL" if avg_pol < -0.05 else "NEUTRAL SIGNAL")
    )
    sig_color = "#00C896" if avg_pol > 0.05 else ("#EF476F" if avg_pol < -0.05 else "#FFD166")
    bar_w = min(abs(avg_pol) * 400, 100)
    st.markdown(
        f"""
    <div style="background:#111827;border:1px solid #1E2D45;border-radius:4px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
      <div style="font-family:Syne;font-size:0.65rem;letter-spacing:0.25em;color:#6B7A99;margin-bottom:0.5rem;">AGGREGATE SENTIMENT SIGNAL</div>
      <div style="font-family:JetBrains Mono;font-size:1.6rem;color:{sig_color};font-weight:600;">{direction}</div>
      <div style="background:#0A0E1A;height:6px;border-radius:3px;margin-top:0.7rem;">
        <div style="background:{sig_color};height:6px;border-radius:3px;width:{bar_w:.1f}%;"></div>
      </div>
      <div style="font-family:JetBrains Mono;font-size:0.7rem;color:#6B7A99;margin-top:0.4rem;">
        polarity={avg_pol:.4f} | subjectivity={avg_sub:.4f}
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    signal_df = fdf.copy()
    signal_df["FeedbackTokens"] = signal_df["Feedback on Fest"].apply(tokenize_feedback)
    signal_df["Emotion"] = signal_df["Feedback on Fest"].apply(detect_emotion)
    signal_df["SignalConfidence"] = signal_df.apply(
        lambda r: confidence_score(r["Feedback on Fest"], r["Polarity"], r["Sentiment"]), axis=1
    )
    signal_df["ConfidenceBand"] = pd.cut(
        signal_df["SignalConfidence"], bins=[-1, 40, 70, 100], labels=["Low", "Medium", "High"]
    )
    signal_df["ThemeHits"] = signal_df["Feedback on Fest"].apply(detect_themes)

    # Emotion spectrum + classic sentiment matrix.
    c1, c2 = st.columns([1, 1.6])
    with c1:
        emotion_counts = signal_df["Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        fig_em = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count",
            color="Emotion",
            color_discrete_sequence=COLORS,
        )
        fig_em.update_layout(**chart_layout("Emotion Spectrum", height=340))
        st.plotly_chart(fig_em, use_container_width=True)
    with c2:
        pivot = signal_df.groupby(["Event Name", "Sentiment"]).size().unstack(fill_value=0)
        fig_h = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#0A0E1A"], [0.5, "#005F80"], [1, "#00D4FF"]],
                text=pivot.values,
                texttemplate="%{text}",
                textfont=dict(family="JetBrains Mono"),
                showscale=True,
            )
        )
        fig_h.update_layout(**chart_layout("Event × Sentiment Matrix", height=340))
        st.plotly_chart(fig_h, use_container_width=True)

    # Confidence layer.
    conf1, conf2 = st.columns([1, 1.3])
    with conf1:
        avg_conf = signal_df["SignalConfidence"].mean() if len(signal_df) else 0.0
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=avg_conf,
                number={"suffix": " / 100", "font": {"family": "JetBrains Mono", "color": "#E8EDF5"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6B7A99"},
                    "bar": {"color": "#00D4FF"},
                    "steps": [
                        {"range": [0, 40], "color": "#2B0F19"},
                        {"range": [40, 70], "color": "#3A2F12"},
                        {"range": [70, 100], "color": "#0B2A25"},
                    ],
                },
            )
        )
        fig_g.update_layout(**chart_layout("Sentiment Confidence Gauge", height=330))
        st.plotly_chart(fig_g, use_container_width=True)
    with conf2:
        conf_dist = signal_df["ConfidenceBand"].value_counts(dropna=False).reset_index()
        conf_dist.columns = ["Band", "Count"]
        band_color = {"Low": "#EF476F", "Medium": "#FFD166", "High": "#00C896"}
        fig_cd = px.bar(conf_dist, x="Band", y="Count", color="Band", color_discrete_map=band_color)
        fig_cd.update_layout(**chart_layout("Confidence Distribution", height=330))
        st.plotly_chart(fig_cd, use_container_width=True)

    # Risk radar.
    dim_choice = st.selectbox("Risk/Trend Dimension", ["Event Name", "State", "College"], key="act2_dim_choice")
    risk_df = (
        signal_df.groupby(dim_choice)
        .agg(
            Participants=("Student Name", "count"),
            Avg_Rating=("Rating", "mean"),
            Negative_Rate=("Sentiment", lambda x: (x == "Negative").mean() * 100),
            Avg_Confidence=("SignalConfidence", "mean"),
        )
        .reset_index()
    )
    risk_df["RiskScore"] = (
        _minmax_100(risk_df["Negative_Rate"]) * 0.5
        + _minmax_100(5 - risk_df["Avg_Rating"]) * 0.35
        + _minmax_100(100 - risk_df["Avg_Confidence"]) * 0.15
    )
    risk_df = risk_df.sort_values("RiskScore", ascending=False).reset_index(drop=True)

    r1, r2 = st.columns([1.2, 1])
    with r1:
        worst = risk_df.head(min(8, len(risk_df)))
        fig_risk = px.bar(
            worst.sort_values("RiskScore", ascending=True),
            x="RiskScore",
            y=dim_choice,
            orientation="h",
            color="RiskScore",
            color_continuous_scale=["#111827", "#EF476F"],
            hover_data=["Negative_Rate", "Avg_Rating", "Avg_Confidence", "Participants"],
        )
        fig_risk.update_layout(**chart_layout(f"Risk Radar — {dim_choice}", height=360))
        st.plotly_chart(fig_risk, use_container_width=True)
    with r2:
        risk_table = worst[[dim_choice, "RiskScore", "Negative_Rate", "Avg_Rating", "Avg_Confidence", "Participants"]].copy()
        risk_table["RiskScore"] = risk_table["RiskScore"].round(2)
        risk_table["Negative_Rate"] = risk_table["Negative_Rate"].round(2)
        risk_table["Avg_Rating"] = risk_table["Avg_Rating"].round(2)
        risk_table["Avg_Confidence"] = risk_table["Avg_Confidence"].round(2)
        st.markdown(
            "<div style='font-family:Syne;font-size:0.7rem;letter-spacing:0.2em;color:#EF476F;padding:0.4rem 0 0.5rem;'>RISK ALERT TABLE</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(risk_table, use_container_width=True, height=360)

    # Signal trend map.
    trend_df = (
        signal_df.groupby(dim_choice)
        .agg(
            Participants=("Student Name", "count"),
            Avg_Polarity=("Polarity", "mean"),
            Avg_Subjectivity=("Subjectivity", "mean"),
            Positive_Rate=("Sentiment", lambda x: (x == "Positive").mean() * 100),
        )
        .reset_index()
    )
    fig_trend = px.scatter(
        trend_df,
        x="Avg_Polarity",
        y="Positive_Rate",
        size="Participants",
        color="Avg_Subjectivity",
        hover_name=dim_choice,
        color_continuous_scale=["#003D5C", "#00D4FF"],
    )
    fig_trend.update_layout(**chart_layout(f"Signal by {dim_choice}", height=380))
    st.plotly_chart(fig_trend, use_container_width=True)

    # Root-cause heatmap.
    theme_rows = []
    for _, row in signal_df.iterrows():
        for t in row["ThemeHits"]:
            theme_rows.append({"Theme": t, "Sentiment": row["Sentiment"]})
    theme_df = pd.DataFrame(theme_rows)
    if len(theme_df) > 0:
        theme_pivot = theme_df.groupby(["Theme", "Sentiment"]).size().unstack(fill_value=0)
        for col in ["Positive", "Neutral", "Negative"]:
            if col not in theme_pivot.columns:
                theme_pivot[col] = 0
        theme_pivot = theme_pivot[["Positive", "Neutral", "Negative"]]
        fig_theme = go.Figure(
            go.Heatmap(
                z=theme_pivot.values,
                x=theme_pivot.columns.tolist(),
                y=theme_pivot.index.tolist(),
                colorscale=[[0, "#0A0E1A"], [0.5, "#FF6B35"], [1, "#FFD166"]],
                text=theme_pivot.values,
                texttemplate="%{text}",
                textfont=dict(family="JetBrains Mono"),
            )
        )
        fig_theme.update_layout(**chart_layout("Root-Cause Theme Heatmap", height=350))
        st.plotly_chart(fig_theme, use_container_width=True)
    else:
        st.info("No theme-level keywords detected in current feedback slice.")

    # Dynamic action brief.
    st.markdown(
        "<div style='font-family:Syne;font-size:0.72rem;letter-spacing:0.22em;color:#00D4FF;padding:1rem 0 0.8rem;'>ACTION BRIEF — SIGNAL COMMAND</div>",
        unsafe_allow_html=True,
    )
    if len(signal_df) == 0:
        st.info("No data matches current filters.")
    else:
        top_emotion = signal_df["Emotion"].value_counts().idxmax()
        high_conf_pct = (signal_df["ConfidenceBand"] == "High").mean() * 100
        neg_rate = (signal_df["Sentiment"] == "Negative").mean() * 100
        worst_row = risk_df.iloc[0]
        weakest_unit = worst_row[dim_choice]
        weakest_risk = worst_row["RiskScore"]
        trend_leader = trend_df.sort_values(["Positive_Rate", "Avg_Polarity"], ascending=False).iloc[0][dim_choice]

        briefs = [
            f"Primary emotional tone is **{top_emotion}**. Tune messaging and on-ground experience to reinforce this mood.",
            f"High-confidence sentiment signals are **{high_conf_pct:.1f}%** of responses. Low-confidence feedback should be probed with follow-up questions.",
            f"Overall negative sentiment is **{neg_rate:.1f}%**. Trigger corrective review if this crosses 15%.",
            f"Highest-risk {dim_choice.lower()} is **{weakest_unit}** with risk score **{weakest_risk:.2f}**. Prioritize operational intervention here.",
            f"Top signal performer is **{trend_leader}**. Replicate its practices across weaker slices.",
        ]
        for i, insight in enumerate(briefs):
            shade = "#111827" if i % 2 == 0 else "#0D1526"
            st.markdown(
                f"<div style='background:{shade};border:1px solid #1E2D45;border-left:3px solid #00D4FF;padding:0.85rem 1.2rem;border-radius:3px;margin-bottom:0.4rem;font-family:DM Sans;font-size:0.88rem;color:#E8EDF5;line-height:1.6;'>{insight}</div>",
                unsafe_allow_html=True,
            )

    stop_words = set(stopwords.words("english"))

    def get_wc(texts, cmap):
        text = " ".join(str(t) for t in texts if t)
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text) if w.lower() not in stop_words]
        if not words:
            return None
        return WordCloud(
            width=500,
            height=240,
            background_color="#111827",
            colormap=cmap,
            max_words=60,
            prefer_horizontal=0.85,
        ).generate(" ".join(words))

    wc_cols = st.columns(3)
    for (sent, cmap, label, accent), wcol in zip(
        [
            ("Positive", "GnBu", "😊 POSITIVE", "#00C896"),
            ("Neutral", "YlOrBr", "😐 NEUTRAL", "#FFD166"),
            ("Negative", "OrRd", "😞 NEGATIVE", "#EF476F"),
        ],
        wc_cols,
    ):
        with wcol:
            st.markdown(
                f"<div style='font-family:Syne;font-size:0.7rem;letter-spacing:0.2em;color:{accent};text-align:center;padding-bottom:0.4rem;'>{label}</div>",
                unsafe_allow_html=True,
            )
            wc = get_wc(fdf[fdf["Sentiment"] == sent]["Feedback on Fest"], cmap)
            if wc:
                fig_wc, ax = plt.subplots(figsize=(5, 2.5), facecolor="#111827")
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)
            else:
                st.info(f"No {sent.lower()} feedback in selection")

    fig_v = px.violin(
        fdf,
        x="Event Name",
        y="Rating",
        color="Event Name",
        box=True,
        points="outliers",
        color_discrete_sequence=COLORS,
    )
    fig_v.update_layout(**chart_layout("Rating Distribution by Event", height=400))
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown(
        "<div style='font-family:Syne;font-size:0.72rem;letter-spacing:0.2em;color:#6B7A99;padding:1rem 0 0.5rem;'>◆ FEEDBACK EXPLORER</div>",
        unsafe_allow_html=True,
    )
    kw = st.text_input("", placeholder="Search keywords: logistics, mentor, food…")
    tbl = fdf[["Student Name", "College", "State", "Event Name", "Rating", "Sentiment", "Feedback on Fest"]].copy()
    if kw:
        tbl = tbl[tbl["Feedback on Fest"].str.contains(kw, case=False, na=False, regex=False)]
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=320)

with tab4:
    st.markdown(
        "<div style='font-family:Syne;font-size:0.78rem;letter-spacing:0.22em;color:#00D4FF;padding:0.2rem 0 0.9rem;'>◆ BENCHMARK INTELLIGENCE CONSOLE</div>",
        unsafe_allow_html=True,
    )

    if len(fdf) == 0:
        st.info("No records available for current filters. Relax filters to generate benchmark intelligence.")
    else:
        control1, control2 = st.columns([1.4, 1])
        with control1:
            benchmark_dim = st.selectbox("Benchmark Entity", ["College", "State", "Event Name"], index=0)
        with control2:
            max_items = int(fdf[benchmark_dim].nunique()) if benchmark_dim in fdf.columns else 5
            top_n = st.slider("Top N", min_value=3, max_value=max(3, min(25, max_items)), value=min(10, max(3, max_items)))

        with st.expander("Benchmark Weights (Composite Score)", expanded=True):
            w1, w2, w3, w4 = st.columns(4)
            wp = w1.slider("Participation", 0, 100, 35, 5)
            wr = w2.slider("Rating", 0, 100, 25, 5)
            ws = w3.slider("Sentiment", 0, 100, 20, 5)
            wv = w4.slider("Revenue", 0, 100, 20, 5)

        weights = {
            "participants": float(wp),
            "rating": float(wr),
            "sentiment": float(ws),
            "revenue": float(wv),
        }
        bench = build_benchmark_frame(fdf, benchmark_dim, weights)

        if len(bench) == 0:
            st.warning("Benchmark table is empty for this selection.")
        else:
            top_bench = bench.head(top_n).copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Entities Compared", f"{len(bench):,}")
            c2.metric("Top Benchmark", top_bench.iloc[0][benchmark_dim])
            c3.metric("Best Score", f"{top_bench['Benchmark_Score'].max():.2f}")

            rank_plot = top_bench.sort_values("Benchmark_Score", ascending=True)
            fig_rank = px.bar(
                rank_plot,
                x="Benchmark_Score",
                y=benchmark_dim,
                orientation="h",
                color="Benchmark_Score",
                color_continuous_scale=["#003D5C", "#00D4FF"],
                text="Benchmark_Score",
                hover_data=["Participants", "Avg_Rating", "Positive_Rate", "Revenue"],
            )
            fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_rank.update_layout(**chart_layout(f"Benchmark Ranking — Top {top_n}", height=420))
            st.plotly_chart(fig_rank, use_container_width=True)

            v1, v2 = st.columns([1.2, 1.4])
            with v1:
                radar_base = top_bench.head(min(4, len(top_bench)))
                fig_radar = go.Figure()
                radar_metrics = ["N_Participants", "N_Rating", "N_Positive", "N_Revenue"]
                radar_labels = ["Participation", "Rating", "Sentiment", "Revenue"]
                for _, row in radar_base.iterrows():
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=[row[m] for m in radar_metrics],
                            theta=radar_labels,
                            fill="toself",
                            name=str(row[benchmark_dim]),
                        )
                    )
                fig_radar.update_layout(
                    **chart_layout("Top Profiles — Radar", height=420),
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E2D45", tickfont=dict(color="#6B7A99")),
                        angularaxis=dict(gridcolor="#1E2D45", tickfont=dict(color="#6B7A99")),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with v2:
                fig_scatter = px.scatter(
                    bench,
                    x="Avg_Rating",
                    y="Positive_Rate",
                    size="Participants",
                    color="Benchmark_Score",
                    hover_name=benchmark_dim,
                    hover_data=["Revenue", "Revenue_Share", "Rank"],
                    color_continuous_scale=["#111827", "#00D4FF"],
                )
                fig_scatter.update_layout(**chart_layout("Quality vs Sentiment Map", height=420))
                st.plotly_chart(fig_scatter, use_container_width=True)

            table_cols = [
                "Rank",
                benchmark_dim,
                "Benchmark_Score",
                "Participants",
                "Avg_Rating",
                "Positive_Rate",
                "Revenue",
                "Revenue_Share",
            ]
            score_table = top_bench[table_cols].copy()
            score_table["Avg_Rating"] = score_table["Avg_Rating"].round(2)
            score_table["Positive_Rate"] = score_table["Positive_Rate"].round(2)

            st.markdown(
                "<div style='font-family:Syne;font-size:0.72rem;letter-spacing:0.2em;color:#6B7A99;padding:1rem 0 0.5rem;'>◆ BENCHMARK SCORECARD</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(score_table.reset_index(drop=True), use_container_width=True, height=320)

            leader = top_bench.iloc[0]
            laggard = top_bench.iloc[-1]
            gap = leader["Benchmark_Score"] - laggard["Benchmark_Score"]
            msg = [
                f"🏆 **{leader[benchmark_dim]}** leads with score **{leader['Benchmark_Score']:.2f}** and **{leader['Participants']}** participants.",
                f"📉 **{laggard[benchmark_dim]}** trails at **{laggard['Benchmark_Score']:.2f}**. Score gap to leader: **{gap:.2f}** points.",
                f"⭐ Highest average rating in top-{top_n}: **{top_bench.loc[top_bench['Avg_Rating'].idxmax(), benchmark_dim]}** ({top_bench['Avg_Rating'].max():.2f}★).",
                f"💬 Strongest positive sentiment: **{top_bench.loc[top_bench['Positive_Rate'].idxmax(), benchmark_dim]}** ({top_bench['Positive_Rate'].max():.1f}%).",
                f"💰 Largest revenue share: **{top_bench.loc[top_bench['Revenue'].idxmax(), benchmark_dim]}** at **{top_bench['Revenue_Share'].max():.1f}%** of benchmark revenue.",
                "🎯 Action: replicate top performer event operations and mentorship playbooks in the bottom quartile entities first.",
            ]
            st.markdown(
                "<div style='font-family:Syne;font-size:0.72rem;letter-spacing:0.22em;color:#00D4FF;padding:1.2rem 0 0.8rem;'>⚡ BENCHMARK BRIEF</div>",
                unsafe_allow_html=True,
            )
            for i, insight in enumerate(msg):
                shade = "#111827" if i % 2 == 0 else "#0D1526"
                st.markdown(
                    f"<div style='background:{shade};border:1px solid #1E2D45;border-left:3px solid #00D4FF;padding:0.85rem 1.2rem;border-radius:3px;margin-bottom:0.4rem;font-family:DM Sans;font-size:0.88rem;color:#E8EDF5;line-height:1.6;'>{insight}</div>",
                    unsafe_allow_html=True,
                )

with tab3:
    total = len(fdf)
    gave_fb = (fdf["Feedback on Fest"].str.len() > 10).sum()
    rated_5 = (fdf["Rating"] == 5).sum()
    fig_fn = go.Figure(
        go.Funnel(
            y=["Registered", "Gave Meaningful Feedback", "Rated 5 Stars"],
            x=[total, gave_fb, rated_5],
            marker=dict(color=["#00D4FF", "#00C896", "#FFD166"]),
            textfont=dict(family="JetBrains Mono", color="white", size=12),
            connector=dict(line=dict(color="#1E2D45", width=2)),
        )
    )
    fig_fn.update_layout(**chart_layout("Engagement Funnel", height=320))
    st.plotly_chart(fig_fn, use_container_width=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        rev_ev = fdf.groupby("Event Name")["Amount Paid"].sum().reset_index()
        fig_rev = px.bar(
            rev_ev,
            x="Event Name",
            y="Amount Paid",
            color="Amount Paid",
            color_continuous_scale=["#003D5C", "#FF6B35"],
        )
        fig_rev.update_layout(**chart_layout("Revenue by Event", height=350))
        st.plotly_chart(fig_rev, use_container_width=True)
    with rc2:
        rev_type = fdf.groupby("Event Type")["Amount Paid"].sum().reset_index()
        fig_rt = px.pie(
            rev_type, names="Event Type", values="Amount Paid", hole=0.55, color_discrete_sequence=["#00D4FF", "#FF6B35"]
        )
        fig_rt.update_layout(**chart_layout("Individual vs Group Revenue", height=350))
        st.plotly_chart(fig_rt, use_container_width=True)

    def oracle(data):
        if len(data) == 0:
            return ["No data matches current filters."]

        out = []
        top_ev = data["Event Name"].value_counts()
        out.append(
            f"🏆 **{top_ev.index[0]}** dominates at **{top_ev.iloc[0] / len(data) * 100:.0f}%** of registrations. Consider a second edition next year."
        )

        top_st = data["State"].value_counts()
        out.append(
            f"🗺️ **{top_st.index[0]}** leads with **{top_st.iloc[0]}** participants ({top_st.iloc[0] / len(data) * 100:.0f}%). Expand outreach to **{top_st.index[-1]}** where participation is weakest."
        )

        neg_pct = (data["Sentiment"] == "Negative").sum() / len(data) * 100
        if neg_pct > 15:
            worst = data[data["Sentiment"] == "Negative"]["Event Name"].value_counts().index[0]
            out.append(
                f"⚠️ Negative sentiment **{neg_pct:.0f}%** — above 15% threshold. **{worst}** is primary contributor. Post-mortem recommended."
            )
        else:
            out.append(
                f"✅ Negative sentiment at **{neg_pct:.0f}%** — within healthy range. Participant experience is broadly positive."
            )

        ev_r = data.groupby("Event Name")["Rating"].mean()
        low, high = ev_r.idxmin(), ev_r.idxmax()
        out.append(
            f"⭐ Rating gap: **{high}** ({ev_r[high]:.2f}★) vs **{low}** ({ev_r[low]:.2f}★). Investigate logistics and mentorship for **{low}**."
        )

        total_rev = data["Amount Paid"].sum()
        rev_ev = data.groupby("Event Name")["Amount Paid"].sum()
        share = (rev_ev.max() / total_rev * 100) if total_rev > 0 else 0
        out.append(
            f"💰 Total revenue: **₹{total_rev:,.0f}**. **{rev_ev.idxmax()}** generates **{share:.0f}%** — protect this event."
        )

        grp_pct = (data["Event Type"] == "Group").sum() / len(data) * 100
        out.append(
            f"👥 **{grp_pct:.0f}%** group registrations. {'Lean into collaborative formats.' if grp_pct > 50 else 'Consider incentivizing team participation.'}"
        )

        uniq_c = data["College"].nunique()
        top_col = data["College"].value_counts().index[0]
        out.append(
            f"🏛️ **{uniq_c}** colleges represented. **{top_col}** sent the most delegates. Diversity index: {uniq_c / len(data) * 100:.1f}"
        )
        return out

    st.markdown(
        "<div style='font-family:Syne;font-size:0.72rem;letter-spacing:0.22em;color:#00D4FF;padding:1.5rem 0 0.8rem;'>⚡ ORACLE — INTELLIGENCE REPORT</div>",
        unsafe_allow_html=True,
    )
    for i, ins in enumerate(oracle(fdf)):
        shade = "#111827" if i % 2 == 0 else "#0D1526"
        st.markdown(
            f"<div style='background:{shade};border:1px solid #1E2D45;border-left:3px solid #00D4FF;padding:0.85rem 1.2rem;border-radius:3px;margin-bottom:0.4rem;font-family:DM Sans;font-size:0.88rem;color:#E8EDF5;line-height:1.6;'>{ins}</div>",
            unsafe_allow_html=True,
        )

    if len(fdf) > 0:
        college_sent = (
            fdf.groupby("College")
            .apply(lambda x: (x["Sentiment"] == "Positive").mean(), include_groups=False)
            .sort_values(ascending=False)
        )
        happiest = college_sent.index[0]
        hpy_pct = college_sent.iloc[0] * 100
        hpy_count = len(fdf[fdf["College"] == happiest])
        st.markdown(
            f"""
        <div style="background:linear-gradient(135deg,#060E08,#0D1820);border:1px solid #00C896;border-radius:4px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
          <div style="font-family:Syne;font-size:0.65rem;letter-spacing:0.25em;color:#00C896;margin-bottom:0.3rem;">🏅 HAPPIEST DELEGATION</div>
          <div style="font-family:Syne;font-size:1.3rem;font-weight:700;color:#E8EDF5;">{happiest}</div>
          <div style="font-family:JetBrains Mono;font-size:0.78rem;color:#6B7A99;margin-top:0.3rem;">
            {hpy_pct:.0f}% positive sentiment &nbsp;|&nbsp; {hpy_count} participants
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    goal = 150000
    actual = fdf["Amount Paid"].sum()
    pct = min(actual / goal * 100, 100) if goal > 0 else 0
    bc = "#00C896" if pct >= 80 else ("#FFD166" if pct >= 50 else "#EF476F")
    status = "🟢 ON TARGET" if pct >= 80 else ("🟡 PROGRESSING" if pct >= 50 else "🔴 BELOW TARGET")
    st.markdown(
        f"""
    <div style="background:#111827;border:1px solid #1E2D45;border-radius:4px;padding:1.3rem 1.5rem;">
      <div style="font-family:Syne;font-size:0.65rem;letter-spacing:0.25em;color:#6B7A99;">REVENUE GOAL TRACKER &nbsp;{status}</div>
      <div style="font-family:JetBrains Mono;font-size:2rem;color:{bc};font-weight:600;margin-top:0.4rem;">
        ₹{actual:,.0f}<span style='font-size:0.9rem;color:#6B7A99;'> / ₹{goal:,}</span>
      </div>
      <div style='background:#0A0E1A;height:8px;border-radius:4px;margin-top:0.8rem;'>
        <div style='background:{bc};height:8px;border-radius:4px;width:{pct:.1f}%;'></div>
      </div>
      <div style='font-family:JetBrains Mono;font-size:0.7rem;color:#6B7A99;margin-top:0.4rem;'>{pct:.1f}% of ₹{goal:,} goal achieved</div>
    </div>""",
        unsafe_allow_html=True,
    )
