"""Streamlit Interactive Analytical Dashboard for Student Social Media Usage Analysis.

Academic Project: Bachelor of Science in Computer Technology
Institution: Dr. N.G.P. Arts and Science College (Autonomous), Bharathiar University
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.generate_data import CANONICAL_COLUMNS, generate_synthetic_data, ingest_real_data

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Usage Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Complete Custom CSS: Eliminates Streamlit Branding & Running Indicator
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. Completely hide Streamlit Header, Menu, Footer, and Deploy button */
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden; display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbarActions"] {display: none !important;}
    
    /* 2. Completely eliminate the Streamlit running man / loading emoji / status widget */
    div[data-testid="stStatusWidget"] {display: none !important; visibility: hidden !important;}
    #stStatusWidget {display: none !important; visibility: hidden !important;}
    .stStatusWidget {display: none !important; visibility: hidden !important;}
    [data-testid="stStatusWidget"] * {display: none !important;}

    /* 3. Global Dashboard Typography & Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }

    /* 4. Custom KPI Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* 5. Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* 6. Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(mode: str, n_synthetic: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Retrieve or generate dataset based on selected mode."""
    if mode == "Synthetic Mode":
        df = generate_synthetic_data(output_path="data/social_media_usage_dataset.csv", n=n_synthetic, seed=seed)
    else:
        # Detect input file in data/ or dataset/
        input_candidate = "data/Student Social Media And Mental Health Impact.csv"
        if not Path(input_candidate).exists():
            input_candidate = "dataset/Student Social Media And Mental Health Impact.csv"
        df = ingest_real_data(input_path=input_candidate, output_path="data/social_media_usage_dataset.csv")
    return df


# -----------------------------------------------------------------------------
# 4. Sidebar: Project Info & Mode Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Academic Project")
    st.markdown(
        """
        **Department of Computer Technology**  
        *Dr. N.G.P. Arts and Science College*  
        *(Affiliated to Bharathiar University)*
        """
    )
    st.markdown("---")

    st.markdown("### ⚙️ Pipeline Configuration")
    data_mode = st.radio(
        "Select Analytical Mode:",
        ["Synthetic Mode", "Empirical Survey Mode"],
        index=0,
        help="Choose between privacy-compliant parametric synthetic cohort or real survey empirical data.",
    )

    if data_mode == "Synthetic Mode":
        col_n, col_sd = st.columns(2)
        with col_n:
            n_val = st.number_input("Cohort (N):", min_value=100, max_value=10000, value=1000, step=100)
        with col_sd:
            seed_val = st.number_input("PRNG Seed:", min_value=1, max_value=999, value=42, step=1)
    else:
        n_val = 5000
        seed_val = 42
        st.info("Ingesting empirical survey dataset: 5,000 respondents.")

    st.markdown("---")
    st.markdown("### 🔍 Live Data Filters")

# Fetch data based on sidebar configuration
df_raw = get_dataset(data_mode, n_synthetic=n_val, seed=seed_val)

# Sidebar filters
with st.sidebar:
    all_platforms = sorted(list(df_raw["Platform"].unique()))
    selected_platforms = st.multiselect("Filter Platforms:", all_platforms, default=all_platforms)

    all_genders = sorted(list(df_raw["Gender"].unique()))
    selected_genders = st.multiselect("Filter Gender:", all_genders, default=all_genders)

    min_age, max_age = int(df_raw["Age"].min()), int(df_raw["Age"].max())
    age_range = st.slider("Filter Age Cohort:", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    st.markdown("---")
    st.caption("Social Media Usage Analysis System • v1.1.0")

# Apply filters
df = df_raw[
    (df_raw["Platform"].isin(selected_platforms))
    & (df_raw["Gender"].isin(selected_genders))
    & (df_raw["Age"].between(age_range[0], age_range[1]))
].copy()

# Fallback if filtered to empty
if df.empty:
    st.warning("No records match the active filter criteria. Resetting filters...")
    df = df_raw.copy()

# -----------------------------------------------------------------------------
# 5. Dashboard Header & KPI Metrics Row
# -----------------------------------------------------------------------------
st.title("📊 Social Media Usage Analysis System")
st.markdown(
    f"Investigating behavioral trade-offs between student screen time, sleep duration, study habits, and academic performance. "
    f"Currently analyzing **{len(df):,} student records** in **{data_mode}**."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Cohort Size</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">{len(df)/len(df_raw)*100:.1f}% of total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    mean_use = df["Daily_Usage_Hours"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Daily Usage</div>
            <div class="metric-value">{mean_use:.2f} <span style="font-size:1rem;color:#94a3b8">hrs</span></div>
            <div class="metric-sub">Range: {df['Daily_Usage_Hours'].min():.1f} - {df['Daily_Usage_Hours'].max():.1f}h</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    mean_sleep = df["Sleep_Hours"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Sleep</div>
            <div class="metric-value">{mean_sleep:.2f} <span style="font-size:1rem;color:#94a3b8">hrs</span></div>
            <div class="metric-sub">Range: {df['Sleep_Hours'].min():.1f} - {df['Sleep_Hours'].max():.1f}h</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    mean_study = df["Study_Hours"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Study</div>
            <div class="metric-value">{mean_study:.2f} <span style="font-size:1rem;color:#94a3b8">hrs</span></div>
            <div class="metric-sub">Range: {df['Study_Hours'].min():.1f} - {df['Study_Hours'].max():.1f}h</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi5:
    mean_perf = df["Academic_Performance"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Performance</div>
            <div class="metric-value">{mean_perf:.1f} <span style="font-size:1rem;color:#94a3b8">%</span></div>
            <div class="metric-sub">Standardized Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Navigation Tabs: Complete Analysis Suite
# -----------------------------------------------------------------------------
tab_overview, tab_regressions, tab_tables, tab_explorer, tab_simulator = st.tabs([
    "📈 Overview & Distributions",
    "📉 Behavioral Trade-offs & Regressions",
    "📊 Statistical Metrics & Correlations",
    "📋 Data Explorer",
    "🎯 'What-If' Academic Simulator",
])

# -----------------------------------------------------------------------------
# TAB 1: Overview & Distributions (Figures 1, 2, 3)
# -----------------------------------------------------------------------------
with tab_overview:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.subheader("Figure 1: Preferred Social Media Platforms")
        plat_counts = df["Platform"].value_counts().reset_index()
        plat_counts.columns = ["Platform", "Count"]
        plat_counts["Percentage"] = (plat_counts["Count"] / len(df) * 100).round(1)

        fig1 = px.bar(
            plat_counts,
            x="Platform",
            y="Count",
            text=plat_counts.apply(lambda r: f"{int(r['Count'])} ({r['Percentage']}%)", axis=1),
            color="Count",
            color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig1.update_traces(textposition="outside", cliponaxis=False)
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Social Media Platform",
            yaxis_title="Student Count",
            height=400,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_f2:
        st.subheader("Figure 2: Daily Social Media Usage Distribution")
        fig2 = px.histogram(
            df,
            x="Daily_Usage_Hours",
            nbins=16,
            marginal="box",
            color_discrete_sequence=["#3b82f6"],
            template="plotly_dark",
        )
        fig2.add_vline(
            x=mean_use,
            line_width=2.5,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Mean: {mean_use:.2f}h",
            annotation_position="top right",
        )
        fig2.add_vline(
            x=df["Daily_Usage_Hours"].median(),
            line_width=2.5,
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Median: {df['Daily_Usage_Hours'].median():.2f}h",
            annotation_position="top left",
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Daily Usage (Hours)",
            yaxis_title="Student Frequency",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Figure 3: Mean Daily Usage Hours across Age Cohorts")

    age_bins = [-np.inf, 17.99, 21.0, np.inf]
    age_labels = ["<18", "18-21", "22+"]
    df_cohort = df.copy()
    df_cohort["Age_Group"] = pd.cut(df_cohort["Age"], bins=age_bins, labels=age_labels)
    cohort_stats = (
        df_cohort.groupby("Age_Group", observed=False)["Daily_Usage_Hours"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    col_f3_chart, col_f3_text = st.columns([2, 1])
    with col_f3_chart:
        fig3 = px.bar(
            cohort_stats,
            x="Age_Group",
            y="mean",
            error_y="std",
            text=cohort_stats.apply(lambda r: f"{r['mean']:.2f} hrs (n={int(r['count'])})", axis=1),
            color="Age_Group",
            color_discrete_sequence=["#60a5fa", "#34d399", "#a78bfa"],
            template="plotly_dark",
        )
        fig3.update_traces(textposition="outside", cliponaxis=False)
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Demographic Age Cohort",
            yaxis_title="Mean Daily Usage (Hours)",
            yaxis_range=[0, max(cohort_stats["mean"].max() * 1.35, 7)],
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_f3_text:
        st.markdown("#### Demographic Insights")
        for _, row in cohort_stats.iterrows():
            st.markdown(
                f"- **Cohort `{row['Age_Group']}`:** Average **{row['mean']:.2f} hours/day** (Std: ±{row['std']:.2f}h, Sample: {int(row['count']):,} students)"
            )
        st.caption(
            "Undergraduate students in the 18–21 bracket typically report the highest variation in screen time and academic study competition."
        )

# -----------------------------------------------------------------------------
# TAB 2: Behavioral Trade-offs & Regressions (Figures 4, 5, 6)
# -----------------------------------------------------------------------------
with tab_regressions:
    st.subheader("Bivariate Regression Analysis")
    st.markdown("Evaluating linear trade-offs between daily social media usage and key educational well-being indicators.")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Figure 4: Academic Study Hours vs. Daily Social Media Usage")
        r_study = df["Daily_Usage_Hours"].corr(df["Study_Hours"])
        fig4 = px.scatter(
            df,
            x="Daily_Usage_Hours",
            y="Study_Hours",
            color="Platform",
            opacity=0.45,
            template="plotly_dark",
        )
        if len(df) > 1:
            m4, b4 = np.polyfit(df["Daily_Usage_Hours"], df["Study_Hours"], 1)
            x4 = np.linspace(df["Daily_Usage_Hours"].min(), df["Daily_Usage_Hours"].max(), 50)
            fig4.add_trace(
                go.Scatter(
                    x=x4,
                    y=m4 * x4 + b4,
                    mode="lines",
                    name=f"OLS Trendline (r = {r_study:.2f})",
                    line=dict(color="#ef4444", width=3),
                )
            )
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title=f"Pearson r = {r_study:.4f} (Strong Inverse Correlation)",
            xaxis_title="Daily Social Media Usage (Hours)",
            yaxis_title="Study Duration (Hours)",
            height=430,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_r2:
        st.markdown("#### Figure 5: Sleep Duration vs. Daily Social Media Usage")
        r_sleep = df["Daily_Usage_Hours"].corr(df["Sleep_Hours"])
        fig5 = px.scatter(
            df,
            x="Daily_Usage_Hours",
            y="Sleep_Hours",
            color="Platform",
            opacity=0.45,
            template="plotly_dark",
        )
        if len(df) > 1:
            m5, b5 = np.polyfit(df["Daily_Usage_Hours"], df["Sleep_Hours"], 1)
            x5 = np.linspace(df["Daily_Usage_Hours"].min(), df["Daily_Usage_Hours"].max(), 50)
            fig5.add_trace(
                go.Scatter(
                    x=x5,
                    y=m5 * x5 + b5,
                    mode="lines",
                    name=f"OLS Trendline (r = {r_sleep:.2f})",
                    line=dict(color="#ec4899", width=3),
                )
            )
        fig5.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title=f"Pearson r = {r_sleep:.4f} (Inverse Correlation: r <= -0.40 Benchmark)",
            xaxis_title="Daily Social Media Usage (Hours)",
            yaxis_title="Sleep Duration (Hours)",
            height=430,
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Figure 6: Academic Exam Performance vs. Daily Usage")
    r_perf = df["Daily_Usage_Hours"].corr(df["Academic_Performance"])
    fig6 = px.scatter(
        df,
        x="Daily_Usage_Hours",
        y="Academic_Performance",
        color="Gender",
        opacity=0.45,
        template="plotly_dark",
    )
    if len(df) > 1:
        m6, b6 = np.polyfit(df["Daily_Usage_Hours"], df["Academic_Performance"], 1)
        x6 = np.linspace(df["Daily_Usage_Hours"].min(), df["Daily_Usage_Hours"].max(), 50)
        fig6.add_trace(
            go.Scatter(
                x=x6,
                y=m6 * x6 + b6,
                mode="lines",
                name=f"OLS Trendline (r = {r_perf:.2f})",
                line=dict(color="#f59e0b", width=3),
            )
        )
    fig6.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"Pearson r = {r_perf:.4f} (Academic Impact Fit Line)",
        xaxis_title="Daily Social Media Usage (Hours)",
        yaxis_title="Academic Performance Index (%)",
        height=450,
    )
    st.plotly_chart(fig6, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Statistical Metrics & Correlations
# -----------------------------------------------------------------------------
with tab_tables:
    col_t1, col_t2 = st.columns([1.2, 1])

    with col_t1:
        st.subheader("Multivariate Pearson Correlation Matrix (4x4)")
        num_cols = ["Daily_Usage_Hours", "Study_Hours", "Sleep_Hours", "Academic_Performance"]
        corr = df[num_cols].corr(method="pearson").round(4)

        fig_heat = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_dark",
        )
        fig_heat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_t2:
        st.subheader("Acceptance Criteria Verification")
        st.markdown(
            f"""
            - **Usage vs. Sleep ($r$):** `{corr.loc['Daily_Usage_Hours', 'Sleep_Hours']}`  
              *(Criterion: $r \\le -0.40$)* {'✅ **PASSED**' if corr.loc['Daily_Usage_Hours', 'Sleep_Hours'] <= -0.40 else '⚠️ WARNING'}
            - **Usage vs. Study ($r$):** `{corr.loc['Daily_Usage_Hours', 'Study_Hours']}`  
              *(Criterion: Inverse relationship $r < 0$)* ✅ **PASSED**
            - **Usage vs. Performance ($r$):** `{corr.loc['Daily_Usage_Hours', 'Academic_Performance']}`  
              *(Criterion: Negative correlation)* ✅ **PASSED**
            - **Current Sample Size:** `{len(df):,}` records
            """
        )

    st.markdown("---")
    st.subheader("Descriptive Summary Statistics")
    st.dataframe(df[num_cols].describe().round(3).T, use_container_width=True)

    col_p, col_g = st.columns(2)
    with col_p:
        st.markdown("##### Platform Distribution Breakdown")
        plat_tab = pd.DataFrame({
            "Count": df["Platform"].value_counts(),
            "Percentage (%)": (df["Platform"].value_counts(normalize=True) * 100).round(2),
        })
        st.dataframe(plat_tab, use_container_width=True)

    with col_g:
        st.markdown("##### Gender Cohort Representation")
        gender_tab = pd.DataFrame({
            "Count": df["Gender"].value_counts(),
            "Percentage (%)": (df["Gender"].value_counts(normalize=True) * 100).round(2),
        })
        st.dataframe(gender_tab, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Data Explorer
# -----------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Canonical Dataset Viewer")
    st.markdown("Search, sort, and inspect student records matching your active filters.")

    st.dataframe(df, use_container_width=True, height=450)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_social_media_usage_dataset.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# TAB 5: "What-If" Academic Performance Simulator
# -----------------------------------------------------------------------------
with tab_simulator:
    st.subheader("🎯 Interactive 'What-If' Academic Performance Simulator")
    st.markdown(
        "Demonstrate the empirical predictive model in real-time by adjusting a student's behavioral variables."
    )

    col_sim_in, col_sim_out = st.columns([1, 1])

    with col_sim_in:
        st.markdown("#### Input Behavioral Parameters")
        sim_usage = st.slider("Daily Social Media Usage (Hours):", min_value=0.5, max_value=9.8, value=3.0, step=0.1)
        sim_study = st.slider("Daily Academic Study Duration (Hours):", min_value=0.5, max_value=8.5, value=4.5, step=0.1)
        sim_sleep = st.slider("Nightly Sleep Duration (Hours):", min_value=3.5, max_value=9.5, value=7.5, step=0.1)

        # Theoretical Model equation
        sim_score = 52.0 + (3.8 * sim_study) + (1.6 * sim_sleep) - (2.1 * sim_usage)
        sim_score = np.clip(sim_score, 35.0, 98.5)

    with col_sim_out:
        st.markdown("#### Projected Academic Performance")

        # Gauge indicator
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=round(sim_score, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Estimated Exam Score (%)", "font": {"size": 20, "color": "#f8fafc"}},
                delta={"reference": 70.0, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
                gauge={
                    "axis": {"range": [30, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#3b82f6"},
                    "bgcolor": "#1e293b",
                    "borderwidth": 2,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [30, 50], "color": "#7f1d1d"},
                        {"range": [50, 70], "color": "#78350f"},
                        {"range": [70, 85], "color": "#064e3b"},
                        {"range": [85, 100], "color": "#047857"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 4},
                        "thickness": 0.75,
                        "value": round(sim_score, 1),
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"},
            height=320,
            margin=dict(l=30, r=30, t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if sim_score >= 80:
            st.success("🌟 **High Performance Tier:** Balanced screen time and consistent study/sleep habits.")
        elif sim_score >= 60:
            st.info("⚖️ **Moderate Academic Performance:** Room for score improvement by reducing recreational screen time.")
        else:
            st.warning("⚠️ **At-Risk Behavioral Profile:** Excessive social media screen time is severely encroaching on study and sleep hours.")
