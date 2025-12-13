import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

# -----------------------
# Page config + base CSS
# -----------------------
st.set_page_config(page_title="Traffic & Weather Dashboard", layout="wide", initial_sidebar_state="expanded")

base_css = """
<style>
:root{
  --bg:#ffffff;
  --muted:#6b7280;
  --card:#f8fafc;
  --accent:#ef4444;
}
body { background: var(--bg); }
.header-title { font-size: 36px; font-weight:800; margin:0 0 0 -10px; }
.header-sub { color: var(--muted); margin-top:6px; margin-bottom:18px; margin-left:5px; font-size:14px; }

/* ======= KPI card style (clean + fixed height) ======= */
.kpi {
  background: #ffffff;
  border-radius: 12px;
  padding: 14px;
  margin-bottom:5px;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
  border: 1px solid rgba(15,23,42,0.03);
  height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-sizing: border-box;
}

/* label, value, caption */
.metric-label { font-size:13px; color:var(--muted); margin:0; }
.metric-value { font-size:24px; font-weight:700; color:#111827; margin:0; }
.metric-caption { font-size:12px; color:var(--muted); margin:0; }

/* small note style */
.small-note { font-size:12px; color:var(--muted); margin-top:6px; }

/* responsive tweaks */
@media (max-width:900px){
  .header-title { font-size:26px; }
  .metric-value { font-size:20px; }
  .kpi { height: 115px; padding:12px; }
}
@media (max-width:480px){
  .metric-label { font-size:12px; }
  .metric-value { font-size:18px; }
  .metric-caption { font-size:11px; }
  .kpi { height: 105px; }
}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)

# -------------------------
# Palette and constants
# -------------------------
PALETTE = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
ICON_TEMP = "🌡️"
ICON_HUM = "💧"
ICON_VEH = "🚗"
ICON_ACC = "⚠️"

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data(path="merged_dataset.csv"):
    df = pd.read_csv(path)
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    else:
        raise KeyError("Missing required column: date_time")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed loading dataset: {e}")
    st.stop()

# ---------------------
# Header + description
# ---------------------
st.markdown(
    '<div>'
    '<div class="header-title">🚦Traffic & Weather</div>'
    '<div class="header-sub">Interactive dashboard for traffic and weather with fast filters and quick insights.</div>'
    '</div>',
    unsafe_allow_html=True
)

# metrics (safe fallbacks if columns missing)
avg_temp = df["temperature_c"].mean() if "temperature_c" in df.columns else 0
avg_humidity = df["humidity"].mean() if "humidity" in df.columns else 0
total_vehicles = int(df["vehicle_count"].sum()) if "vehicle_count" in df.columns else 0
total_accidents = int(df["accident_count"].sum()) if "accident_count" in df.columns else 0

# -------
# KPI Row
# -------
k1, k2, k3, k4 = st.columns([1.3, 1.3, 1.3, 1.3])

with k1:
    st.markdown(f"""
        <div class="kpi">
            <div class="metric-label">{ICON_TEMP} Average Temperature (°C)</div>
            <div class="metric-value">{avg_temp:.1f}</div>
            <div class="metric-caption">Average temperature for the selected period.</div>
        </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
        <div class="kpi">
            <div class="metric-label">{ICON_HUM} Avg Humidity (%)</div>
            <div class="metric-value">{avg_humidity:.0f}</div>
            <div class="metric-caption">Average humidity for the selected period.</div>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
        <div class="kpi">
            <div class="metric-label">{ICON_VEH} Total Vehicles</div>
            <div class="metric-value">{total_vehicles:,}</div>
            <div class="metric-caption">Total vehicle count in the filtered dataset.</div>
        </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
        <div class="kpi">
            <div class="metric-label">{ICON_ACC} Total Accidents</div>
            <div class="metric-value">{total_accidents:,}</div>
            <div class="metric-caption">Total accidents in the selected filters.</div>
        </div>
    """, unsafe_allow_html=True)

st.write("\n")

# ----------------------------
# Sidebar — Controls & Filters
# ----------------------------
st.sidebar.title("Filters")

# --- Seasons as individual checkboxes (easier to use) ---
seasons_list = sorted(df["season"].dropna().unique()) if "season" in df.columns else []
st.sidebar.markdown("### Seasons")
season_checks = {}
for s in seasons_list:
    # default to True to keep previous behavior (all selected)
    season_checks[s] = st.sidebar.checkbox(s, value=True)
# collect selected seasons
season = [s for s, checked in season_checks.items() if checked]

# --- Area dropdown with fixed choices ---
st.sidebar.markdown("### Area filter")
AREA_OPTIONS = [
    "Westminster", "Kensington", "Islington", "Camden",
    "Greenwich", "Chelsea", "Southwark"
]
area = st.sidebar.selectbox("Area (select a single area or choose All)", options=["All"] + AREA_OPTIONS, index=0)

# Date range & aggregation
min_dt, max_dt = df["date_time"].min(), df["date_time"].max()
dt_range = st.sidebar.date_input("Date range", value=[min_dt.date(), max_dt.date()])
agg = st.sidebar.selectbox("Aggregation (for trends)", ["Daily", "Weekly", "Monthly"])

# Accident line chart frequency (single remaining toggle)
st.sidebar.markdown("---")
acc_line_freq = st.sidebar.selectbox("Accident chart frequency", ["Daily", "Weekly", "Monthly"])

# Advanced
with st.sidebar.expander("Advanced"):
    if "accident_count" in df.columns:
        min_acc, max_acc = int(df["accident_count"].min()), int(df["accident_count"].max())
    else:
        min_acc, max_acc = 0, 1
    acc_threshold = st.slider("Min accidents per record", min_acc, max_acc, min_acc)
    show_grid = st.checkbox("Show grid lines on trends", value=False)
    theme_choice = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark"])
    compact_mode = st.checkbox("Compact mode (less spacing)", value=False)
    # boxplot option remains in advanced: show or hide outliers
    show_outliers = st.checkbox("Box plot show outliers", value=True)

# -------------------------
# Apply filters
# -------------------------
df_f = df.copy()

# seasons filter (from checkboxes)
if season and "season" in df_f.columns:
    df_f = df_f[df_f["season"].isin(season)]

# area dropdown filter
if area and area != "All" and "area" in df_f.columns:
    df_f = df_f[df_f["area"] == area]

# advanced filters
if "accident_count" in df_f.columns:
    df_f = df_f[df_f["accident_count"] >= acc_threshold]

# date range
start = pd.to_datetime(dt_range[0])
end = pd.to_datetime(dt_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df_f = df_f[(df_f["date_time"] >= start) & (df_f["date_time"] <= end)]

# empty guard
if df_f.empty:
    st.warning("No records match the filters — try widening the date range or removing filters.")
    st.stop()

# -------------------------
# Prepare series + aggregations
# -------------------------
df_time_all = df_f.set_index("date_time").sort_index()
vehicle_daily = df_time_all["vehicle_count"].resample("D").sum().fillna(0) if "vehicle_count" in df_time_all.columns else pd.Series(dtype="float64")

if agg == "Daily":
    vehicle_agg = vehicle_daily
    delta_period = 7
elif agg == "Weekly":
    vehicle_agg = vehicle_daily.resample("W").sum() if not vehicle_daily.empty else vehicle_daily
    delta_period = 1
else:
    vehicle_agg = vehicle_daily.resample("M").sum() if not vehicle_daily.empty else vehicle_daily
    delta_period = 1

def pct_change(s, shift):
    if len(s) <= shift:
        return 0.0
    curr = s.iloc[-1]
    prev = s.iloc[-1-shift]
    if prev == 0:
        return 0.0
    return (curr - prev) / prev * 100

# -------------------------
# Tabs with improved charts
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Detailed", "Data & Export", "Monte Carlo", "Factor Analysis"]
)


with tab1:
    st.subheader("Overview")
    # layout: left column for main scatter + extra charts stacked; right column for weather pie etc.
    left, right = st.columns([2, 1])

    # Scatter with min/max size and better legend placement
    with left:
        st.markdown("**Temperature × Avg Speed × Vehicle Count**  \n*Bubble size represents vehicle density*")
        size_ref = max(df_f["vehicle_count"].max() if "vehicle_count" in df_f.columns else 1, 1)
        fig = px.scatter(
            df_f,
            x="temperature_c" if "temperature_c" in df_f.columns else None,
            y="avg_speed_kmh" if "avg_speed_kmh" in df_f.columns else None,
            size="vehicle_count" if "vehicle_count" in df_f.columns else None,
            size_max=50,
            color="congestion_level" if "congestion_level" in df_f.columns else None,
            color_discrete_map=PALETTE,
            hover_name="area" if "area" in df_f.columns else None,
            hover_data={"vehicle_count": True, "date_time": True} if "vehicle_count" in df_f.columns else {"date_time": True},
            labels={"temperature_c": "Temp (°C)", "avg_speed_kmh": "Avg Speed (km/h)"},
            title="Temperature vs Speed (bubble = vehicles)"
        )
        fig.update_layout(
            legend=dict(title="Congestion", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template=theme_choice,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box Plot  Avg Speed vs Congestion Level 
        st.markdown("### Box Plot Avg Speed vs Congestion Level")
        if "avg_speed_kmh" in df_f.columns and "congestion_level" in df_f.columns:
            try:
                fig_box = px.box(
                    df_f,
                    x="congestion_level",
                    y="avg_speed_kmh",
                    points="outliers" if show_outliers else False,
                    category_orders={"congestion_level": ["Low", "Medium", "High"]},
                    title="Distribution of Avg Speed by Congestion Level",
                    labels={"avg_speed_kmh": "Avg Speed (km/h)", "congestion_level": "Congestion Level"},
                )
                fig_box.update_layout(template=theme_choice, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_box, use_container_width=True)
                st.markdown("<div class='small-note'>Box plot helps compare speed variability across congestion levels.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Couldn't build box plot: {e}")
        else:
            st.info("Box plot needs columns: `avg_speed_kmh` and `congestion_level`.")

        # --- Line Chart — Accident Count vs Time (Always On) ---
        st.markdown("### Line Chart Accident Count vs Time (by Weather Condition)")
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        freq = freq_map.get(acc_line_freq, "D")

        if "accident_count" in df_f.columns and "weather_condition" in df_f.columns:
            try:
                acc_df = (
                    df_f
                    .groupby([pd.Grouper(key="date_time", freq=freq), "weather_condition"])["accident_count"]
                    .sum()
                    .reset_index()
                )

                # reduce clutter: keep top weather categories
                weather_counts = df_f["weather_condition"].value_counts().index.tolist()
                top_weather = weather_counts[:8]  # show up to 8 lines
                acc_df = acc_df[acc_df["weather_condition"].isin(top_weather)]

                fig_acc = px.line(
                    acc_df,
                    x="date_time",
                    y="accident_count",
                    color="weather_condition",
                    title=f"{acc_line_freq} Accident Count by Weather Condition",
                    labels={"date_time": "Date", "accident_count": "Accidents", "weather_condition": "Weather"},
                )
                fig_acc.update_layout(
                    template=theme_choice,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=40, b=20, l=10, r=10)
                )
                fig_acc.update_xaxes(showgrid=show_grid)
                fig_acc.update_yaxes(showgrid=show_grid)
                st.plotly_chart(fig_acc, use_container_width=True)
                st.markdown("<div class='small-note'>Aggregation frequency can be changed from the sidebar. Top weather types are shown to reduce clutter.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Couldn't build accidents line chart: {e}")
        else:
            st.info("Accident line chart needs columns: `accident_count`, `date_time`, and `weather_condition`.")

    with right:
        st.subheader("Weather Conditions")
        if "weather_condition" in df_f.columns:
            cond = df_f["weather_condition"].value_counts().reset_index()
            cond.columns = ["condition", "count"]
            fig2 = px.pie(cond, values="count", names="condition", hole=0.55, title="Weather distribution")
            fig2.update_layout(template=theme_choice, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='small-note'>Click a slice to inspect details in hover.</div>", unsafe_allow_html=True)
        else:
            st.info("No `weather_condition` column available to show distribution.")

        # ------------------------
        # NEW: Road condition pie
        # ------------------------
        st.subheader("Road Conditions")
        # Ensure we show the four requested categories in the specified order even if some are absent
        desired_order = ["Snowy", "Dry", "Wet", "Damaged"]
        if "road_condition" in df_f.columns:
            # fill NaNs with 'Unknown' if you want them to appear, otherwise dropna()
            road_series = df_f["road_condition"].astype(str).fillna("Unknown")
            # build counts and ensure the desired order + include any other categories at the end
            counts = road_series.value_counts()
            # build dataframe that ensures the 4 categories are present (0 if missing)
            road_df = pd.DataFrame({
                "condition": desired_order,
                "count": [int(counts.get(k, 0)) for k in desired_order]
            })
            # include other categories (if any) after the main four
            others = [k for k in counts.index if k not in desired_order]
            if others:
                others_df = pd.DataFrame({"condition": others, "count": [int(counts[k]) for k in others]})
                road_df = pd.concat([road_df, others_df], ignore_index=True)

            # use a blue sequential palette for road conditions
            try:
                color_seq = px.colors.qualitative.Plotly
            except Exception:
                color_seq = None

            fig_road = px.pie(
                road_df,
                values="count",
                names="condition",
                hole=0.55,
                title="Road condition distribution",
                color_discrete_sequence=color_seq
            )
            fig_road.update_layout(template=theme_choice, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_road, use_container_width=True)
            st.markdown("<div class='small-note'>Road condition distribution for the filtered data. Categories: Snowy, Dry, Wet, Damaged.</div>", unsafe_allow_html=True)
        else:
            st.info("No `road_condition` column available to show distribution. Make sure your CSV has a `road_condition` column with values like Snowy, Dry, Wet, Damaged.")

with tab2:
    st.subheader("Detailed analysis")

    # Heatmap accidents
    df_f = df_f.copy()
    if "date_time" in df_f.columns:
        df_f["hour"] = df_f["date_time"].dt.hour
        df_f["weekday"] = df_f["date_time"].dt.day_name()
    else:
        df_f["hour"] = 0
        df_f["weekday"] = "Monday"

    if "accident_count" in df_f.columns:
        heat = df_f.groupby(["weekday", "hour"])["accident_count"].sum().reset_index()
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat["weekday"] = pd.Categorical(heat["weekday"], categories=weekdays, ordered=True)
        heat = heat.sort_values(["weekday", "hour"])
        fig3 = px.density_heatmap(
            heat, x="hour", y="weekday", z="accident_count",
            category_orders={"weekday": weekdays},
            title="Accidents heatmap (hour vs day)"
        )
        fig3.update_layout(template=theme_choice)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Heatmap requires `accident_count` column.")

    st.markdown("---")

    # Trend
    st.subheader("Vehicle trend")
    if "vehicle_count" in df_time_all.columns:
        if agg == "Daily":
            plot_series = df_time_all["vehicle_count"].resample("D").sum().fillna(0)
            roll = 7
        elif agg == "Weekly":
            plot_series = df_time_all["vehicle_count"].resample("W").sum().fillna(0)
            roll = 4
        else:
            plot_series = df_time_all["vehicle_count"].resample("M").sum().fillna(0)
            roll = 3
        rolling = plot_series.rolling(roll).mean()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=plot_series.index, y=plot_series.values, mode="lines", name="Total"))
        fig4.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name=f"{roll}-period rolling mean"))
        fig4.update_layout(
            title="Vehicle counts over time",
            xaxis_title="Date",
            yaxis_title="Vehicles",
            template=theme_choice,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid),
            margin=dict(t=40, b=20, l=10, r=10)
        )
        st.plotly_chart(fig4, use_container_width=True)

    #     # download PNG
    #     try:
    #         img = fig4.to_image(format="png", scale=2)
    #         st.download_button("Download trend image (PNG)", data=img, file_name="vehicle_trend.png", mime="image/png")
    #     except Exception:
    #         st.info("To download PNG: install `kaleido` (pip install kaleido).")
    # else:
    #     st.info("Vehicle trend chart requires `vehicle_count` column.")

    st.markdown("---")

    # Congestion by area stacked
    st.subheader("Congestion by area")
    if "area" in df_f.columns and "vehicle_count" in df_f.columns:
        if "congestion_level" in df_f.columns:
            cong = df_f.groupby(["area", "congestion_level"])["vehicle_count"].sum().reset_index()
            fig5 = px.bar(cong, x="area", y="vehicle_count", color="congestion_level", color_discrete_map=PALETTE,
                          title="Vehicles by congestion level per area")
        else:
            cong = df_f.groupby("area")["vehicle_count"].sum().reset_index().sort_values("vehicle_count", ascending=False)
            fig5 = px.bar(cong, x="area", y="vehicle_count", title="Vehicles per area")
        fig5.update_layout(template=theme_choice, margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Congestion chart requires `area` and `vehicle_count` columns.")

with tab3:
    st.subheader("Data & Export")
    st.write("Preview of filtered data (first 200 rows):")
    st.dataframe(df_f.head(200))

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="filtered_traffic_weather.csv", mime="text/csv")

    st.markdown("**Quick insights:**")
    try:
        top_area = df_f.groupby("area")["vehicle_count"].sum().idxmax()
    except Exception:
        top_area = "N/A"
    try:
        worst_cond = df_f.groupby("weather_condition")["accident_count"].sum().idxmax()
    except Exception:
        worst_cond = "N/A"
    st.markdown(f"- Area with highest vehicles: **{top_area}**")
    st.markdown(f"- Weather with most accidents: **{worst_cond}**")

with tab4:
    # Title
    st.title("Monte Carlo Simulation")

    # Image 
    st.image(
        "assets/congestion_probability_distribution_1-converted.webp",  
        use_container_width=True
    )
    # Description text
    st.markdown(
        """
        This is the distribution of congestion level probability extracted 
        from 10,000 samples under the different weather scenarios we have.
        """
    )
    st.divider()
  
    try:
        mc_df = pd.read_csv("assets/simulation_results.csv")

        st.markdown("### Simulation Results (CSV)")
        st.dataframe(mc_df, use_container_width=True)
        st.caption(
        "Each row contains the actual probability of high congestion and accident risk calculated from the original dataset and the predicted probability extracted from the Monte Carlo simulation along with standard deviation and 95% confidence interval"
        )

        # Download button
        csv_bytes = mc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Monte Carlo Results (CSV)",
            data=csv_bytes,
            file_name="monte_carlo_results.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.warning("Monte Carlo results CSV not found in the assets folder.")

with tab5:
    # Title
    st.title("Factor Analysis")

    st.image(
            "assets/scree_plot.webp", use_container_width=True
        )
    st.caption(
            "Pick best factors based on eginvalues"
        )

    st.image(
            "assets/correlation_matrix.webp" 
            , use_container_width=True
        )
    st.caption(
            "Loading table for factors to interpret them"
        )

# -------------------------
# Footer tips + accessibility
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Tips & Accessibility**")
st.sidebar.markdown("- Use the area dropdown to focus on a specific neighborhood quickly.")
st.sidebar.markdown("- Toggle aggregation frequency to change accident chart granularity.")


# End of dashboard






















