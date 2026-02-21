import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from datetime import datetime
import os, json
from PIL import Image
import streamlit.components.v1 as components

st.set_page_config(page_title="Chicago Crime Dashboard", page_icon="🚨",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Poppins',sans-serif!important;}
.main-title{font-size:2.8rem;font-weight:800;color:#1a1a2e;letter-spacing:-1px;line-height:1.15;}
.red-accent{color:#e63946;}
.main-subtitle{font-size:1rem;color:#6b7280;font-weight:300;margin-top:4px;}
.kpi-card{background:white;border-radius:16px;padding:24px 20px;border:1px solid #f0f0f0;
          box-shadow:0 2px 12px rgba(0,0,0,.06);text-align:center;margin-bottom:8px;}
.kpi-number{font-size:2.4rem;font-weight:800;line-height:1;}
.kpi-label{font-size:.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1.5px;margin-top:6px;font-weight:500;}
.kpi-red{color:#e63946;}.kpi-blue{color:#2563eb;}.kpi-green{color:#16a34a;}.kpi-purple{color:#7c3aed;}
.section-header{font-size:1.3rem;font-weight:700;color:#1a1a2e;margin-bottom:16px;
                padding-bottom:8px;border-bottom:3px solid #e63946;display:inline-block;}
.category-card{background:white;border-radius:12px;padding:14px 18px;border-left:4px solid;
               box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:10px;}
.insight-card{background:#f8fafc;border-radius:12px;padding:14px 18px;
              border-left:4px solid #e63946;margin-bottom:12px;}
.insight-title{font-weight:600;color:#1a1a2e;font-size:.88rem;}
.insight-text{color:#6b7280;font-size:.82rem;margin-top:2px;}
.metric-card{background:white;border-radius:14px;padding:20px 18px;
             box-shadow:0 2px 10px rgba(0,0,0,.07);border-top:4px solid;text-align:center;}
.result-box{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:20px;
            padding:36px;text-align:center;color:white;margin:20px 0;}
.result-category{font-size:2.2rem;font-weight:800;color:#e63946;margin:8px 0;}
.form-section-title{font-weight:700;font-size:1rem;color:#1a1a2e;margin:22px 0 10px;}
.stButton>button{background:#e63946!important;color:white!important;border:none!important;
                 border-radius:10px!important;font-weight:700!important;font-size:1rem!important;
                 padding:14px 28px!important;width:100%!important;}
.underfit{background:#fef3c7;border-radius:12px;padding:14px 18px;
          border-left:4px solid #f59e0b;margin-bottom:16px;font-size:.85rem;color:#78350f;}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS = {
    "accuracy":    0.5188,
    "f1_weighted": 0.5307,
    "classes": {
        "PROPERTY_CRIME": {"precision":0.74,"recall":0.54,"f1":0.62,"support":176227},
        "VIOLENT_CRIME":  {"precision":0.51,"recall":0.43,"f1":0.47,"support":112621},
        "OTHER":          {"precision":0.31,"recall":0.51,"f1":0.39,"support": 52007},
        "DRUG_CRIME":     {"precision":0.39,"recall":0.69,"f1":0.50,"support": 44036},
    },
    "feature_importance": [
        ("Location_Index",       37.11),("District_Crime_Count",11.22),
        ("District",             10.88),("Latitude",             9.49),
        ("Beat",                  9.11),("Hour",                 8.64),
        ("Beat_Crime_Count",      7.52),("Distance_from_center", 7.04),
        ("Longitude",             7.00),("IsWeekend",            5.81),
    ],
    "training_log": [
        (0,   1.37456,1.37549),(500, 1.04274,1.11310),(1000,1.01611,1.10121),
        (1500,0.99653,1.09489),(2000,0.98003,1.09035),(2500,0.96533,1.08709),
        (3000,0.95181,1.08432),(3500,0.93924,1.08219),(4000,0.92749,1.08023),
        (4999,0.90550,1.07720),
    ],
}

CRIME_EMOJIS  = {"PROPERTY_CRIME":"🏠","VIOLENT_CRIME":"⚠️","DRUG_CRIME":"💊","OTHER":"📋"}
CRIME_COLORS  = {"PROPERTY_CRIME":"#2563eb","VIOLENT_CRIME":"#e63946","DRUG_CRIME":"#8b5cf6","OTHER":"#16a34a"}
FEATURE_COLS  = ['District','Community Area','Year','Month','Hour','DayOfWeek',
                 'Latitude','Longitude','Beat','Ward','Hour_sin','Hour_cos','IsWeekend',
                 'Distance_from_center','District_Crime_Count','Beat_Crime_Count',
                 'Community_Crime_Count','Location_Index']
DOW_NAMES     = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
PROJECT_DIR   = r"C:\Users\Omen\PycharmProjects\lab3\Chicago_project"


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_indexer_mappings():
    for path in [
        "indexer_mappings.json",
        os.path.join(PROJECT_DIR, "indexer_mappings.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexer_mappings.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            loc_map    = {v: i for i, v in enumerate(data["location_labels"])}
            label_list = data.get("label_labels",
                                  ["PROPERTY_CRIME","VIOLENT_CRIME","OTHER","DRUG_CRIME"])
            return loc_map, label_list
    st.error("⚠️ indexer_mappings.json not found.")
    return {}, ["PROPERTY_CRIME","VIOLENT_CRIME","OTHER","DRUG_CRIME"]


@st.cache_resource
def load_model():
    for path in [
        "crime_xgboost_clean.json",
        "crime_xgboost_clean1.json",
        os.path.join(PROJECT_DIR, "crime_xgboost_clean.json"),
        os.path.join(PROJECT_DIR, "crime_xgboost_clean1.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "crime_xgboost_clean.json"),
    ]:
        if os.path.exists(path):
            try:
                m = xgb.Booster()
                m.load_model(path)
                return m
            except Exception as e:
                st.error(f"Error loading model: {e}")
    st.error("⚠️ Model file not found.")
    return None


# ── Feature engineering ───────────────────────────────────────────────────────
def prepare_features(district, community_area, year, month, hour, day_of_week,
                     latitude, longitude, beat, ward,
                     district_count, beat_count, community_count, location_index):
    spark_dow   = day_of_week + 2 if day_of_week < 6 else 1
    dist_center = np.sqrt((latitude - 41.8781)**2 + (longitude + 87.6298)**2) * 111
    features = [
        district, community_area, year, month, hour, spark_dow,
        latitude, longitude, beat, ward,
        np.sin(np.radians(hour * 15)), np.cos(np.radians(hour * 15)),
        1.0 if spark_dow in [1, 7] else 0.0,
        dist_center, district_count, beat_count, community_count, location_index,
    ]
    assert len(features) == 18, f"Expected 18 features, got {len(features)}"
    return np.array(features, dtype=np.float32).reshape(1, -1)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🚨 Chicago Crime")
        st.caption("Analysis & Prediction Dashboard")
        st.markdown("---")
        page = st.radio("Navigate", [
            "📊  Overview",
            "📈  Visualizations",
            "🔮  Crime Prediction",
            "📍  Interactive Maps",
            "📉  Trends Analysis",

        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.75rem;color:#9ca3af;line-height:2.1;">
        <b style="color:#374151;">Dataset:</b> Chicago 2001–2004<br>
        <b style="color:#374151;">Train:</b> 1,425,334 rows<br>
        <b style="color:#374151;">Test:</b> 384,891 rows<br>
        <b style="color:#374151;">Classes:</b> 4<br>
        <b style="color:#374151;">Features:</b> 18<br>
        <b style="color:#374151;">Accuracy:</b>
          <span style="color:#e63946;font-weight:800;">51.88%</span><br>
        <b style="color:#374151;">F1 Weighted:</b>
          <span style="color:#e63946;font-weight:800;">53.07%</span>
        </div>""", unsafe_allow_html=True)
    return page


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview():
    st.markdown('<div class="main-title">Chicago Crime <span class="red-accent">Analysis</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">XGBoost · Apache Spark · 4 Crime Categories · 2001–2004</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    kpi_data = [
        ("1.92M",  "TOTAL RECORDS",    "kpi-red"),
        ("25",     "POLICE DISTRICTS", "kpi-blue"),
        ("4",      "CRIME CLASSES",    "kpi-purple"),
        ("52%",  "ACCURACY",         "kpi-green")
    ]
    for col, (num, lbl, cls) in zip(st.columns(5), kpi_data):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-number {cls}">{num}</div>'
                f'<div class="kpi-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Crime Categories</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for color, emoji, title, desc, samples in [
            ("#2563eb","🏠","PROPERTY CRIME",
             "Theft · Burglary · Vehicle Theft · Criminal Damage · Trespass",
             "659K train | 176K test"),
            ("#e63946","⚠️","VIOLENT CRIME",
             "Battery · Assault · Robbery · Homicide · Weapons Violation",
             "434K train | 113K test"),
            ("#8b5cf6","💊","DRUG CRIME",
             "Narcotics · Other Narcotic Violations",
             "158K train | 44K test"),
            ("#16a34a","📋","OTHER",
             "All remaining crime types",
             "175K train | 52K test"),
        ]:
            st.markdown(f"""<div class="category-card" style="border-color:{color};">
                <span style="font-size:1.1rem;">{emoji}</span>
                <span style="font-weight:700;color:{color};margin-left:8px;">{title}</span>
                <span style="float:right;font-size:.72rem;color:#9ca3af;">{samples}</span><br>
                <span style="font-size:.8rem;color:#6b7280;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        cats   = list(RESULTS["classes"])
        labels = [c.replace("_CRIME","").replace("_"," ") for c in cats]
        fig = go.Figure()
        for vals, name, color in [
            ([RESULTS["classes"][c]["precision"] for c in cats], "Precision", "#e63946"),
            ([RESULTS["classes"][c]["recall"]    for c in cats], "Recall",    "#2563eb"),
            ([RESULTS["classes"][c]["f1"]        for c in cats], "F1-Score",  "#16a34a"),
        ]:
            fig.add_trace(go.Bar(name=name, x=labels, y=vals, marker_color=color,
                                 marker_line_width=0,
                                 text=[f"{v:.2f}" for v in vals],
                                 textposition="outside"))
        fig.update_layout(barmode="group", height=310, template="plotly_white",
                          margin=dict(t=10,b=0,l=0,r=0),
                          legend=dict(orientation="h", y=1.12),
                          yaxis=dict(range=[0,1.15], tickformat=".0%"),
                          font=dict(family="Poppins"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div style="background:#1a1a2e;border-radius:10px;padding:14px 20px;
                        display:flex;justify-content:space-between;align-items:center;color:white;">
            <span style="font-weight:600;">Overall</span>
            <span style="color:#e63946;font-weight:800;">Acc 51.88% &nbsp;·&nbsp; F1 53.07%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    insights = [
        ("🗺️","Top Feature",     "Location_Index gain 37.1 — 3× stronger than any other feature"),
        ("💊","DRUG over-recall", "Recall=69% but precision=39% — model over-predicts drug crimes"),
        ("🏠","PROPERTY gap",     "Precision=74% but recall=54% — misses nearly half of actual property crimes"),
        ("⚠️","VIOLENT weakest", "Both precision(51%) and recall(43%) weak — hardest class to separate"),
    ]
    for col, (icon, title, text) in zip(st.columns(4), insights):
        with col:
            st.markdown(
                f'<div class="insight-card"><div style="font-size:1.3rem;">{icon}</div>'
                f'<div class="insight-title">{title}</div>'
                f'<div class="insight-text">{text}</div></div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    st.markdown('<div class="main-title">Crime <span class="red-accent">Visualizations</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">EDA charts from the cleaned dataset</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    viz_files = {
        "Top 10 Crime Types":             "top_10_crime_types.png",
        "Crime Distribution by District": "district_crime_distribution.png",
        "Yearly Crime Trend":             "yearly_crime_trend.png",
        "Crime Type vs District Heatmap": "crime_type_district_heatmap.png",
        "Year vs District Heatmap":       "year_district_heatmap.png",
        "Hourly Crime Pattern":           "hourly_crime_pattern.png",
        "Crime Type vs Hour Heatmap":     "crime_type_hour_heatmap.png",
        "Day of Week Pattern":            "day_of_week_crime_pattern.png",
        "Monthly Crime Pattern":          "monthly_crime_pattern.png",
        "Correlation Matrix":             "correlation_matrix.png",
    }
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else "."
    available  = {}
    for title, fname in viz_files.items():
        for base in [".", script_dir, PROJECT_DIR]:
            full = os.path.join(base, fname)
            if os.path.exists(full):
                available[title] = full
                break

    if not available:
        st.warning("⚠️ No chart images found in the expected directories.")
        return

    mode = st.radio("", ["Select one chart","Show all charts"],
                    horizontal=True, label_visibility="collapsed")
    if mode == "Select one chart":
        sel = st.selectbox("", list(available), label_visibility="collapsed")
        st.markdown(f'<div class="section-header">{sel}</div>', unsafe_allow_html=True)
        st.image(Image.open(available[sel]), use_container_width=True)
    else:
        for title, path in available.items():
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(Image.open(path), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# BUG FIX: use len(proba) as the single source of truth for all array lengths
# ═══════════════════════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown('<div class="main-title">Crime Category <span class="red-accent">Prediction</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">XGBoost predicts the crime category from location & time</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        return

    loc_map, label_list = load_indexer_mappings()

    # Build category dict keyed by index
    CRIME_CATEGORIES = {i: name for i, name in enumerate(label_list)}

    # Location options: first 30 sorted by index, with safe fallback
    LOC_OPTIONS = (
        {label: idx for label, idx in sorted(loc_map.items(), key=lambda x: x[1])[:30]}
        if loc_map
        else {"STREET": 0, "RESIDENCE": 1, "APARTMENT": 2, "SIDEWALK": 3}
    )

    with st.form("pred_form"):
        # ── Date & Time ────────────────────────────────────────────────────
        st.markdown('<div class="form-section-title">📅 Date & Time</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            date_input = st.date_input("Date", datetime.now())
        with c2:
            hour_val = st.slider("Hour (24h)", 0, 23, 12, format="%d:00")
        with c3:
            is_night   = hour_val >= 20 or hour_val < 6
            is_weekend = date_input.weekday() >= 5
            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:10px;padding:14px;margin-top:28px;">
                <div style="font-size:.7rem;color:#9ca3af;text-transform:uppercase;">Auto-detected</div>
                <div style="font-weight:700;font-size:.95rem;color:#1a1a2e;margin-top:4px;">
                {DOW_NAMES[date_input.weekday()]} ·
                {"🌙 Night" if is_night else "☀️ Day"} ·
                {"Weekend" if is_weekend else "Weekday"}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Location ───────────────────────────────────────────────────────
        st.markdown('<div class="form-section-title">📍 Location</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            district = st.selectbox("Police District", list(range(1, 32)), index=10)
            ward     = st.number_input("Ward", 1, 50, 28)
        with c2:
            community_area = st.number_input("Community Area", 1, 77, 32)
            beat           = st.number_input("Beat", 100, 2535, 1100)
        with c3:
            latitude  = st.number_input("Latitude",  41.60, 42.00, 41.8781, format="%.4f")
            longitude = st.number_input("Longitude", -87.90,-87.50,-87.6298, format="%.4f")

        # ── Location Type ──────────────────────────────────────────────────
        st.markdown('<div class="form-section-title">🗺️ Location Type</div>', unsafe_allow_html=True)
        st.info("💡 Location_Index is the #1 predictor (gain 37.1). Select what best matches the crime scene.")
        loc_label = st.selectbox("Where did it happen?", list(LOC_OPTIONS))
        loc_index = LOC_OPTIONS[loc_label]

        # ── Advanced ───────────────────────────────────────────────────────
        with st.expander("⚙️ Advanced: Historical Crime Density"):
            c1, c2, c3 = st.columns(3)
            with c1: district_count  = st.number_input("District Crime Count",  0, 200000, 50000)
            with c2: beat_count      = st.number_input("Beat Crime Count",       0,  10000,  1500)
            with c3: community_count = st.number_input("Community Crime Count",  0,  50000, 10000)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Predict Crime Category")

    # ── Run prediction ─────────────────────────────────────────────────────────
    if submitted:
        with st.spinner("Running model..."):
            features = prepare_features(
                district, community_area, date_input.year, date_input.month,
                hour_val, date_input.weekday(),
                latitude, longitude, beat, ward,
                district_count, beat_count, community_count, loc_index,
            )
            dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLS)
            proba   = model.predict(dmatrix)[0]   # shape: (n_classes,)

        # ── KEY FIX: derive everything from the actual model output length ──
        n_classes = len(proba)
        pred_idx  = int(np.argmax(proba))
        category  = CRIME_CATEGORIES.get(pred_idx, f"Class_{pred_idx}")
        emoji     = CRIME_EMOJIS.get(category, "❓")

        # Result banner
        st.markdown(f"""<div class="result-box">
            <div style="font-size:3rem;">{emoji}</div>
            <div style="color:#9ca3af;font-size:.8rem;text-transform:uppercase;letter-spacing:2px;">Predicted</div>
            <div class="result-category">{category.replace("_"," ")}</div>
            <div style="font-size:1rem;color:#9ca3af;">Confidence: {proba[pred_idx]*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])

        with c1:
            # All three arrays use n_classes — no more length mismatch
            category_labels = [
                CRIME_CATEGORIES.get(i, f"Class_{i}").replace("_", " ")
                for i in range(n_classes)
            ]
            prob_df = pd.DataFrame({
                "Category":    category_labels,
                "Probability": proba * 100,
                "idx":         list(range(n_classes)),
            }).sort_values("Probability", ascending=True)

            bar_colors = [
                CRIME_COLORS.get(CRIME_CATEGORIES.get(i, ""), "#64748b")
                for i in prob_df["idx"]
            ]
            fig_prob = go.Figure(go.Bar(
                x=prob_df["Probability"],
                y=prob_df["Category"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                textposition="outside",
            ))
            fig_prob.update_layout(
                height=240, template="plotly_white",
                xaxis=dict(range=[0, 110], title="Probability (%)"),
                font=dict(family="Poppins"),
                margin=dict(t=10, b=10, l=10, r=60),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with c2:
            m         = RESULTS["classes"].get(category, {})
            p         = m.get("precision", 0)
            bar_color = "#16a34a" if p >= .65 else "#f59e0b" if p >= .45 else "#e63946"
            night_str = "🌙 Night" if (hour_val >= 20 or hour_val < 6) else "☀️ Day"
            st.markdown(f"""
            <div style="font-size:.87rem;line-height:2.2;color:#374151;">
                📅 <b>{date_input.strftime('%b %d, %Y')}</b><br>
                ⏰ <b>{hour_val}:00</b> — {night_str}<br>
                📆 <b>{DOW_NAMES[date_input.weekday()]}</b><br>
                🏛️ District <b>{district}</b> · Ward <b>{ward}</b><br>
                📍 <b>{loc_label}</b> (idx: {loc_index})<br>
                🗺️ {latitude:.4f}, {longitude:.4f}
            </div>
            <div style="background:#f8fafc;border-radius:10px;padding:12px 16px;
                        margin-top:12px;border-left:4px solid {bar_color};">
                <div style="font-size:.78rem;color:#6b7280;">
                    Historical precision for {category.replace('_',' ')}
                </div>
                <div style="font-size:1.8rem;font-weight:800;color:{bar_color};">{p:.0%}</div>
                <div style="font-size:.75rem;color:#9ca3af;">
                    {"✅ Usually correct" if p >= .65 else "⚠️ Often wrong — treat as a hint"}
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MAPS
# ═══════════════════════════════════════════════════════════════════════════════
def page_maps():
    st.markdown('<div class="main-title">Interactive <span class="red-accent">Maps</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Explore crime geography across Chicago</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    map_files = {
        "🎨 Crime by Type":     "chicago_crime_by_type_map.html",
        "🏛️ Crime by Category": "chicago_crime_category_map.html",
    }
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else "."
    available  = {}
    for label, fname in map_files.items():
        for base in [".", script_dir, PROJECT_DIR]:
            full = os.path.join(base, fname)
            if os.path.exists(full):
                available[label] = full
                break

    if not available:
        st.warning("⚠️ No map HTML files found.")
        return

    sel = st.radio("", list(available), horizontal=True, label_visibility="collapsed")
    try:
        with open(available[sel], encoding="utf-8") as f:
            components.html(f.read(), height=580, scrolling=True)
    except Exception as e:
        st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
def page_trends():
    st.markdown('<div class="main-title">Crime <span class="red-accent">Trends</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Year-over-year patterns across 4 categories</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    years      = [2001, 2002, 2003, 2004]
    cat_trends = {
        "PROPERTY_CRIME": [210000, 205000, 198000, 191000],
        "VIOLENT_CRIME":  [139100, 136000, 131900, 128800],
        "DRUG_CRIME":     [ 51000,  50000,  48000,  47000],
        "OTHER":          [ 63000,  62000,  60000,  58000],
    }
    palette = ["#2563eb","#e63946","#8b5cf6","#16a34a"]
    total   = [sum(cat_trends[c][i] for c in cat_trends) for i in range(4)]

    fig_t = go.Figure(go.Scatter(
        x=years, y=total, mode="lines+markers",
        line=dict(color="#e63946", width=4), marker=dict(size=12),
        fill="tozeroy", fillcolor="rgba(230,57,70,.08)",
    ))
    fig_t.update_layout(
        title="Overall Crime Trend (2001–2004)",
        xaxis_title="Year", yaxis_title="Crimes",
        height=320, template="plotly_white",
        font=dict(family="Poppins"), margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-header">By Category</div>', unsafe_allow_html=True)
        fig_c = go.Figure()
        for i, (cat, vals) in enumerate(cat_trends.items()):
            fig_c.add_trace(go.Scatter(
                x=years, y=vals, mode="lines+markers",
                name=f"{CRIME_EMOJIS[cat]} {cat.replace('_CRIME','').replace('_',' ')}",
                line=dict(width=3, color=palette[i]), marker=dict(size=9),
            ))
        fig_c.update_layout(
            height=360, template="plotly_white",
            font=dict(family="Poppins"), margin=dict(t=20, b=50),
            legend=dict(orientation="h", y=-0.3), hovermode="x unified",
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
        totals = {cat: sum(vals) for cat, vals in cat_trends.items()}
        fig_p  = px.pie(
            values=list(totals.values()),
            names=[c.replace("_CRIME","").replace("_"," ") for c in totals],
            color_discrete_sequence=palette,
            hole=0.45,
        )
        fig_p.update_layout(
            height=280, font=dict(family="Poppins"),
            margin=dict(t=20, b=20, l=0, r=0),
        )
        fig_p.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown("""
        <div style="font-size:.84rem;line-height:2;color:#374151;">
        🏠 <b>Property</b> ~45%<br>
        ⚠️ <b>Violent</b> ~31% (incl. Weapons)<br>
        📋 <b>Other</b> ~13%<br>
        💊 <b>Drug</b> ~11%
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()
    if   page == "📊  Overview":         page_overview()
    elif page == "📈  Visualizations":   page_visualizations()
    elif page == "🔮  Crime Prediction": page_prediction()
    elif page == "📍  Interactive Maps": page_maps()
    elif page == "📉  Trends Analysis":  page_trends()


if __name__ == "__main__":
    main()
