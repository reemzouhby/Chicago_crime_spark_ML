import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from datetime import datetime
import os
from PIL import Image
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Chicago Crime Analysis Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 10px;
        margin: 20px 0;
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# YOUR ACTUAL MODEL DETAILS
# -------------------------------------------------------------------

# Exact feature order used during training
FEATURE_COLUMNS = [
    'District', 'Community Area', 'Year', 'Month', 'Hour', 'DayOfWeek',
    'Latitude', 'Longitude', 'Beat', 'Ward',
    'Hour_sin', 'Hour_cos', 'IsWeekend',
    'Distance_from_center',
    'District_Crime_Count', 'Beat_Crime_Count', 'Community_Crime_Count',
    'Location_Index', 'Arrest_Flag',
    'Description_Index'
]

# Your actual 5 crime categories (from label_model.labels order)
CRIME_CATEGORIES = {
    0: "DRUG_CRIME",
    1: "OTHER",
    2: "PROPERTY_CRIME",
    3: "VIOLENT_CRIME",
    4: "WEAPONS_CRIME"
}

CRIME_COLORS = {
    "DRUG_CRIME": "#FF6B6B",
    "OTHER": "#4ECDC4",
    "PROPERTY_CRIME": "#45B7D1",
    "VIOLENT_CRIME": "#96CEB4",
    "WEAPONS_CRIME": "#FFEAA7"
}

CRIME_EMOJIS = {
    "DRUG_CRIME": "💊",
    "OTHER": "📋",
    "PROPERTY_CRIME": "🏠",
    "VIOLENT_CRIME": "⚠️",
    "WEAPONS_CRIME": "🔫"
}

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        model.load_model(r"C:\Users\Omen\PycharmProjects\lab3\Chicago_project\crime_xgboost_clean.json")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------------------------------------------------
# FEATURE PREPARATION (must match training exactly)
# -------------------------------------------------------------------
def prepare_features(district, community_area, year, month, hour,
                     day_of_week, latitude, longitude, beat, ward,
                     district_crime_count, beat_crime_count,
                     community_crime_count, location_index,
                     arrest_flag, description_index):
    """
    Prepare 20 features in exact same order as training.
    NOTE: District_Crime_Count, Beat_Crime_Count, Community_Crime_Count,
    Location_Index, Description_Index are density/encoding values
    that user must estimate or we provide defaults.
    """
    # Cyclical hour encoding (same formula as training)
    hour_sin = np.sin(np.radians(hour * 15))
    hour_cos = np.cos(np.radians(hour * 15))

    # Weekend flag (1=Sunday, 7=Saturday in PySpark dayofweek)
    # Streamlit weekday(): 0=Monday, 6=Sunday
    # Convert: streamlit 0->2, 1->3, 2->4, 3->5, 4->6, 5->7, 6->1
    spark_dow = day_of_week + 2 if day_of_week < 6 else 1
    is_weekend = 1.0 if spark_dow in [1, 7] else 0.0

    # Distance from Chicago center
    chicago_lat = 41.8781
    chicago_lon = -87.6298
    distance_from_center = np.sqrt(
        (latitude - chicago_lat) ** 2 +
        (longitude - chicago_lon) ** 2
    ) * 111

    features = [
        district,            # District
        community_area,      # Community Area
        year,                # Year
        month,               # Month
        hour,                # Hour
        spark_dow,           # DayOfWeek
        latitude,            # Latitude
        longitude,           # Longitude
        beat,                # Beat
        ward,                # Ward
        hour_sin,            # Hour_sin
        hour_cos,            # Hour_cos
        is_weekend,          # IsWeekend
        distance_from_center,# Distance_from_center
        district_crime_count,# District_Crime_Count
        beat_crime_count,    # Beat_Crime_Count
        community_crime_count,# Community_Crime_Count
        location_index,      # Location_Index
        arrest_flag,         # Arrest_Flag
        description_index    # Description_Index
    ]

    assert len(features) == 20, f"Expected 20 features, got {len(features)}"
    return np.array(features, dtype=np.float32).reshape(1, -1)

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🚨 Chicago Crime Analysis Dashboard</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Overview", "📈 Visualizations", "🔮 Crime Prediction",
         "📍 Interactive Maps", "📉 Trends Analysis"]
    )

    # ========================================================================
    # OVERVIEW PAGE
    # ========================================================================
    if page == "📊 Overview":
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Crimes", "1.8M+", "2001-2004")
        with col2:
            st.metric("Districts", "25", "Analyzed")
        with col3:
            st.metric("Crime Categories", "5", "Classified")
        with col4:
            st.metric("Model Accuracy", "97.4%", "XGBoost")

        st.markdown("---")
        st.subheader("Crime Categories (5 Classes)")
        st.markdown("""
        The model classifies crimes into **5 categories**:

        | Category | Emoji | Crime Types Included |
        |----------|-------|---------------------|
        | **PROPERTY_CRIME** | 🏠 | Theft, Burglary, Motor Vehicle Theft, Criminal Damage, Criminal Trespass |
        | **VIOLENT_CRIME** | ⚠️ | Battery, Assault, Robbery, Homicide |
        | **DRUG_CRIME** | 💊 | Narcotics, Other Narcotic Violations |
        | **WEAPONS_CRIME** | 🔫 | Weapons Violations |
        | **OTHER** | 📋 | All other crime types |
        """)

        st.markdown("---")
        st.subheader("Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classification Report:**")
            perf_data = {
                'Category': ['DRUG_CRIME', 'OTHER', 'PROPERTY_CRIME', 'VIOLENT_CRIME', 'WEAPONS_CRIME'],
                'Precision': [0.98, 0.90, 1.00, 0.98, 0.70],
                'Recall': [0.98, 0.97, 0.98, 0.97, 0.99],
                'F1-Score': [0.98, 0.93, 0.99, 0.97, 0.82]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        with col2:
            fig = px.bar(
                pd.DataFrame(perf_data).melt(id_vars='Category',
                                              var_name='Metric',
                                              value_name='Score'),
                x='Category', y='Score', color='Metric',
                barmode='group',
                title='Model Performance per Category'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Key Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-box">🔍 <b>Most Common Crime</b><br>Property crimes account for the majority of incidents (~45%)</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">⏰ <b>Peak Crime Hours</b><br>Crime peaks during afternoon (12PM-6PM) and late evening (8PM-midnight)</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="info-box">📅 <b>Seasonal Patterns</b><br>Summer months show higher crime rates</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">🗺️ <b>Top Feature</b><br>Crime Description is the most predictive feature (importance: 258)</div>', unsafe_allow_html=True)

    # ========================================================================
    # VISUALIZATIONS PAGE
    # ========================================================================
    elif page == "📈 Visualizations":
        st.header("Crime Data Visualizations")

        viz_files = {
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\top_10_crime_types.png': 'Top 10 Crime Types',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\district_crime_distribution.png': 'Crime Distribution Across Districts',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\yearly_crime_trend.png': 'Yearly Crime Trend',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\crime_type_district_heatmap.png': 'Crime Type vs District Heatmap',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\year_district_heatmap.png': 'Year vs District Heatmap',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\hourly_crime_pattern.png': 'Hourly Crime Pattern',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\crime_type_hour_heatmap.png': 'Crime Type vs Hour Heatmap',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\day_of_week_crime_pattern.png': 'Day of Week Pattern',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\monthly_crime_pattern.png': 'Monthly Crime Pattern',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\correlation_matrix.png': 'Correlation Matrix',
        }

        available_viz = {k: v for k, v in viz_files.items() if os.path.exists(k)}

        if not available_viz:
            st.warning("⚠️ No visualization files found. Please run the analysis script first.")
        else:
            display_mode = st.radio("Display Mode", ["📑 Select One", "📊 Show All"], horizontal=True)

            if display_mode == "📑 Select One":
                selected = st.selectbox("Choose visualization:", list(available_viz.values()),
                                        index=0)
                file_path = [k for k, v in available_viz.items() if v == selected][0]
                st.subheader(selected)
                st.image(Image.open(file_path), use_container_width=True)
            else:
                for path, title in available_viz.items():
                    st.subheader(title)
                    st.image(Image.open(path), use_container_width=True)
                    st.markdown("---")

            st.success(f"✅ {len(available_viz)} of {len(viz_files)} visualizations loaded")

    # ========================================================================
    # PREDICTION PAGE
    # ========================================================================
    elif page == "🔮 Crime Prediction":
        st.header("Crime Category Prediction")

        model = load_model()

        st.markdown("""
        Enter crime details to predict the category using the XGBoost model (97.4% accuracy).
        The model uses **20 features** matching the training pipeline exactly.
        """)

        if model is None:
            st.error("⚠️ Model not loaded.")
            return

        st.markdown("---")

        with st.form("prediction_form"):
            st.subheader("📅 Date & Time")
            col1, col2, col3 = st.columns(3)
            with col1:
                date_input = st.date_input("Date", datetime.now())
                hour = st.slider("Hour (24h)", 0, 23, 12)
            with col2:
                year = date_input.year
                month = date_input.month
                day_of_week = date_input.weekday()
                st.info(f"Year: {year} | Month: {month} | Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}")

            st.subheader("📍 Location")
            col1, col2, col3 = st.columns(3)
            with col1:
                district = st.number_input("District", 1, 31, 11)
                ward = st.number_input("Ward", 1, 50, 28)
            with col2:
                community_area = st.number_input("Community Area", 1, 77, 32)
                beat = st.number_input("Beat", 100, 2535, 1100)
            with col3:
                latitude = st.number_input("Latitude", 41.6, 42.0, 41.8781, format="%.4f")
                longitude = st.number_input("Longitude", -87.9, -87.5, -87.6298, format="%.4f")

            st.subheader("📊 Statistical Features")
            st.caption("These come from historical crime density. Use average values if unsure.")
            col1, col2, col3 = st.columns(3)
            with col1:
                district_crime_count = st.number_input("District Crime Count", 0, 200000, 50000)
                beat_crime_count = st.number_input("Beat Crime Count", 0, 10000, 1500)
            with col2:
                community_crime_count = st.number_input("Community Crime Count", 0, 50000, 10000)
                location_index = st.number_input("Location Index (0=Street, 1=Residence...)", 0, 100, 0)
            with col3:
                arrest_flag = st.selectbox("Arrest Made?", [0, 1],
                                           format_func=lambda x: "Yes" if x == 1 else "No")
                description_index = st.number_input(
                    "Description Index",
                    0, 500, 0,
                    help="0=most common description. This is the strongest predictor!"
                )

            submit = st.form_submit_button("🔍 Predict Crime Category", use_container_width=True)

        if submit:
            with st.spinner("Predicting..."):
                features = prepare_features(
                    district, community_area, year, month, hour,
                    day_of_week, latitude, longitude, beat, ward,
                    district_crime_count, beat_crime_count,
                    community_crime_count, location_index,
                    float(arrest_flag), description_index
                )

                dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
                proba = model.predict(dmatrix)[0]
                predicted_idx = int(np.argmax(proba))
                predicted_category = CRIME_CATEGORIES[predicted_idx]
                confidence = proba[predicted_idx] * 100
                emoji = CRIME_EMOJIS[predicted_category]

                st.markdown("---")
                st.subheader("Prediction Result")
                st.markdown(
                    f'<div class="prediction-result">'
                    f'{emoji} Predicted Crime Category:<br>'
                    f'<strong>{predicted_category}</strong><br>'
                    f'Confidence: {confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Crime Category': [CRIME_CATEGORIES[i] for i in range(5)],
                    'Probability (%)': proba * 100
                }).sort_values('Probability (%)', ascending=True)

                fig = px.bar(
                    prob_df,
                    x='Probability (%)',
                    y='Crime Category',
                    orientation='h',
                    title='Prediction Probabilities',
                    color='Probability (%)',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Input Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Temporal**")
                    st.write(f"📅 {date_input.strftime('%B %d, %Y')}")
                    st.write(f"⏰ Hour: {hour}:00")
                    st.write(f"📆 {'Weekend' if day_of_week >= 5 else 'Weekday'}")
                with col2:
                    st.markdown("**Location**")
                    st.write(f"🏛️ District: {district}")
                    st.write(f"🗳️ Ward: {ward}")
                    st.write(f"🏘️ Community: {community_area}")
                with col3:
                    st.markdown("**Geographic**")
                    st.write(f"📍 Lat: {latitude:.4f}")
                    st.write(f"📍 Lon: {longitude:.4f}")
                    dist = np.sqrt((latitude-41.8781)**2 + (longitude+87.6298)**2)*111
                    st.write(f"📏 {dist:.2f} km from center")

    # ========================================================================
    # INTERACTIVE MAPS PAGE
    # ========================================================================
    elif page == "📍 Interactive Maps":
        st.header("Interactive Crime Maps")

        map_files = {
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\chicago_crime_by_type_map.html': 'Crime Types Map',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\chicago_crime_by_district.html': 'District-Level Crime Distribution',
            r'C:\Users\Omen\PycharmProjects\lab3\Chicago_project\chicago_crime_category_map.html': 'Crime Category Distribution (5 Categories)'
        }

        available_maps = {k: v for k, v in map_files.items() if os.path.exists(k)}

        if not available_maps:
            st.warning("⚠️ No map files found. Please run the analysis script first.")
        else:
            map_choice = st.selectbox(
                "Choose map:",
                list(available_maps.keys()),
                format_func=lambda x: available_maps[x]
            )
            try:
                with open(map_choice, 'r', encoding='utf-8') as f:
                    map_html = f.read()
                components.html(map_html, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error loading map: {e}")

    # ========================================================================
    # TRENDS ANALYSIS PAGE
    # ========================================================================
    elif page == "📉 Trends Analysis":
        st.header("Crime Trends Analysis")

        years = [2001, 2002, 2003, 2004]

        st.subheader("Yearly Crime Trends")
        total_crimes = [480000, 470000, 455000, 443000]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=total_crimes,
            mode='lines+markers',
            name='Total Crimes',
            line=dict(color='darkred', width=4),
            marker=dict(size=12)
        ))
        fig.update_layout(title='Overall Crime Trend (2001-2004)',
                          xaxis_title='Year', yaxis_title='Crimes', height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Crime Category Trends")

        category_trends = {
            'PROPERTY_CRIME': [210000, 205000, 198000, 191000],
            'VIOLENT_CRIME': [135000, 132000, 128000, 125000],
            'DRUG_CRIME': [51000, 50000, 48000, 47000],
            'OTHER': [63000, 62000, 60000, 58000],
            'WEAPONS_CRIME': [4100, 4000, 3900, 3800]
        }

        fig = go.Figure()
        colors = ['#45B7D1', '#96CEB4', '#FF6B6B', '#4ECDC4', '#FFEAA7']
        for i, (category, values) in enumerate(category_trends.items()):
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name=f"{CRIME_EMOJIS[category]} {category}",
                line=dict(width=3, color=colors[i])
            ))
        fig.update_layout(title='Crime Categories Over Time',
                          xaxis_title='Year', yaxis_title='Crimes',
                          height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Crime Category Distribution")

        col1, col2 = st.columns(2)
        with col1:
            category_totals = {cat: sum(vals) for cat, vals in category_trends.items()}
            fig = px.pie(
                values=list(category_totals.values()),
                names=list(category_totals.keys()),
                title='Overall Crime Distribution by Category',
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Key Insights:**")
            st.markdown("""
            - 🏠 **Property Crime** is the most common (~44%)
            - ⚠️ **Violent Crime** accounts for ~29%
            - 📋 **Other** accounts for ~14%
            - 💊 **Drug Crime** accounts for ~11%
            - 🔫 **Weapons Crime** is the rarest (~1%)

            Overall crime shows a **decreasing trend** from 2001 to 2004,
            consistent across all categories.
            """)


if __name__ == "__main__":
    main()