import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Disaster Alert System",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #07111f, #173b63);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 40px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    padding: 20px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

.low {
    padding: 20px;
    border-radius: 15px;
    background: #dcfce7;
    border-left: 7px solid #16a34a;
}

.medium {
    padding: 20px;
    border-radius: 15px;
    background: #fef9c3;
    border-left: 7px solid #ca8a04;
}

.high {
    padding: 20px;
    border-radius: 15px;
    background: #ffedd5;
    border-left: 7px solid #ea580c;
}

.extreme {
    padding: 20px;
    border-radius: 15px;
    background: #fee2e2;
    border-left: 7px solid #dc2626;
}

.disclaimer {
    padding: 18px;
    border-radius: 12px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "model" not in st.session_state:
    st.session_state.model = None

if "model_features" not in st.session_state:
    st.session_state.model_features = []

if "target_encoder" not in st.session_state:
    st.session_state.target_encoder = None

if "model_accuracy" not in st.session_state:
    st.session_state.model_accuracy = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(df, keywords):

    for col in df.columns:

        name = clean_name(col)

        for keyword in keywords:

            if keyword in name:
                return col

    return None


def risk_class(score):

    if score >= 80:
        return "extreme"

    if score >= 60:
        return "high"

    if score >= 35:
        return "medium"

    return "low"


def flood_category(score):

    if score >= 80:
        return "EXTREME"

    if score >= 60:
        return "HIGH"

    if score >= 35:
        return "MODERATE"

    return "LOW"


def calculate_flood_score(
    rainfall,
    humidity,
    temperature,
    wind,
    pressure,
    water_level,
    soil_moisture
):

    score = 0

    # Rainfall
    if rainfall >= 200:
        score += 35

    elif rainfall >= 100:
        score += 27

    elif rainfall >= 50:
        score += 17

    elif rainfall >= 20:
        score += 8

    # Water level
    if water_level >= 8:
        score += 30

    elif water_level >= 5:
        score += 23

    elif water_level >= 3:
        score += 12

    # Soil moisture
    if soil_moisture >= 80:
        score += 15

    elif soil_moisture >= 60:
        score += 10

    elif soil_moisture >= 40:
        score += 5

    # Humidity
    if humidity >= 90:
        score += 10

    elif humidity >= 75:
        score += 6

    # Pressure
    if pressure < 990:
        score += 8

    elif pressure < 1000:
        score += 4

    # Wind
    if wind >= 70:
        score += 7

    elif wind >= 40:
        score += 4

    return min(score, 100)


def earthquake_score(
    historical,
    magnitude,
    depth,
    latitude,
    longitude
):

    score = 0

    if historical:
        score += 30

    if magnitude >= 6:
        score += 35

    elif magnitude >= 5:
        score += 25

    elif magnitude >= 4:
        score += 15

    elif magnitude >= 3:
        score += 8

    if depth < 10:
        score += 20

    elif depth < 30:
        score += 12

    elif depth < 70:
        score += 6

    if latitude != 0 and longitude != 0:
        score += 5

    return min(score, 100)


def earthquake_category(score):

    if score >= 75:
        return "HIGH"

    elif score >= 50:
        return "MODERATE"

    elif score >= 25:
        return "LOW"

    return "VERY LOW"


def get_flood_advice(level):

    if level in ["HIGH", "EXTREME"]:

        return [
            "Move to higher ground if flooding is occurring or evacuation is advised.",
            "Never walk or drive through floodwater.",
            "Keep your phone charged.",
            "Keep medicines and important documents ready.",
            "Stay away from electrical equipment and downed power lines.",
            "Follow official disaster-management instructions."
        ]

    return [
        "Monitor official weather and flood warnings.",
        "Keep an emergency kit ready.",
        "Know the nearest safe/high-ground location.",
        "Protect important documents.",
        "Keep your phone and power bank charged."
    ]


def train_flood_model(df):

    """
    Automatically attempts to train a Random Forest
    when the uploaded dataset contains a recognizable
    flood target column.
    """

    target_col = find_column(
        df,
        [
            "flood",
            "flood risk",
            "floodrisk",
            "flood label",
            "flood status"
        ]
    )

    if target_col is None:
        return None, [], None, None

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        c for c in numeric_columns
        if c != target_col
    ]

    if len(numeric_columns) < 2:
        return None, [], None, None

    work = df[numeric_columns + [target_col]].copy()

    work = work.dropna()

    if len(work) < 30:
        return None, [], None, None

    if work[target_col].nunique() < 2:
        return None, [], None, None

    X = work[numeric_columns]

    y_raw = work[target_col].astype(str)

    encoder = LabelEncoder()

    y = encoder.fit_transform(y_raw)

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

    except:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    return (
        model,
        numeric_columns,
        encoder,
        accuracy
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🌍 AI Disaster Alert System</h1>

<p>
AI-powered Flood Risk Prediction,
Earthquake Risk Analysis and Emergency Assistant
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌍 Disaster AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🌊 Flood Prediction",
        "🌎 Earthquake Risk",
        "🤖 AI Assistant",
        "🚨 Emergency Guide",
        "📊 Dataset Analysis",
        "🗺️ Risk Map",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.session_state.dataset = df

        st.sidebar.success(
            f"Loaded {len(df):,} records"
        )

        # Automatically attempt ML training
        model, features, encoder, accuracy = train_flood_model(df)

        if model is not None:

            st.session_state.model = model
            st.session_state.model_features = features
            st.session_state.target_encoder = encoder
            st.session_state.model_accuracy = accuracy

            st.sidebar.success(
                f"Flood ML model trained: {accuracy:.1%}"
            )

        else:

            st.session_state.model = None

            st.sidebar.info(
                "No compatible flood target found. "
                "Risk-score mode is active."
            )

    except Exception as e:

        st.sidebar.error(
            f"Dataset error: {e}"
        )


# ============================================================
# DEMO MODE
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🎬 Demo Mode")

demo = st.sidebar.checkbox(
    "Enable Demo"
)

demo_scenario = st.sidebar.selectbox(
    "Select scenario",
    [
        "Normal",
        "Heavy Rainfall",
        "Extreme Flood"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader(
        "📊 Disaster Monitoring Dashboard"
    )

    if demo:

        if demo_scenario == "Normal":

            rainfall = 10
            humidity = 55
            temperature = 28
            wind = 15
            pressure = 1015
            water_level = 1
            soil = 25

        elif demo_scenario == "Heavy Rainfall":

            rainfall = 130
            humidity = 88
            temperature = 25
            wind = 35
            pressure = 995
            water_level = 5
            soil = 70

        else:

            rainfall = 250
            humidity = 95
            temperature = 24
            wind = 75
            pressure = 985
            water_level = 9
            soil = 90

        flood_score = calculate_flood_score(
            rainfall,
            humidity,
            temperature,
            wind,
            pressure,
            water_level,
            soil
        )

    else:

        flood_score = 20

    flood_level = flood_category(
        flood_score
    )

    earthquake = 15

    earthquake_level = earthquake_category(
        earthquake
    )

    overall_score = max(
        flood_score,
        earthquake
    )

    if overall_score >= 80:
        alert = "EMERGENCY"

    elif overall_score >= 60:
        alert = "WARNING"

    elif overall_score >= 35:
        alert = "WATCH"

    else:
        alert = "NORMAL"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌊 Flood Risk",
        flood_level,
        f"{flood_score}%"
    )

    col2.metric(
        "🌎 Earthquake Risk",
        earthquake_level,
        f"{earthquake}%"
    )

    col3.metric(
        "🚨 Alert Status",
        alert
    )

    col4.metric(
        "📁 Dataset Records",
        len(st.session_state.dataset)
        if st.session_state.dataset is not None
        else 0
    )

    st.markdown("---")

    if alert == "EMERGENCY":

        st.error(
            "🔴 EMERGENCY — High-risk conditions detected. "
            "Follow official emergency instructions."
        )

    elif alert == "WARNING":

        st.warning(
            "🟠 WARNING — Hazardous conditions may be developing."
        )

    elif alert == "WATCH":

        st.warning(
            "🟡 WATCH — Monitor weather and official alerts."
        )

    else:

        st.success(
            "🟢 NORMAL — No major risk detected."
        )

    if demo:

        st.subheader(
            "🌧️ Environmental Conditions"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Rainfall",
            f"{rainfall} mm"
        )

        b.metric(
            "Humidity",
            f"{humidity}%"
        )

        c.metric(
            "Water Level",
            f"{water_level} m"
        )

        d.metric(
            "Soil Moisture",
            f"{soil}%"
        )

        chart_data = pd.DataFrame({
            "Factor": [
                "Rainfall",
                "Humidity",
                "Water Level",
                "Soil Moisture",
                "Wind"
            ],
            "Value": [
                rainfall,
                humidity,
                water_level * 10,
                soil,
                wind
            ]
        })

        fig = px.bar(
            chart_data,
            x="Factor",
            y="Value",
            title="Environmental Risk Factors"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "🧠 AI Risk Explanation"
    )

    if flood_score >= 60:

        st.write(
            "The flood-risk indicator is elevated because "
            "environmental conditions such as rainfall, "
            "water level, humidity and soil moisture "
            "indicate increased flooding potential."
        )

    else:

        st.write(
            "Current conditions indicate relatively "
            "low flood risk."
        )

    st.markdown("""
<div class="disclaimer">

<b>⚠️ Important Safety Disclaimer</b>

This is an academic AI decision-support prototype.
It does not guarantee natural-disaster prediction.

Earthquake risk is an indicator and cannot predict
the exact time, location or magnitude of an earthquake.

Always follow official government and emergency-management
warnings.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FLOOD PREDICTION
# ============================================================

elif page == "🌊 Flood Prediction":

    st.subheader(
        "🌊 AI Flood Risk Prediction"
    )

    st.write(
        "Enter current environmental conditions."
    )

    col1, col2 = st.columns(2)

    with col1:

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            0.0,
            1000.0,
            50.0
        )

        humidity = st.slider(
            "💧 Humidity (%)",
            0,
            100,
            70
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            -20.0,
            60.0,
            28.0
        )

        wind = st.number_input(
            "💨 Wind Speed (km/h)",
            0.0,
            300.0,
            20.0
        )

    with col2:

        pressure = st.number_input(
            "🌬️ Pressure (hPa)",
            850.0,
            1100.0,
            1010.0
        )

        water_level = st.number_input(
            "🌊 Water Level (m)",
            0.0,
            20.0,
            2.0
        )

        soil = st.slider(
            "🌱 Soil Moisture (%)",
            0,
            100,
            40
        )

    if st.button(
        "🔍 ANALYZE FLOOD RISK",
        type="primary",
        use_container_width=True
    ):

        score = calculate_flood_score(
            rainfall,
            humidity,
            temperature,
            wind,
            pressure,
            water_level,
            soil
        )

        level = flood_category(
            score
        )

        st.markdown("---")

        st.markdown(
            f"""
<div class="{risk_class(score)}">

<h2>🌊 Flood Risk: {level}</h2>

<h3>Risk Score: {score}%</h3>

</div>
""",
            unsafe_allow_html=True
        )

        st.subheader(
            "🧠 Major Risk Factors"
        )

        factors = []

        if rainfall >= 100:
            factors.append(
                "🌧️ Heavy rainfall"
            )

        if water_level >= 5:
            factors.append(
                "🌊 High water level"
            )

        if soil >= 70:
            factors.append(
                "🌱 High soil moisture"
            )

        if humidity >= 85:
            factors.append(
                "💧 High humidity"
            )

        if pressure < 1000:
            factors.append(
                "🌬️ Low atmospheric pressure"
            )

        if wind >= 50:
            factors.append(
                "💨 Strong wind"
            )

        if not factors:

            factors.append(
                "No major risk factor detected."
            )

        for factor in factors:

            st.write(
                "• " + factor
            )

        st.subheader(
            "🚨 Recommended Safety Actions"
        )

        for advice in get_flood_advice(level):

            st.write(
                "✅ " + advice
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={
                    "text": "Flood Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ML model information
    if st.session_state.model is not None:

        st.markdown("---")

        st.subheader(
            "🤖 Machine Learning Model"
        )

        st.success(
            f"Random Forest model trained from uploaded dataset. "
            f"Test accuracy: "
            f"{st.session_state.model_accuracy:.2%}"
        )

        st.write(
            "Model features:"
        )

        st.write(
            st.session_state.model_features
        )


# ============================================================
# EARTHQUAKE
# ============================================================

elif page == "🌎 Earthquake Risk":

    st.subheader(
        "🌎 Earthquake Risk Indicator"
    )

    st.warning(
        "This module provides a risk indicator, not an exact "
        "earthquake prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        historical = st.checkbox(
            "Historical seismic activity"
        )

        magnitude = st.number_input(
            "Historical/Recent Magnitude",
            0.0,
            10.0,
            3.0
        )

        depth = st.number_input(
            "Depth (km)",
            0.0,
            700.0,
            50.0
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            -90.0,
            90.0,
            23.25
        )

        longitude = st.number_input(
            "Longitude",
            -180.0,
            180.0,
            77.41
        )

    if st.button(
        "🌎 ANALYZE EARTHQUAKE RISK",
        type="primary",
        use_container_width=True
    ):

        score = earthquake_score(
            historical,
            magnitude,
            depth,
            latitude,
            longitude
        )

        level = earthquake_category(
            score
        )

        st.markdown("---")

        st.markdown(
            f"""
<div class="{risk_class(score)}">

<h2>🌎 Earthquake Risk: {level}</h2>

<h3>Risk Indicator: {score}%</h3>

</div>
""",
            unsafe_allow_html=True
        )

        st.subheader(
            "🔎 Factors Considered"
        )

        st.write(
            "• Historical seismic activity"
        )

        st.write(
            "• Magnitude"
        )

        st.write(
            "• Depth"
        )

        st.write(
            "• Geographic location"
        )

        st.subheader(
            "🛡️ Earthquake Safety"
        )

        st.write(
            "⬇️ DROP to the ground."
        )

        st.write(
            "🛡️ COVER your head and neck."
        )

        st.write(
            "✊ HOLD ON to sturdy shelter."
        )

        st.write(
            "🚫 Stay away from windows."
        )

        st.write(
            "🚫 Do not use elevators."
        )

        st.markdown("""
<div class="disclaimer">

<b>Scientific Limitation:</b>

Ordinary weather data cannot reliably predict earthquakes.
A real earthquake-risk system requires appropriate seismic,
geological and historical earthquake datasets.

</div>
""", unsafe_allow_html=True)


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "🤖 AI Assistant":

    st.subheader(
        "🤖 Disaster AI Assistant"
    )

    st.write(
        "Ask a disaster-safety question."
    )

    question = st.text_input(
        "💬 Your question",
        placeholder="What should I do during an earthquake?"
    )

    if question:

        q = question.lower()

        if "earthquake" in q:

            response = """
## 🌎 Earthquake Safety

During an earthquake:

1. DROP to the ground.
2. COVER your head and neck.
3. HOLD ON to sturdy shelter.
4. Stay away from windows.
5. Do not use elevators.
6. If outdoors, move away from buildings,
   trees and power lines.

After the earthquake:

- Check for injuries.
- Expect aftershocks.
- Avoid damaged buildings.
- Follow official instructions.
"""

        elif "flood" in q:

            response = """
## 🌊 Flood Safety

During flooding:

1. Move to higher ground if advised.
2. Never walk or drive through floodwater.
3. Stay away from electrical hazards.
4. Keep your phone charged.
5. Follow official evacuation instructions.
"""

        elif "kit" in q:

            response = """
## 🎒 Emergency Kit

Recommended items include:

- Drinking water
- Non-perishable food
- First-aid kit
- Medicines
- Flashlight
- Batteries
- Power bank
- Important documents
- Emergency contacts
"""

        elif "evacuat" in q:

            response = """
## 🚨 Evacuation

If official authorities issue an evacuation order,
follow it promptly.

Use official evacuation routes and do not depend
only on this application for emergency decisions.
"""

        else:

            response = """
## 🤖 I Can Help With

🌊 Flood safety

🌎 Earthquake safety

🎒 Emergency kits

🚨 Evacuation preparation

🏠 Disaster preparedness

For a real emergency, always follow official
emergency services and government instructions.
"""

        st.markdown(
            response
        )


# ============================================================
# EMERGENCY GUIDE
# ============================================================

elif page == "🚨 Emergency Guide":

    st.subheader(
        "🚨 Emergency Preparedness"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🌊 Flood",
            "🌎 Earthquake",
            "🎒 Emergency Kit"
        ]
    )

    with tab1:

        st.header(
            "🌊 Flood Safety"
        )

        st.subheader(
            "Before"
        )

        st.write(
            "✅ Monitor official warnings."
        )

        st.write(
            "✅ Prepare an emergency kit."
        )

        st.write(
            "✅ Protect important documents."
        )

        st.write(
            "✅ Know evacuation routes."
        )

        st.write(
            "✅ Charge phones and power banks."
        )

        st.subheader(
            "During"
        )

        st.write(
            "🚨 Move to higher ground when advised."
        )

        st.write(
            "🚨 Never drive through floodwater."
        )

        st.write(
            "🚨 Avoid electrical hazards."
        )

        st.write(
            "🚨 Follow evacuation instructions."
        )

        st.subheader(
            "After"
        )

        st.write(
            "✅ Avoid contaminated water."
        )

        st.write(
            "✅ Avoid damaged buildings."
        )

        st.write(
            "✅ Watch for electrical hazards."
        )

    with tab2:

        st.header(
            "🌎 Earthquake Safety"
        )

        st.write(
            "⬇️ DROP"
        )

        st.write(
            "🛡️ COVER"
        )

        st.write(
            "✊ HOLD ON"
        )

        st.write(
            "🚫 Stay away from windows."
        )

        st.write(
            "🚫 Do not use elevators."
        )

        st.subheader(
            "After Earthquake"
        )

        st.write(
            "✅ Check injuries."
        )

        st.write(
            "✅ Expect aftershocks."
        )

        st.write(
            "✅ Avoid damaged structures."
        )

        st.write(
            "✅ Follow official instructions."
        )

    with tab3:

        st.header(
            "🎒 Emergency Kit"
        )

        items = [
            "Drinking water",
            "Non-perishable food",
            "First-aid kit",
            "Medicines",
            "Flashlight",
            "Batteries",
            "Power bank",
            "Important documents",
            "Emergency contacts",
            "Hygiene supplies"
        ]

        for item in items:

            st.checkbox(
                item
            )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":

    st.subheader(
        "📊 Dataset Analysis"
    )

    df = st.session_state.dataset

    if df is None:

        st.info(
            "Upload a CSV dataset from the sidebar."
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

        c2.metric(
            "Columns",
            df.shape[1]
        )

        c3.metric(
            "Missing Values",
            int(
                df.isna().sum().sum()
            )
        )

        st.markdown("---")

        st.subheader(
            "👀 Dataset Preview"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader(
            "📋 Column Information"
        )

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(x)
                for x in df.dtypes
            ],
            "Missing": [
                int(df[c].isna().sum())
                for c in df.columns
            ],
            "Unique": [
                int(df[c].nunique())
                for c in df.columns
            ]
        })

        st.dataframe(
            info,
            use_container_width=True
        )

        numeric = df.select_dtypes(
            include=np.number
        )

        if not numeric.empty:

            st.subheader(
                "📈 Data Distribution"
            )

            selected = st.selectbox(
                "Select feature",
                numeric.columns
            )

            fig = px.histogram(
                df,
                x=selected,
                title=f"Distribution: {selected}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if len(numeric.columns) >= 2:

                st.subheader(
                    "🔥 Correlation Matrix"
                )

                fig = px.imshow(
                    numeric.corr(),
                    text_auto=True,
                    title="Feature Correlation"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# MAP
# ============================================================

elif page == "🗺️ Risk Map":

    st.subheader(
        "🗺️ Geographic Risk Map"
    )

    df = st.session_state.dataset

    if df is None:

        st.info(
            "Upload a CSV dataset containing latitude "
            "and longitude."
        )

    else:

        lat_col = find_column(
            df,
            ["latitude", "lat"]
        )

        lon_col = find_column(
            df,
            ["longitude", "longitude", "lon", "lng"]
        )

        if lat_col and lon_col:

            map_df = df[
                [lat_col, lon_col]
            ].copy()

            map_df.columns = [
                "latitude",
                "longitude"
            ]

            map_df["latitude"] = pd.to_numeric(
                map_df["latitude"],
                errors="coerce"
            )

            map_df["longitude"] = pd.to_numeric(
                map_df["longitude"],
                errors="coerce"
            )

            map_df = map_df.dropna()

            st.map(
                map_df,
                latitude="latitude",
                longitude="longitude",
                use_container_width=True
            )

            st.success(
                f"{len(map_df):,} locations displayed."
            )

        else:

            st.warning(
                "Latitude/longitude columns were not found."
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.subheader(
        "ℹ️ About Project"
    )

    st.markdown("""
# 🌍 AI Disaster Alert & Emergency Assistant

## Objective

To develop an AI-powered disaster decision-support
system that analyzes environmental information and
provides flood-risk estimation, earthquake-risk
indicators and emergency safety assistance.

## Main Features

### 🌊 Flood Prediction
- Rainfall analysis
- Humidity
- Water level
- Soil moisture
- Atmospheric pressure
- Wind speed
- Random Forest ML when a suitable labeled dataset is uploaded

### 🌎 Earthquake Risk
- Historical seismic activity
- Magnitude
- Depth
- Geographic location

### 🤖 AI Assistant
Provides safety guidance for:

- Floods
- Earthquakes
- Emergency kits
- Evacuation
- Disaster preparation

### 📊 Visualization

- Interactive charts
- Dataset analysis
- Correlation matrix
- Geographic map
- Risk indicators

## Technologies

Python  
Streamlit  
Pandas  
NumPy  
Scikit-learn  
Plotly  
Machine Learning

## Future Scope

- Real-time weather APIs
- Government disaster alerts
- IoT sensors
- Satellite imagery
- GIS analysis
- Real-time river monitoring
- SMS/email notifications
- Advanced deep-learning models

## Limitation

This is an academic decision-support prototype.

It cannot guarantee prediction of natural disasters.

Earthquake prediction requires specialized seismic,
geological and historical datasets and cannot be
reliably obtained from ordinary weather variables alone.
""")

    st.markdown("""
<div class="disclaimer">

<b>⚠️ Safety Disclaimer:</b>

For actual emergencies, always follow official government,
emergency-service and disaster-management instructions.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌍 AI Disaster Alert System | BTech AI/ML Project"
)
