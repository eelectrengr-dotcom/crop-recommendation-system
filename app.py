import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG (must be first st. call)
# -----------------------------
st.set_page_config(page_title="Crop AI", page_icon="🌱", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
le_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
    le = pickle.load(open(le_path, "rb"))
    load_error = None
except Exception as e:
    model, le, load_error = None, None, str(e)

# -----------------------------
# CROP EMOJIS / BLURBS
# -----------------------------
CROP_INFO = {
    "apple": ("🍎", "A temperate fruit crop, thrives in cool climates."),
    "banana": ("🍌", "A tropical fruit crop needing warmth and humidity."),
    "blackgram": ("🫘", "A short-duration pulse crop, drought tolerant."),
    "chickpea": ("🫛", "A cool-season legume, fixes nitrogen in soil."),
    "coconut": ("🥥", "A tropical palm needing high humidity and rainfall."),
    "coffee": ("☕", "A shade-loving shrub needing consistent rainfall."),
    "cotton": ("🌱", "A warm-season fiber crop needing long frost-free periods."),
    "grapes": ("🍇", "A vine crop that prefers well-drained soil."),
    "jute": ("🌾", "A fiber crop needing high humidity and warm weather."),
    "kidneybeans": ("🫘", "A legume crop preferring moderate rainfall."),
    "lentil": ("🥣", "A cool-season pulse, low water requirement."),
    "maize": ("🌽", "A cereal crop adaptable to varied climates."),
    "mango": ("🥭", "A tropical fruit tree needing a dry flowering season."),
    "mothbeans": ("🫘", "A drought-resistant legume suited to arid regions."),
    "mungbean": ("🫛", "A short-duration pulse, thrives in warm weather."),
    "muskmelon": ("🍈", "A warm-season fruit needing sandy, well-drained soil."),
    "orange": ("🍊", "A citrus fruit tree needing a subtropical climate."),
    "papaya": ("🍈", "A fast-growing tropical fruit tree."),
    "pigeonpeas": ("🫘", "A hardy legume suited to semi-arid regions."),
    "pomegranate": ("🍎", "A drought-tolerant fruit shrub."),
    "rice": ("🌾", "A staple cereal needing standing water / high rainfall."),
    "watermelon": ("🍉", "A warm-season fruit needing sandy soil and sun."),
}


def crop_display(name: str):
    key = str(name).strip().lower()
    return CROP_INFO.get(key, ("🌿", "A recommended crop for these conditions."))


# -----------------------------
# STYLING — instrument panel theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;700&display=swap');

.stApp {
    background:
        linear-gradient(rgba(10,20,15,0.94), rgba(8,14,11,0.97)),
        repeating-linear-gradient(0deg, #14231b 0px, #14231b 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, #14231b 0px, #14231b 1px, transparent 1px, transparent 40px),
        #0b0f0d;
}
.block-container { padding-top: 1.4rem; font-family: 'Rajdhani', sans-serif; }

/* ---------- HEADER / STATUS BAR ---------- */
.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0f1713;
    border: 1px solid #1f3d2c;
    border-radius: 10px;
    padding: 14px 22px;
    margin-bottom: 1.2rem;
}
.panel-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    color: #d6ffe8;
    letter-spacing: 1px;
}
.panel-sub {
    color: #5f8f75;
    font-size: 0.8rem;
    font-family: 'Share Tech Mono', monospace;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #0d1a13;
    border: 1px solid #2ecc8f;
    border-radius: 30px;
    padding: 6px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #2ecc8f;
}
.dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #2ecc8f;
    box-shadow: 0 0 8px #2ecc8f, 0 0 16px #2ecc8f;
    animation: pulse 1.6s infinite ease-in-out;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ---------- SENSOR TILES ---------- */
.tile-row { display: flex; gap: 10px; margin-bottom: 1.3rem; flex-wrap: wrap; }
.tile {
    flex: 1;
    min-width: 110px;
    background: #0f1713;
    border: 1px solid #1f3d2c;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}
.tile-icon { font-size: 1.1rem; }
.tile-label {
    color: #5f8f75;
    font-size: 0.68rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
}
.tile-value {
    font-family: 'Share Tech Mono', monospace;
    color: #2ecc8f;
    font-size: 1.25rem;
    font-weight: 700;
    text-shadow: 0 0 6px rgba(46,204,143,0.5);
}

/* ---------- INPUT PANELS ---------- */
.card {
    background: #0f1713;
    border: 1px solid #1f3d2c;
    border-radius: 14px;
    padding: 22px 26px 10px 26px;
    box-shadow: inset 0 0 25px rgba(0,0,0,0.35);
    margin-bottom: 1rem;
}
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    color: #d6ffe8;
    letter-spacing: 1px;
    margin-bottom: 14px;
    border-bottom: 1px solid #1f3d2c;
    padding-bottom: 8px;
}

/* Slider color override */
div[data-testid="stSlider"] span { font-family: 'Share Tech Mono', monospace; }

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(90deg, #0e8a55, #17c778);
    color: #eafff5;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    letter-spacing: 1px;
    border-radius: 8px;
    padding: 12px 30px;
    border: 1px solid #2ecc8f;
    width: 100%;
    transition: 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #17c778, #26e896);
    box-shadow: 0 0 18px rgba(46,204,143,0.45);
}

/* ---------- RESULT READOUT ---------- */
.readout {
    background: #0d1a13;
    border: 1px solid #2ecc8f;
    border-radius: 14px;
    padding: 26px;
    margin-top: 1.3rem;
    box-shadow: 0 0 30px rgba(46,204,143,0.12), inset 0 0 25px rgba(0,0,0,0.4);
}
.readout-label {
    font-family: 'Share Tech Mono', monospace;
    color: #5f8f75;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
}
.readout-crop {
    font-family: 'Share Tech Mono', monospace;
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    color: #2ecc8f;
    text-shadow: 0 0 14px rgba(46,204,143,0.6);
    margin: 0.2rem 0;
}
.readout-desc {
    text-align: center;
    color: #a9c9b8;
    font-size: 0.92rem;
}

/* ---------- TOP-3 CARDS ---------- */
.top-card {
    background: #0f1713;
    border: 1px solid #1f3d2c;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
}
.top-card:hover { border-color: #2ecc8f; }
.top-rank {
    font-family: 'Share Tech Mono', monospace;
    color: #5f8f75;
    font-size: 0.7rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER / STATUS BAR
# -----------------------------
st.markdown("""
<div class="panel-header">
    <div>
        <div class="panel-title">🌱 CROP-AI // RECOMMENDATION UNIT</div>
        <div class="panel-sub">RANDOM FOREST CLASSIFIER · SOIL &amp; WEATHER ANALYSIS MODULE</div>
    </div>
    <div class="status-pill"><div class="dot"></div> SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Couldn't load model files: {load_error}\n\nMake sure models/model.pkl and "
             f"models/label_encoder.pkl exist (run train.py first).")

# -----------------------------
# SIDEBAR — presets & about
# -----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Recommends the best crop using a **Random Forest** model trained on "
        "soil nutrients (N, P, K) and weather conditions (temperature, "
        "humidity, pH, rainfall)."
    )
    st.divider()
    st.subheader("⚡ Quick presets")
    preset = st.selectbox(
        "Load example conditions",
        ["-- Select a preset --", "Rice-like (wet, hot)", "Chickpea-like (cool, dry)", "Coffee-like (humid, mild)"],
    )

preset_values = {
    "Rice-like (wet, hot)": dict(n=90, p=42, k=43, temp=27.0, hum=82.0, ph=6.5, rain=230.0),
    "Chickpea-like (cool, dry)": dict(n=40, p=65, k=80, temp=18.0, hum=17.0, ph=7.3, rain=80.0),
    "Coffee-like (humid, mild)": dict(n=100, p=20, k=30, temp=25.0, hum=58.0, ph=6.8, rain=150.0),
}
defaults = preset_values.get(preset, dict(n=50, p=50, k=50, temp=25.0, hum=60.0, ph=6.5, rain=100.0))

# -----------------------------
# INPUT LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 SOIL NUTRIENT SENSORS</div>', unsafe_allow_html=True)
    N = st.slider("Nitrogen (N) — kg/ha", 0, 140, defaults["n"])
    P = st.slider("Phosphorus (P) — kg/ha", 0, 145, defaults["p"])
    K = st.slider("Potassium (K) — kg/ha", 0, 205, defaults["k"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌦 WEATHER TELEMETRY</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature — °C", 0.0, 50.0, float(defaults["temp"]))
    humidity = st.slider("Humidity — %", 0.0, 100.0, float(defaults["hum"]))
    ph = st.slider("Soil pH", 0.0, 14.0, float(defaults["ph"]))
    rainfall = st.slider("Rainfall — mm", 0.0, 350.0, float(defaults["rain"]))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# LIVE SENSOR READOUT STRIP
# -----------------------------
tiles = [
    ("🧪", "N", f"{N}"),
    ("⚗️", "P", f"{P}"),
    ("🔬", "K", f"{K}"),
    ("🌡️", "TEMP", f"{temperature:.1f}°"),
    ("💧", "HUMID", f"{humidity:.0f}%"),
    ("⚖️", "pH", f"{ph:.1f}"),
    ("🌧️", "RAIN", f"{rainfall:.0f}mm"),
]
tile_html = '<div class="tile-row">' + "".join(
    f'<div class="tile"><div class="tile-icon">{icon}</div>'
    f'<div class="tile-label">{label}</div>'
    f'<div class="tile-value">{value}</div></div>'
    for icon, label, value in tiles
) + '</div>'
st.markdown(tile_html, unsafe_allow_html=True)

predict_clicked = st.button("▶  RUN ANALYSIS", disabled=model is None)

# -----------------------------
# PREDICTION
# -----------------------------
if predict_clicked and model is not None:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    with st.spinner("Running inference..."):
        prediction = model.predict(input_data)
        probs = model.predict_proba(input_data)[0]
        crop_name = le.inverse_transform(prediction)[0]

    emoji, desc = crop_display(crop_name)
    top_conf = float(probs.max()) * 100

    # ---- Result readout + gauge, side by side ----
    r1, r2 = st.columns([1.3, 1])

    with r1:
        st.markdown(f"""
        <div class="readout">
            <div class="readout-label">// RECOMMENDED CROP</div>
            <div style="text-align:center; font-size:2rem;">{emoji}</div>
            <div class="readout-crop">{crop_name.upper()}</div>
            <div class="readout-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top_conf,
            number={"suffix": "%", "font": {"color": "#2ecc8f", "family": "Share Tech Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5f8f75"},
                "bar": {"color": "#2ecc8f"},
                "bgcolor": "#0d1a13",
                "borderwidth": 1,
                "bordercolor": "#1f3d2c",
                "steps": [
                    {"range": [0, 40], "color": "#1a2b21"},
                    {"range": [40, 70], "color": "#1f4a34"},
                    {"range": [70, 100], "color": "#1f6b46"},
                ],
            },
        ))
        gauge.update_layout(
            height=230,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eafff5", family="Share Tech Mono"),
        )
        st.plotly_chart(gauge, use_container_width=True)

    # -----------------------------
    # TOP 3 CROPS
    # -----------------------------
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = le.classes_[top3_idx]
    top3_probs = probs[top3_idx]
    ranks = ["RANK 01", "RANK 02", "RANK 03"]

    st.markdown("##### 🏅 TOP 3 CANDIDATES")
    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        e, _ = crop_display(top3_crops[i])
        with col:
            st.markdown(f"""
            <div class="top-card">
                <div class="top-rank">{ranks[i]}</div>
                <div style="font-size:1.4rem;">{e}</div>
                <h3 style="text-transform:capitalize; color:#eafff5; margin:4px 0;">{top3_crops[i]}</h3>
                <p style="color:#2ecc8f; font-weight:600; font-family:'Share Tech Mono', monospace;">{top3_probs[i]*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------
    # GRAPH — top 5, horizontal
    # -----------------------------
    st.markdown("##### 📊 CONFIDENCE DISTRIBUTION — TOP 5")

    top5_idx = np.argsort(probs)[-5:]
    top5_crops = le.classes_[top5_idx]
    top5_probs = probs[top5_idx] * 100
    labels = [f"{crop_display(c)[0]} {c}" for c in top5_crops]

    fig = go.Figure(go.Bar(
        x=top5_probs,
        y=labels,
        orientation="h",
        marker=dict(color=top5_probs, colorscale="Greens"),
        text=[f"{v:.1f}%" for v in top5_probs],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Probability (%)",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eafff5", family="Share Tech Mono"),
        xaxis=dict(gridcolor="#1f3d2c"),
    )
    st.plotly_chart(fig, use_container_width=True)

elif predict_clicked and model is None:
    st.warning("Model isn't loaded — check the error message above.")
else:
    st.info("👆 Adjust sensor readings above, then hit **RUN ANALYSIS**.")
