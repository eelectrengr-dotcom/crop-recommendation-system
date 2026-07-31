import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = "model.pkl"
le_path = "label_encoder.pkl"

model = pickle.load(open(model_path, "rb"))
le = pickle.load(open(le_path, "rb"))

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Crop AI", layout="wide")

# -----------------------------
# DARK UI CSS
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Cards */
.card {
    background: #161b22;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.6);
}

/* Titles */
.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 15px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c853, #00e676);
    color: white;
    font-size: 20px;
    border-radius: 10px;
    padding: 12px 30px;
    border: none;
}

/* Result Box */
.result-box {
    text-align: center;
    padding: 30px;
    background: #161b22;
    border-radius: 15px;
    margin-top: 20px;
}

/* Top 3 Cards */
.top-card {
    background: #161b22;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h2 style='text-align:center;'>🌱 Crop Recommendation AI</h2>", unsafe_allow_html=True)

# -----------------------------
# INPUT LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 Soil Nutrients</div>', unsafe_allow_html=True)

    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 50)
    K = st.slider("Potassium (K)", 0, 140, 50)

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌦 Weather Conditions</div>', unsafe_allow_html=True)

    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# BUTTON
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🌾 Recommend Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Prediction
    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]

    crop_name = le.inverse_transform(prediction)[0]

    # -----------------------------
    # BIG RESULT
    # -----------------------------
    st.markdown(f"""
    <div class="result-box">
        <h1 style="color:#00ffcc; font-size:60px;">🌾 {crop_name.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # TOP 3 CROPS
    # -----------------------------
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = le.classes_[top3_idx]
    top3_probs = probs[top3_idx]

    st.markdown("### 🥇 Top 3 Recommendations")

    c1, c2, c3 = st.columns(3)

    for i, col in enumerate([c1, c2, c3]):
        with col:
            st.markdown(f"""
            <div class="top-card">
                <h3>{top3_crops[i]}</h3>
                <p>{top3_probs[i]*100:.2f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------
    # GRAPH
    # -----------------------------
    st.markdown("### 📊 Prediction Confidence Graph")

    fig, ax = plt.subplots()
    ax.bar(le.classes_, probs)
    plt.xticks(rotation=90)
    plt.ylabel("Probability")

    st.pyplot(fig)
