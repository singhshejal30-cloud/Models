import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Fraud Shield AI",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------------------
# Background + Theme
# ------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)),
    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

html, body, div, span, p, label, input {
    color: white !important;
}

h1, h2, h3 {
    color: white !important;
    font-weight: 800 !important;
}

.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

.result-box {
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    color: white;
}
/* Transaction Amount input text black */
div[data-testid="stNumberInput"] input {
    color: black !important;
    font-weight: bold !important;
}

/* Also placeholder text black */
div[data-testid="stNumberInput"] input::placeholder {
    color: black !important;
    opacity: 1;
}            
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Cache Model (IMPORTANT for Deployment)
# ------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("fraud_rf.csv")

    X = df[["Amount", "Time", "LocationRisk"]]
    y = df["Fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy

model, accuracy = load_model()

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("💳 Fraud Shield AI")
st.subheader("Real-Time Transaction Fraud Detection System")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔎 Enter Transaction Details")

    amount = st.number_input("Transaction Amount", min_value=0.0)
    time = st.slider("Transaction Hour", 0, 23, 12)
    location_risk = st.slider("Location Risk (1-10)", 1, 10, 5)

    predict_btn = st.button("🚀 Predict Fraud")

with col2:
    st.markdown("### 📊 Model Info")
    st.write(f"Model Accuracy: **{accuracy:.2f}**")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Trees Used: 300")

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if predict_btn:
    new_data = np.array([[amount, time, location_risk]])
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)

    fraud_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.markdown(
            f'<div class="result-box" style="background:#ff0033;">⚠ FRAUD DETECTED<br>Risk: {fraud_prob:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background:#00cc66;">✅ TRANSACTION SAFE<br>Risk: {fraud_prob:.2f}%</div>',
            unsafe_allow_html=True
        )