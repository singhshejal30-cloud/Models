import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- REAL HOUSE BACKGROUND ----------------
background_image = "https://images.unsplash.com/photo-1568605114967-8130f3a36994"

st.markdown(f"""
<style>
.stApp {{
    background-image: url("{background_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.75);
    z-index: -1;
}}

.card {{
    background: rgba(255, 255, 255, 0.08);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
}}

h2, h3 {{
    color: white !important;
}}

label {{
    color: #f5f5f5 !important;
    font-weight: bold;
}}

.stButton>button {{
    background: linear-gradient(45deg, #ff9933, #ffffff, #138808);
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    width: 100%;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:white;'>
🏠 Indian House Price Prediction System 🇮🇳
</h1>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- DATA ----------------
data = {
    "Size_sqft": [800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900],
    "Bedrooms": [2,2,3,3,3,4,4,4,5,5,5,6],
    "Age_years": [10,8,7,5,6,4,3,2,2,1,1,1],
    "Price": [30000,35000,42000,46000,48000,54000,60000,65000,70000,76000,82000,88000]
}

df = pd.DataFrame(data)

# ---------------- MODEL TRAINING ----------------
X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Dashboard")
st.sidebar.info("Algorithm: Linear Regression")
st.sidebar.success("Model Ready ✅")

# ---------------- MAIN CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<h2>🏡 Enter Property Details</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    size = st.number_input("📏 Size (sq ft)", 800, 2500, 1200)

with col2:
    bedrooms = st.number_input("🛏 Bedrooms", 1, 6, 3)

with col3:
    age = st.number_input("🏗 Property Age (years)", 0, 20, 5)

if st.button("🔮 Predict Price in ₹"):
    new_data = np.array([[size, bedrooms, age]])
    prediction = model.predict(new_data)

    st.markdown(f"""
        <h2 style='text-align:center; color:#00ffcc;'>
        💰 Estimated Price: ₹ {prediction[0]:,.0f}
        </h2>
    """, unsafe_allow_html=True)

# ---------------- HIDE / SHOW DATASET ----------------
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📋 Dataset Used for Model Training (Click to View)"):
    st.dataframe(df)

st.markdown("</div>", unsafe_allow_html=True)



