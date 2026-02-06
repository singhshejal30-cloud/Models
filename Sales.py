import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide"
)

# ------------------ LIGHT THEME CSS ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #FDFBFB, #EBEDEE);
}
h1, h2, h3 {
    color: #2C3E50;
    text-align: center;
}
.metric-box {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
label {
    color: #34495E !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("📈 Sales Prediction using Linear Regression")
st.markdown(
    "<p style='text-align:center; color:#555;'>Predict future sales using data-driven insights</p>",
    unsafe_allow_html=True
)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("sales_1000_data.csv")

with st.expander("📂 View Dataset"):
    st.dataframe(df.head())

# ------------------ FEATURES & TARGET ------------------
X = df[["AdvertisingSpend", "StoreVisitors", "Discount"]]
y = df["Sales"]

# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)

# ------------------ METRICS ------------------
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-box">
        <h3>R² Score</h3>
        <h2>{r2*100:.2f}%</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
        <h3>MSE</h3>
        <h2>{mse:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
        <h3>MAE</h3>
        <h2>{mae:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

# ------------------ GRAPH ------------------
st.subheader("📉 Actual vs Predicted Sales")

fig, ax = plt.subplots()
ax.scatter(y_test, pred)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")
st.pyplot(fig)

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔮 Predict New Sales")

ad_spend = st.sidebar.number_input("📢 Advertising Spend", min_value=0)
visitors = st.sidebar.number_input("👥 Store Visitors", min_value=0)
discount = st.sidebar.number_input("💸 Discount", min_value=0)

if st.sidebar.button("🚀 Predict Sales"):
    new_data = [[ad_spend, visitors, discount]]
    prediction = model.predict(new_data)[0]

    st.subheader("📌 Prediction Result")

    if prediction > 0:
        st.success(f"✅ Expected Sales: ₹ {prediction:.2f}")
        st.balloons()
    else:
        st.error("❌ Sales may result in loss 📉")