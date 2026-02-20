import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="NSTI Smart Library AI", layout="wide")

# -------------------------------
# Background
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507842217343-583bb7270b66");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background: rgba(255,255,255,0.92);
        padding: 30px;
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Login System
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Library Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
        else:
            st.error("Invalid Credentials ❌")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("📚 NSTI Smart Library AI Dashboard")

# -------------------------------
# Upload or Load Default Data
# -------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("library_data_100.csv")

# Encode Target
le = LabelEncoder()
data["FrequentUser"] = le.fit_transform(data["FrequentUser"])

X = data[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = data["FrequentUser"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Selection
# -------------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Logistic Regression"]
)

if model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = LogisticRegression()

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# -------------------------------
# User Input
# -------------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Student Age", 10, 60, 20)
books = st.sidebar.slider("Books Issued", 0, 50, 5)
late = st.sidebar.slider("Late Returns", 0, 20, 1)
membership = st.sidebar.slider("Membership Years", 0, 10, 1)

# -------------------------------
# Prediction + Save History
# -------------------------------
if st.sidebar.button("Predict"):

    input_data = np.array([[age, books, late, membership]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Frequent User" if prediction == 1 else "Not Frequent"

    st.subheader("Prediction Result")
    st.success(result)
    st.info(f"Confidence: {probability*100:.2f}%")

    # Save to CSV
    history = pd.DataFrame({
        "Age": [age],
        "BooksIssued": [books],
        "LateReturns": [late],
        "MembershipYears": [membership],
        "Prediction": [result],
        "Confidence": [probability]
    })

    if os.path.exists("prediction_history.csv"):
        history.to_csv("prediction_history.csv", mode='a', header=False, index=False)
    else:
        history.to_csv("prediction_history.csv", index=False)

# -------------------------------
# Show Prediction History
# -------------------------------
st.subheader("📜 Prediction History")

if os.path.exists("prediction_history.csv"):
    history_data = pd.read_csv("prediction_history.csv")
    st.dataframe(history_data)

    st.download_button(
        label="📥 Download History",
        data=history_data.to_csv(index=False),
        file_name="library_prediction_history.csv",
        mime="text/csv"
    )
else:
    st.write("No history available yet.")

# -------------------------------
# Feature Importance (RF Only)
# -------------------------------
if model_choice == "Random Forest":
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_

    fig = plt.figure()
    plt.bar(X.columns, importances)
    plt.xticks(rotation=45)
    st.pyplot(fig)
