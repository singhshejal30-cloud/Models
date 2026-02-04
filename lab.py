import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Title
# -------------------------------
st.title("📚 Library Management System")
st.subheader("Frequent Library User Prediction")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("library_data_100.csv")
    return data

data = load_data()

st.write("### Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Encoding Target Column
# -------------------------------
le = LabelEncoder()
data["FrequentUser"] = le.fit_transform(data["FrequentUser"])
# Yes -> 1, No -> 0

# -------------------------------
# Features & Target
# -------------------------------
X = data[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = data["FrequentUser"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.number_input("Student Age", min_value=10, max_value=60, value=20)
books = st.sidebar.number_input("Books Issued", min_value=0, max_value=50, value=5)
late = st.sidebar.number_input("Late Returns", min_value=0, max_value=20, value=1)
membership = st.sidebar.number_input("Membership Years", min_value=0, max_value=10, value=1)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict"):
    input_data = [[age, books, late, membership]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediction: Frequent Library User")
    else:
        st.error("❌ Prediction: Not a Frequent Library User")

# -------------------------------
# Model Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)
st.write(f"### 📊 Model Accuracy: {accuracy * 100:.2f}%")