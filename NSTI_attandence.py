import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="NSTI Student Performance", layout="wide")

# -------------------------------
# Background + UI Styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #e0f7fa, #ffffff);
}

.card {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title("🎓 NSTI Student Performance Prediction App")
st.write("📘 *Data Mining & Machine Learning using Decision Tree*")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("student_performance.csv")

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("📊 Key Insights")

total_students = len(df)
pass_count = df[df["Result"] == "Pass"].shape[0]
fail_count = df[df["Result"] == "Fail"].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("👨‍🎓 Total Students", total_students)
col2.metric("✅ Pass Students", pass_count)
col3.metric("❌ Fail Students", fail_count)

# -------------------------------
# Dataset Preview
# -------------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df)

# -------------------------------
# Pattern Finding
# -------------------------------
st.subheader("📈 Performance Pattern (Average)")
grouped_data = df.groupby("Result").mean()
st.dataframe(grouped_data)

# -------------------------------
# Prepare Data
# -------------------------------
X = df[["Attendance", "StudyHours", "PreviousMarks"]]
y = df["Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Model Performance
# -------------------------------
st.subheader("🧠 Model Performance")
st.success(f"🎯 Accuracy: {accuracy * 100:.2f} %")

with st.expander("📑 Classification Report"):
    st.text(classification_report(y_test, y_pred))

# -------------------------------
# Sidebar Prediction
# -------------------------------
st.sidebar.header("🧑‍🎓 Student Details")

attendance = st.sidebar.slider("📅 Attendance (%)", 0, 100, 75)
study_hours = st.sidebar.slider("📖 Study Hours / Day", 0, 10, 3)
previous_marks = st.sidebar.slider("📝 Previous Marks", 0, 100, 65)

new_student = [[attendance, study_hours, previous_marks]]

if st.sidebar.button("🔍 Predict Result"):
    prediction = model.predict(new_student)[0]

    if prediction == 1:
        st.balloons()
        st.success("🎉 **Prediction: Student will PASS!** Keep it up 👍")
    else:
        st.error("⚠️ **Prediction: Student may FAIL** — Needs Improvement 📚")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📉 Student Performance Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["Attendance"],
    df["PreviousMarks"],
    c=y_encoded,
    cmap="coolwarm",
    alpha=0.7
)
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Previous Marks")
ax.set_title("Attendance vs Previous Marks")

st.pyplot(fig)

