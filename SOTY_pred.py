import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="University AI Ultra Pro", layout="wide")

# -------------------------
# 🔐 LOGIN SYSTEM
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    st.title("🔐 NSTI AI Login")
    login()
    st.stop()

# -------------------------
# 🌙 THEME TOGGLE
# -------------------------
theme = st.sidebar.radio("Select Theme", ["Dark", "Light"])

if theme == "Dark":
    overlay = "rgba(0,0,0,0.65)"
else:
    overlay = "rgba(255,255,255,0.6)"

# -------------------------
# 🎨 PREMIUM BACKGROUND
# -------------------------
st.markdown(f"""
<style>
.stApp {{
    background-image: linear-gradient({overlay}, {overlay}),
    url("https://images.unsplash.com/photo-1541339907198-e08756dedf3f");
    background-size: cover;
    background-attachment: fixed;
}}

h1 {{
    text-align:center;
    font-size:45px;
    background: linear-gradient(to right,#00f2fe,#4facfe);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}}

.block-container {{
    background: rgba(255,255,255,0.1);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
}}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Student of the Year - Ultra Pro")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("my_data(2).csv")
X = df[["Marks", "Attendance", "Sports", "StudyHours"]]
y = df["SOTY"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# -------------------------
# 🧠 AUTO BEST MODEL
# -------------------------
best_model_name = None
best_score = 0

for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_model_name = name

st.sidebar.success(f"🏆 Best Model: {best_model_name}")
st.sidebar.info(f"Accuracy: {best_score:.2f}")

model = models[best_model_name]
model.fit(X_train, y_train)

# -------------------------
# 📊 NAVIGATION
# -------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Prediction Center", "Leaderboard", "About"]
)

# -------------------------
# 🔮 PREDICTION CENTER
# -------------------------
if menu == "Prediction Center":

    col1, col2 = st.columns(2)

    with col1:
        marks = st.slider("Marks", 0, 100, 80)
        attendance = st.slider("Attendance", 0, 100, 85)

    with col2:
        sports = st.slider("Sports Score", 0, 10, 7)
        study_hours = st.slider("Study Hours", 0, 12, 6)

    if st.button("🚀 Run AI Prediction"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        new_data = [[marks, attendance, sports, study_hours]]
        prediction = model.predict(new_data)
        prob = model.predict_proba(new_data)
        confidence = np.max(prob) * 100

        result = "Selected" if prediction[0] == 1 else "Not Selected"

        st.success(f"Result: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots()
        ax.imshow(cm)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')
        st.pyplot(fig)

        # Feature Importance (if available)
        if hasattr(model, "feature_importances_"):
            fig2, ax2 = plt.subplots()
            ax2.bar(X.columns, model.feature_importances_)
            ax2.set_title("Feature Importance")
            st.pyplot(fig2)

        # -------------------------
        # 📄 PDF REPORT DOWNLOAD
        # -------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200,10,"Student of the Year AI Report",ln=True)
        pdf.cell(200,10,f"Result: {result}",ln=True)
        pdf.cell(200,10,f"Confidence: {confidence:.2f}%",ln=True)
        pdf.output("report.pdf")

        with open("report.pdf","rb") as file:
            st.download_button(
                label="📥 Download Prediction Report",
                data=file,
                file_name="SOTY_Report.pdf",
                mime="application/pdf"
            )

# -------------------------
# 🏆 LEADERBOARD
# -------------------------
elif menu == "Leaderboard":
    st.subheader("🏆 Top 5 Students")
    top_students = df.sort_values(by="Marks", ascending=False).head(5)
    st.dataframe(top_students)

# -------------------------
# 📘 ABOUT
# -------------------------
else:
    st.write("""
    🎯 AI Powered Student of the Year Prediction System
    
    🔐 Login Protected
    🧠 Auto Best Model Selection
    📊 ML Analytics
    📄 PDF Report Generator
    
    Built by Shejal Singh
    """)