# ---------------- IMPORT LIBRARIES ----------------
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    layout="wide",
    page_icon="🏦"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("bank_churn_model.pkl","rb"))


st.markdown("""
<style>

/* App background with bank image */

.stApp{
background: linear-gradient(rgba(15,32,39,0.85),rgba(32,58,67,0.85),rgba(44,83,100,0.85)),
url("https://images.unsplash.com/photo-1460925895917-afdab827c52f");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

/* All headings white */

h1, h2, h3, h4, h5, h6 {
color:white !important;
font-weight:bold !important;
}

/* Paragraph text */

p, span, div {
color:white !important;
font-weight:bold !important;
}

/* Labels */

label{
color:white !important;
font-weight:bold !important;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background-color: rgba(0,0,0,0.4);
}

/* Metrics cards */

div[data-testid="metric-container"]{
background-color: rgba(255,255,255,0.1);
padding:15px;
border-radius:10px;
}

/* Metric text */

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"]{
color:white !important;
font-weight:bold !important;
}

/* ---------------- INPUT OPTIONS BLACK ---------------- */

/* Selectbox text */

.stSelectbox div{
color:black !important;
font-weight:bold;
}

/* Number input */

.stNumberInput input{
color:black !important;
font-weight:bold;
}

/* Text input */

.stTextInput input{
color:black !important;
font-weight:bold;
}

/* Dropdown expanded box background */
div[data-baseweb="select"] > div {
    background-color: white !important;
}

 div[data-baseweb="popover"] * {
    color: black !important;
} 
/* Buttons */

.stButton>button{
background-color:#00c6ff;
color:white;
font-weight:bold;
border-radius:10px;
}

</style>
""",unsafe_allow_html=True)


# ---------------- LOGIN SYSTEM ----------------

users = {
    "admin":"admin123",
    "student":"pass123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False



if not st.session_state.logged_in:

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.title("🏦 Bank Login🔐")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username] == password:

            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful")

        else:
            st.error("Invalid Credentials")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------- AFTER LOGIN ----------------

if st.session_state.logged_in:

    # ---------------- TITLE ----------------
    st.title("🏦Bank Churn Prediction Dashboard")

    st.write("Machine Learning system to predict whether a customer will leave the bank.")

    # ---------------- KPI METRICS ----------------
    st.subheader("📊 Bank Metrics")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Customers","10,000")
    col2.metric("High Risk Customers","2,350")
    col3.metric("Average Balance","₹75,000")
    col4.metric("Average Salary","₹65,000")

    st.divider()

    # ---------------- CUSTOMER INPUT ----------------
    st.subheader("🧾 Enter Customer Details")

    col1,col2,col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender",["Male","Female"])
        age = st.slider("Age",18,80,30)
        tenure = st.slider("Tenure",0,10,3)

    with col2:
        geography = st.selectbox("Country",["France","Spain","Germany"])
        balance = st.number_input("Balance",0,500000,50000)
        products = st.slider("Number of Products",1,4,2)

    with col3:
        credit = st.selectbox("Has Credit Card",[0,1])
        active = st.selectbox("Active Member",[0,1])
        salary = st.number_input("Estimated Salary",20000,200000,50000)

    predict = st.button("🚀 Predict Churn")

    # ---------------- PREDICTION ----------------
    if predict:

        avg_balance = balance/(tenure+1)
        prod_customer = products/(tenure+1)
        salary_ratio = salary/(balance+1)

        data = pd.DataFrame([{

            "Gender":gender,
            "Geography":geography,
            "Age":age,
            "Tenure":tenure,
            "Balance":balance,
            "NumOfProducts":products,
            "HasCrCard":credit,
            "IsActiveMember":active,
            "EstimatedSalary":salary,

            "AverageMonthlyBalance":avg_balance,
            "ProductsPerCustomer":prod_customer,
            "SalaryToBalanceRatio":salary_ratio

        }])

        prob = model.predict_proba(data)[0][1]

        # ---------------- GAUGE CHART ----------------
        fig = go.Figure(go.Indicator(

            mode="gauge+number",
            value=prob*100,
            title={"text":"Churn Probability (%)"},

            gauge={
                "axis":{"range":[0,100]},

                "bar":{"color":"red" if prob>0.5 else "green"},

                "steps":[
                    {"range":[0,40],"color":"lightgreen"},
                    {"range":[40,70],"color":"yellow"},
                    {"range":[70,100],"color":"red"}
                ]
            }

        ))

        st.plotly_chart(fig,use_container_width=True)

        # ---------------- RESULT ----------------
        if prob > 0.5:
            st.error("⚠️ Very High Risk Customer")

        elif prob > 0.3:
            st.warning("⚠️ Medium Risk Customer")

        else:
            st.success("✅ Customer likely to stay")

        # ---------------- AI EXPLAINABILITY ----------------
        st.subheader("🧠 AI Explanation")

        explanation = []

        if balance < 20000:
            explanation.append("Low account balance")

        if active == 0:
            explanation.append("Customer inactive")

        if products <= 1:
            explanation.append("Customer uses very few bank products")

        if tenure < 2:
            explanation.append("Customer recently joined bank")

        if explanation:
            for e in explanation:
                st.write("•",e)

        else:
            st.write("Customer profile looks stable")

        # ---------------- RECOMMENDATION ----------------
        st.subheader("🤖Recommendation")

        if prob > 0.5:

            st.write("""
• Offer loyalty rewards  
• Provide special loan offers  
• Improve customer engagement  
• Assign personal relationship manager
""")

        else:

            st.write("""
Customer engagement is good.  
Continue providing current banking services.
""")

        # ---------------- INTERACTIVE CHART ----------------
        st.subheader("📈 Customer Analytics")

        chart_data = pd.DataFrame({

            "Feature":["Balance","Salary","Age","Products"],
            "Value":[balance,salary,age,products]

        })

        fig2 = px.bar(chart_data,x="Feature",y="Value",color="Feature")

        st.plotly_chart(fig2,use_container_width=True)

else:

    st.info("Please login from sidebar to access the dashboard")


