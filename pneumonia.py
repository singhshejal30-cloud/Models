
# # ---------------- IMPORT LIBRARIES ----------------
# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # ---------------- LOAD MODEL ----------------
# model = tf.keras.models.load_model("pneumonia_model.h5")

# # ---------------- TITLE ----------------
# st.title("🫁 Pneumonia Detection using AI")
# st.write("Upload Chest X-ray Image to Predict Pneumonia")

# # ---------------- IMAGE UPLOAD ----------------
# uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:

#     # ---------------- LOAD IMAGE ----------------
#     img = Image.open(uploaded_file)

#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # ---------------- PREPROCESS IMAGE ----------------
#     img = img.convert("RGB")        # convert grayscale to RGB
#     img = img.resize((224,224))    # resize for model
    
#     img = np.array(img) / 255.0    # normalization
    
#     img = img.reshape(1,224,224,3)

#     # ---------------- PREDICTION ----------------
#     prediction = model.predict(img)

#     if prediction[0][0] > 0.5:
#         st.error("⚠️ Pneumonia Detected")
#     else:
#         st.success("✅ Normal")


# ---------------- IMPORT LIBRARIES ----------------
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("pneumonia_model.h5")

# ---------------- ADVANCED ANIMATED BACKGROUND ----------------
st.markdown("""
<style>

/* Main animated gradient background */

.stApp{
background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
background-size: 400% 400%;
animation: gradientBG 12s ease infinite;
}

/* Gradient Animation */

@keyframes gradientBG {
0% {background-position: 0% 50%;}
50% {background-position: 100% 50%;}
100% {background-position: 0% 50%;}
}

/* Headings */

h1, h2, h3, h4, h5, h6 {
color:white !important;
font-weight:bold;
}

/* Text */

p, label {
color:white !important;
font-size:17px;
}

/* Glassmorphism container */

[data-testid="stFileUploader"],
[data-testid="stSidebar"],
.stAlert {

background: rgba(255,255,255,0.1);
backdrop-filter: blur(10px);
border-radius:15px;
padding:10px;
}

/* Sidebar style */

[data-testid="stSidebar"]{
background: rgba(0,0,0,0.6);
}

/* Button styling */

.stButton>button{
background: linear-gradient(90deg,#00c6ff,#0072ff);
color:white;
border:none;
border-radius:10px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 AI Medical Assistant")

st.sidebar.info("""
This AI system analyzes chest X-ray images 
and predicts whether a patient has pneumonia 
or not using a deep learning CNN model.
""")

st.sidebar.write("Model Type: CNN")
st.sidebar.write("Input Size: 224 × 224")

# ---------------- TITLE ----------------
st.title("🫁 AI Pneumonia Detection System")
st.write("Upload a Chest X-ray Image to detect Pneumonia")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        img = Image.open(uploaded_file)

        st.image(img, caption="Uploaded X-ray Image", use_column_width=True)

    # ---------------- PREPROCESS IMAGE ----------------
        img = img.convert("RGB")
        img = img.resize((224,224))
        img = np.array(img)/255.0
        img = img.reshape(1,224,224,3)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img)

    probability = float(prediction[0][0])

    normal_prob = 1 - probability
    pneumonia_prob = probability

    with col2:

        st.subheader("🔎 Prediction Result")

        if pneumonia_prob > 0.5:
            st.error("⚠️ Pneumonia Detected")
        else:
            st.success("✅ Normal")

        st.write(f"**Pneumonia Probability:** {pneumonia_prob:.2f}")
        st.write(f"**Normal Probability:** {normal_prob:.2f}")

        # ---------------- CONFIDENCE SCORE ----------------
        confidence = max(pneumonia_prob, normal_prob)*100

        st.write(f"**Model Confidence:** {confidence:.2f}%")

        # ---------------- CHART ----------------
        fig = go.Figure(data=[
            go.Bar(
                x=["Normal","Pneumonia"],
                y=[normal_prob,pneumonia_prob]
            )
        ])

        fig.update_layout(
            title="Prediction Probability",
            xaxis_title="Class",
            yaxis_title="Probability"
        )

        st.plotly_chart(fig)

# ---------------- FOOTER ----------------
st.write("---")
st.write("Developed using AI, Deep Learning and Streamlit")