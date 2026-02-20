import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# Advanced Spam-Themed Background
# -------------------------------
background_url = "https://images.unsplash.com/photo-1550751827-4bd374c3f58b"  # Dark cyber tech background

st.markdown(f"""
<style>

/* Main App Background */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.9)),
                url('{background_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Glowing animated overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 20% 30%, rgba(255,0,0,0.08), transparent 40%),
                radial-gradient(circle at 80% 70%, rgba(0,255,204,0.08), transparent 40%);
    animation: glowMove 8s infinite alternate;
    pointer-events: none;
}}

@keyframes glowMove {{
    0% {{ transform: scale(1); }}
    100% {{ transform: scale(1.05); }}
}}

/* Glass Card Effect */
.main {{
    background-color: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,0,0,0.5);
    box-shadow: 0 0 30px rgba(255,0,0,0.3);
}}

/* Titles */
h1 {{
    text-align: center;
    color: #ff4d4d !important;
    text-shadow: 0 0 15px red;
}}

/* Text Styling */
h2, h3, label, .stMarkdown {{
    color: #00ffcc !important;
}}

/* Textarea Styling */
textarea {{
    border-radius: 12px !important;
    background-color: rgba(0,0,0,0.7);
    color: #00ffcc;
    border: 1px solid rgba(255,0,0,0.6);
}}

/* Button Styling */
.stButton>button {{
    background: linear-gradient(45deg, #ff0000, #ff6600);
    color: white;
    border-radius: 12px;
    padding: 10px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}}

.stButton>button:hover {{
    transform: scale(1.07);
    box-shadow: 0 0 25px red;
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.8);
}}

/* Footer */
.footer {{
    text-align: center;
    color: #ff4d4d;
    font-style: italic;
}}

</style>
""", unsafe_allow_html=True) 

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("📌 About App")
    st.write("""
    This app uses **NLP & Machine Learning**  
    to classify emails as **Spam** or **Ham**.
    
    **Model Used:**
    - TF-IDF Vectorizer
    - Naive Bayes Classifier
    """)
    st.info("Made with ❤️ using Streamlit")

# -------------------------------
# Title
# -------------------------------
st.title("📧 Email Spam Detection")
st.write("Paste an email message below and instantly check if it's **Spam** or **Not Spam**.")

# -------------------------------
# Dataset
# -------------------------------
data = {
    "text": [
        "You won a lottery!",
        "Claim prize by clicking!",
        "Click on amount to redeem reward!",
        "Limited offer for you account!",
        "Addhar OTP",
        "MOdule Assessment",
        "Your ticket is confirmed with IRCTC",
        "Registration number for SSC Exams.",
        "Credit card offer for low interest.",
        "Win cash prize!",
        "ES Class is scheduled on monday"
    ],
    "label": ["spam","spam","spam","spam","ham","ham","ham","ham","spam","spam","ham"]
}
df = pd.DataFrame(data)

# -------------------------------
# Train Model
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB(alpha=0.1))
])
model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
email_text = st.text_area(
    "✉️ Enter Email Text:",
    height=160,
    placeholder="Type or paste email content here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        with st.spinner("Analyzing email..."):
            prediction = model.predict([email_text])[0]
            probability = model.predict_proba([email_text])[0]

        spam_index = list(model.classes_).index("spam")
        ham_index = list(model.classes_).index("ham")

        if prediction == "spam":
            st.error("🚨 SPAM EMAIL DETECTED")
            st.progress(int(probability[spam_index] * 100))
            st.write(f"**Spam Confidence:** {round(probability[spam_index]*100,2)}%")
        else:
            st.success("✅ SAFE EMAIL (HAM)")
            st.progress(int(probability[ham_index] * 100))
            st.write(f"**Ham Confidence:** {round(probability[ham_index]*100,2)}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 Built using NLP, TF-IDF & Naive Bayes</div>", unsafe_allow_html=True)
