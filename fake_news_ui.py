import streamlit as st
import pickle
import time
import pandas as pd

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit settings
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --------------------- DARK MODE TOGGLE ---------------------
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")

if dark_mode:
    bg_color = "#0f0f0f"
    text_color = "#ffffff"
    card_color = "#1c1c1c"
else:
    bg_color = "#eef2f7"
    text_color = "#000000"
    card_color = "#ffffff"

# --------------------- CUSTOM CSS ---------------------
st.markdown(f"""
<style>

body {{
    background-color: {bg_color};
    color: {text_color};
}}

.scroll-container {{
    max-height: 85vh;
    overflow-y: auto;
    padding-right: 15px;
}}

.title {{
    text-align: center;
    background: linear-gradient(90deg, #0061ff, #60efff);
    -webkit-background-clip: text;
    color: transparent;
    font-size: 50px;
    font-weight: 800;
    margin-top: 10px;
}}

.subtitle {{
    text-align: center;
    font-size: 20px;
    color: {'white' if dark_mode else 'gray'};
    opacity: 0.8;
    margin-bottom: 25px;
}}

.card {{
    background: {card_color};
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.20);
    transition: 0.3s;
    margin-bottom: 30px;
}}

.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.35);
}}

.footer {{
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 14px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------- SIDEBAR STATS ---------------------
st.sidebar.header("📊 App Statistics")

st.sidebar.write("📰 **Model:** Passive Aggressive Classifier")  
st.sidebar.write("📄 **Vectorizer:** TF-IDF")  
st.sidebar.write("📦 **Features:** 22 (TF-IDF top features)")  
st.sidebar.write("🤖 **Prediction:** Fake / Real")  

st.sidebar.markdown("---")
st.sidebar.write("Upload news file (.txt or .csv) for **bulk detection** below:")

uploaded_file = st.sidebar.file_uploader("Choose file", type=["txt", "csv"])

# --------------------- MAIN UI ---------------------
st.markdown("<div class='title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered system to detect Fake & Real News using TF-IDF + ML Model</div>", unsafe_allow_html=True)

st.markdown("<div class='scroll-container'>", unsafe_allow_html=True)

# --------------------- BULK FILE PROCESSING ---------------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📁 Bulk File News Detection")

    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
        data = [text]

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            data = df["text"].astype(str).tolist()

    # Loader animation
    with st.spinner("🔍 Analyzing all news... Please wait..."):
        time.sleep(2)

        transformed = vectorizer.transform(data)
        predictions = model.predict(transformed)

        result_df = pd.DataFrame({"News": data, "Prediction": ["Fake" if p == 1 else "Real" for p in predictions]})

    st.success("✅ Bulk analysis completed!")
    st.dataframe(result_df, height=300)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------- SINGLE NEWS INPUT ---------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("✏️ Enter News Content")
news_text = st.text_area("Paste news content here:", height=250)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------- PREDICTION ---------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("📌 Prediction Result")

if st.button("🔎 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠ Please enter some text before predicting.")
    else:
        with st.spinner("🧠 AI thinking... analyzing text..."):
            time.sleep(1.8)

            transformed = vectorizer.transform([news_text])
            prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 Fake News Detected!")
        else:
            st.success("✅ This news appears to be Real!")
else:
    st.info("Click the button to analyze the news.")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------- FOOTER ---------------------
st.markdown("<div class='footer'>Developed by Sameer Tadvi | Fake News Detection Project</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
