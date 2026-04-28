import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Truth-Lens AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 IMPORTANT: Your Render backend endpoint
BACKEND_URL = "https://fake-news-detector-pjm4.onrender.com/detect/"

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Truth-Lens AI")
    st.info("Fake News Detector using ML model deployed on cloud.")
    st.markdown("---")
    st.write("### How it works:")
    st.write("1. FastAPI Backend (Render)")
    st.write("2. Streamlit Frontend")
    st.write("3. ML Model (Logistic Regression)")
    st.markdown("---")

# --- Main UI ---
st.title("📰 Fake News Detection Engine")
st.write("Enter news text below to check authenticity.")

# User input
user_input = st.text_area(
    "Enter News Text Here:",
    "A man from Florida claims he was abducted by aliens.",
    height=250
)

# --- Button Logic ---
if st.button("Analyze Authenticity", type="primary", use_container_width=True):

    if user_input.strip():

        with st.spinner("AI is analyzing..."):
            try:
                payload = {"text": user_input}
                response = requests.post(BACKEND_URL, json=payload, timeout=15)
            except requests.exceptions.RequestException:
                st.error("⚠️ Cannot connect to backend. Check Render URL.")
                response = None

        # --- Handle Response ---
        if response and response.status_code == 200:
            result = response.json()

            label = result.get("prediction", "UNKNOWN").upper()
            score = result.get("confidence", 0)

            st.balloons()

            if "REAL" in label:
                st.success(f"✅ **REAL NEWS** (Confidence: {score:.1%})")
                st.progress(score)
            else:
                st.error(f"🚨 **FAKE NEWS** (Confidence: {score:.1%})")
                st.progress(score)

            with st.expander("🔍 More Details"):
                st.json(result)

        else:
            st.error("❌ Backend error or invalid response")

    else:
        st.warning("⚠️ Please enter some text.")