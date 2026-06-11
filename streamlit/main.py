import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


st.set_page_config(
    page_title="App Review Classifier",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
  /* Black background */
  .stApp { background-color: #000000; }
  section[data-testid="stSidebar"] { background-color: #0a0a0a; }

  /* Text */
  html, body, [class*="css"], p, div, span, label { color: #e5e5e5; }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
      background: #000;
      border-bottom: 1px solid #2a2a2a;
      gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      color: #666;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 12px 28px;
      border: none;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
  }
  .stTabs [aria-selected="true"] {
      color: #fff !important;
      border-bottom: 2px solid #fff !important;
      background: transparent !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #ccc !important; }

  /* Tab content area */
  .stTabs [data-baseweb="tab-panel"] {
      background: #000;
      padding-top: 32px;
  }

  /* Inputs, textareas, selects */
  .stTextInput input,
  .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] {
      background: #111 !important;
      border: 1px solid #333 !important;
      color: #e5e5e5 !important;
      border-radius: 8px !important;
  }

  /* Buttons */
  .stButton button {
      background: #fff;
      color: #000;
      border: none;
      font-weight: 600;
      border-radius: 8px;
  }
  .stButton button:hover { background: #ddd; }

  /* Dividers */
  hr { border-color: #222 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #000; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Header
st.markdown("""
<h1 style="color:#fff;font-size:1.8rem;font-weight:700;margin-bottom:4px;">
  Thread App Review Sentiment Analysis
</h1>
<p style="color:#555;font-size:0.9rem;margin-bottom:32px;">
  An AI that able to do sentiment analysis using Thread app reviews!
</p>
""", unsafe_allow_html=True)

# ── Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict",
    "📊  Visualize",
    "⚖️  Model Comparison",
    "👋  About",
])

with tab1:
    st.subheader("🧠 Model Status")
    #Load Model
    status_box_lr = st.empty()
    try:
        status_box_lr.info("Loading Logistic Regression model...")

        model_lr = joblib.load("models/logistic_regression.pkl")

        status_box_lr.success("Loaded Logistic Regression model successfully ✅")

    except Exception as e:
        status_box_lr.error(f"Failed to load Logistic Regressuin model: {e}")

    status_box_nb = st.empty()
    try:
        status_box_nb.info("Loading Naive Bayes Model...")

        model_nb = joblib.load("models/naive_bayes.pkl")

        status_box_nb.success("Loaded Naive Bayes model successfully ✅")

    except Exception as e:
        status_box_nb.error(f"Failed to load Naive Bayes model: {e}")

    status_box_le = st.empty()
    try:
        status_box_le.info("Loading Label Encoder...")

        le = joblib.load("models/label_encoder.pkl")

        status_box_le.success("Loaded label encoder successfully ✅")

    except Exception as e:
        status_box_nb.error(f"Failed to label encoder: {e}")

        
    #Body
    model_choice = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Naive Bayes"]
    )
    
    models = {
        "Logistic Regression": model_lr,
        "Naive Bayes": model_nb
    }

    model = models[model_choice]
    
    text = st.text_input("Enter review")

    if st.button("Predict"):
        conf = model.predict_proba([text])[0]
        sentiment = le.inverse_transform(model.predict([text]))[0]
        st.write(f"Sentiment: {sentiment}")
        
        confidence_neg = conf[0]
        confidence_neu = conf[1]
        confidence_pos = conf[2]

        st.write("### Negative")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.progress(confidence_neg)

        with col2:
            st.write(f"{confidence_neg*100:.1f}%")

        st.write("### Neutral")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(confidence_neu)
        with col2:
            st.write(f"{confidence_neu*100:.1f}%")

        st.write("### Positive")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(confidence_pos)
        with col2:
            st.write(f"{confidence_pos*100:.1f}%")

    

with tab2:
    pass  # your visualization code here

with tab3:
    pass  # your model comparison code here

with tab4:
    pass  # your about section here