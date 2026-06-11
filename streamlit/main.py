import streamlit as st

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
    pass  # your prediction code here

with tab2:
    pass  # your visualization code here

with tab3:
    pass  # your model comparison code here

with tab4:
    pass  # your about section here