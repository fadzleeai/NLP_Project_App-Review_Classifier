import streamlit as st
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = PROJECT_DIR / "data" / "37000_reviews_with_sentiment.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["review_description"] = df["review_description"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("").astype(str)
    return df

df = load_data()

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

        model_lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")

        status_box_lr.success("Loaded Logistic Regression model successfully ✅")

    except Exception as e:
        status_box_lr.error(f"Failed to load Logistic Regressuin model: {e}")

    status_box_nb = st.empty()
    try:
        status_box_nb.info("Loading Naive Bayes Model...")

        model_nb = joblib.load(MODEL_DIR / "naive_bayes.pkl")

        status_box_nb.success("Loaded Naive Bayes model successfully ✅")

    except Exception as e:
        status_box_nb.error(f"Failed to load Naive Bayes model: {e}")

    status_box_le = st.empty()
    try:
        status_box_le.info("Loading Label Encoder...")

        le = joblib.load(MODEL_DIR / "label_encoder.pkl")

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
    st.subheader("Dataset Visualizations")

    st.write("### 1. Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.write("### 2. Rating Distribution")
    rating_counts = df["rating"].value_counts().sort_index()
    st.bar_chart(rating_counts)

    st.write("### 3. Review Source Distribution")
    source_counts = df["source"].value_counts()
    st.bar_chart(source_counts)

    st.write("### 4. Sentiment Trend Over Time")

    df_trend = df.copy()
    df_trend["review_date"] = pd.to_datetime(df_trend["review_date"], errors="coerce")
    df_trend = df_trend.dropna(subset=["review_date"])

    if len(df_trend) > 0:
        monthly_sentiment = (
            df_trend
            .groupby([pd.Grouper(key="review_date", freq="ME"), "sentiment"])
            .size()
            .unstack(fill_value=0)
        )

        monthly_sentiment.index = monthly_sentiment.index.strftime("%Y-%m")
        st.line_chart(monthly_sentiment)
    else:
        st.warning("No valid review dates found for trend visualization.")

    st.write("### 5. Top 20 Common Words in Reviews")

    try:
        vectorizer = CountVectorizer(stop_words="english", max_features=20)
        word_matrix = vectorizer.fit_transform(df["review_description"])

        word_counts = word_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()

        top_words_df = pd.DataFrame({
            "word": words,
            "count": word_counts
        }).sort_values(by="count", ascending=False)

        st.bar_chart(top_words_df.set_index("word"))
    except Exception as e:
        st.error(f"Failed to generate top words chart: {e}")


with tab3:
    st.subheader("Model Comparison")

    st.write("""
    This section compares two machine learning models trained for app review sentiment classification:
    Logistic Regression and Naive Bayes.
    """)

    X_all = df["review_description"]
    y_all = le.transform(df["sentiment"])

    def evaluate_model(model, model_name):
        y_pred = model.predict(X_all)

        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_all, y_pred),
            "Precision": precision_score(y_all, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_all, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_all, y_pred, average="weighted", zero_division=0)
        }

    try:
        results = [
            evaluate_model(model_lr, "Logistic Regression"),
            evaluate_model(model_nb, "Naive Bayes")
        ]

        results_df = pd.DataFrame(results)

        st.write("### Evaluation Metrics")
        st.dataframe(results_df)

        st.write("### Model Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model")[["Accuracy"]])

        st.write("### Model F1-score Comparison")
        st.bar_chart(results_df.set_index("Model")[["F1-score"]])

        st.write("### Confusion Matrix")

        cm_model_choice = st.selectbox(
            "Choose model for confusion matrix",
            ["Logistic Regression", "Naive Bayes"],
            key="cm_model_choice"
        )

        selected_model = model_lr if cm_model_choice == "Logistic Regression" else model_nb
        y_pred_selected = selected_model.predict(X_all)

        cm = confusion_matrix(y_all, y_pred_selected)
        cm_df = pd.DataFrame(
            cm,
            index=le.classes_,
            columns=le.classes_
        )

        st.dataframe(cm_df)

    except Exception as e:
        st.error(f"Failed to compare models: {e}")


with tab4:
    st.subheader("About This Project")

    st.write("""
    This project is a Natural Language Processing application for app review sentiment analysis.
    The system classifies user reviews into three sentiment categories:
    negative, neutral, and positive.
    """)

    st.write("### Project Theme")
    st.write("Theme 6: App Review Classifier")

    st.write("### Dataset")
    st.write("""
    The dataset contains mobile app reviews with review text, rating, review date, source, and sentiment labels.
    It is used to train and evaluate sentiment classification models.
    """)

    st.write("### NLP Pipeline")
    st.write("""
    1. Load app review dataset  
    2. Clean and prepare review text  
    3. Convert text into numerical features using TF-IDF  
    4. Train machine learning models  
    5. Compare model performance  
    6. Use the selected model for sentiment prediction  
    """)

    st.write("### Models Used")
    st.write("""
    - Logistic Regression  
    - Naive Bayes  
    """)

    st.write("### Checkpoint 2 Features")
    st.write("""
    - Text input and sentiment prediction  
    - Confidence score display  
    - Dataset visualizations  
    - Two-model comparison  
    - Basic Streamlit prototype ready for lab demo  
    """)