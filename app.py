import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords

# ----------------------------
# LOAD MODELS
# ----------------------------
model = pickle.load(open(r"D:\6TH SEM\PROJECTS\TEXT MINING\news-bias-detection\models\bias_model.pkl", "rb"))
vectorizer = pickle.load(open(r"D:\6TH SEM\PROJECTS\TEXT MINING\news-bias-detection\models\tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open(r"D:\6TH SEM\PROJECTS\TEXT MINING\news-bias-detection\models\label_encoder.pkl", "rb"))

# ----------------------------
# TEXT PREPROCESSING
# ----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_bias(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

# ----------------------------
# LOAD LIVE DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\6TH SEM\PROJECTS\TEXT MINING\news-bias-detection\data\datalive_cleaned_data.csv")

df = load_data()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("📰 News Bias Dashboard")

option = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Live News", "🔍 Analyze Article"]
)

# ----------------------------
# HOME PAGE
# ----------------------------
if option == "🏠 Home":
    st.title("🧠 Automated News Bias Detection")
    st.write("Analyze political bias of news articles in real-time.")

    # Bias Weather Report
    if 'bias' in df.columns:
        st.subheader("🌦 Bias Weather Report")

        bias_counts = df['bias'].value_counts()

        st.bar_chart(bias_counts)

        st.write("### Summary:")
        st.write(bias_counts)

# ----------------------------
# LIVE NEWS PAGE
# ----------------------------
elif option == "📊 Live News":
    st.title("📊 Live News Feed")

    st.write("Explore today's news articles")

    # Dropdown to select article
    article_titles = df['title'].tolist()

    selected_title = st.selectbox("Select an Article", article_titles)

    selected_row = df[df['title'] == selected_title].iloc[0]

    st.subheader(selected_row['title'])
    st.write(f"**Source:** {selected_row['site']}")
    st.write(f"**Date:** {selected_row['date']}")
    st.write(f"**Actual Bias:** {selected_row['bias']}")

    # Predict bias
    predicted = predict_bias(selected_row['page_text'])

    st.write(f"### 🤖 Predicted Bias: {predicted}")

    # Show article text
    with st.expander("Read Full Article"):
        st.write(selected_row['page_text'])

    # Show full dataframe (optional)
    st.subheader("📄 Full Dataset")
    st.dataframe(df.head(50))

# ----------------------------
# ANALYZE CUSTOM ARTICLE
# ----------------------------
elif option == "🔍 Analyze Article":
    st.title("🔍 Analyze Your Own Article")

    user_input = st.text_area("Paste article text here")

    if st.button("Predict Bias"):
        if user_input.strip() != "":
            result = predict_bias(user_input)
            st.success(f"Predicted Bias: {result}")
        else:
            st.warning("Please enter some text")