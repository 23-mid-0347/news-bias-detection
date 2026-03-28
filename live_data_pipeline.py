import pandas as pd
from newsapi import NewsApiClient
from newspaper import Article
import re
import nltk
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv

nltk.download('stopwords')

# API SETUP
load_dotenv()

api_key = os.getenv("NEWS_API_KEY")

if not api_key:
    raise ValueError("NEWS_API_KEY not found. Check your .env file")

newsapi = NewsApiClient(api_key=api_key)

# BIAS MAPPING (FROM DATASET)
BIAS_MAPPING = {
    'Fox News': 'right',
    'MSNBC': 'left',
    'BBC News': 'center',
    'Associated Press': 'center',
    'NPR': 'left'
}

# CLEANING FUNCTION
# MUST MATCH clean_dataset.ipynb
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# FETCH + SCRAPE
def fetch_live_news(query='politics', limit=20):
    response = newsapi.get_everything(q=query, language='en')
    data = []

    for article_meta in response['articles'][:limit]:
        url = article_meta['url']
        bad_domains = ["consent.yahoo.com", "accounts.google.com"]

        if any(bad in url for bad in bad_domains):
            continue

        try:
            article = Article(url)
            article.download()
            article.parse()

            text = article.text

            # IMPORTANT FILTER
            if not text or len(text) < 300:
                continue

            row = {
                'url': url,
                'title': article_meta['title'],
                'date': article_meta['publishedAt'][:10],
                'site': article_meta['source']['name'],
                'bias': BIAS_MAPPING.get(article_meta['source']['name'], 'unknown'),
                'page_text': text
            }

            data.append(row)

        except Exception as e:
            print(f"Skipped: {url} | Reason: {e}")

    return pd.DataFrame(data)


# FINAL PIPELINE FUNCTION

def generate_model_input():
    df = fetch_live_news()

    if df.empty:
        print("No usable articles found")
        return None

    # Apply SAME preprocessing
    df['clean_text'] = df['page_text'].apply(clean_text)

    # THIS is what you pass to ML team
    return df


# RUN

if __name__ == "__main__":
    df_ready = generate_model_input()


    if df_ready is not None:

        
        # 1. REMOVE DUPLICATES 
        
        df_ready.drop_duplicates(subset=['url'], inplace=True)

        
        # 2. FILE PATHS
        
        base_path = r"D:\Coding\6th sem\text mining\News_bias_detection\data"

        today_file = os.path.join(base_path, "datalive_cleaned_data.csv")
        history_file = os.path.join(base_path, "bias_history.csv")

        
        # 3. SAVE TODAY'S SNAPSHOT (OVERWRITE)
        
        df_ready.to_csv(today_file, index=False)

        
        # 4. SAVE HISTORY (APPEND MODE)
        
        if not os.path.exists(history_file):
            # First time → create file with header
            df_ready.to_csv(history_file, index=False)
        else:
            # Append without header
            df_ready.to_csv(history_file, mode='a', header=False, index=False)

        
        # 5. CONFIRMATION
        
        print(f"Saved today's data → {today_file}")
        print(f"Updated history → {history_file}")