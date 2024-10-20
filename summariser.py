import time
import pandas as pd
import nltk
import json
import os
import datetime
import http.client, urllib.parse

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# To fetch the news directly from mediastack
def fetch_mediastack_news(uri='api.mediastack.com', persist=False, categories=[], keywords=[], search=[], countries=[], languages=[], key=os.getenv('MEDIASTACK_API_KEY')):
    with http.client.HTTPConnection(uri) as conn:
        params = urllib.parse.urlencode({
            'access_key': f'{key}',
            'categories': ','.join(categories),
            'sort': 'published_desc',
            'keywords': ','.join(keywords),
            'search': ','.join(search),
            'countries': ','.join(countries),
            'languages': ','.join(languages),
            'limit': 100,
        })
    
    conn.request('GET', '/v1/news?{}'.format(params))
    res = conn.getresponse()
    data = res.read()
    json_data = json.loads(data)

    # Format the current date and time as a string (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if persist:
        with open(f'data/{timestamp}_news.json', 'w', encoding='utf-8') as f:
            json.dump(json.loads(data), f, ensure_ascii=True, indent=4)

    return json_data

# Load samples of news to be used to fine tune the ML/AI pipelines
def load_mediastack_news(filename='news.json'):
    titles = []
    descriptions = []
    links = []
    sources = []

    with open(f'data/{filename}', 'r') as f:
        news_results = json.loads(f.read())
        news_data = news_results['data']
        print('[INFO] Loaded news')

        for item in news_data:
            titles.append(item['title'])
            descriptions.append(item['description'])
            links.append(item['url'])
            sources.append(item['source'])
    
    return pd.DataFrame({
        'title': titles,
        'description': descriptions,
        'link': links,
        'source': sources
    })

# Preprocess text function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    print('[INFO] Text has been preprocessed')

    return ' '.join(words)

# Load Hugging Face models
summarizer = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
sentiment_analyzer = pipeline("sentiment-analysis")

# Summarization function
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Sentiment analysis function
def perform_sentiment_analysis(text):
    return sentiment_analyzer(text)[0]


def main():
    # Scrape articles
    df = load_mediastack_news()

    # Preprocess text
    df['clean_description'] = df['description'].apply(preprocess_text)

    # Apply summarization and sentiment analysis
    df['summary'] = df['clean_description'].apply(summarize_text)
    df['sentiment'] = df['summary'].apply(lambda x: perform_sentiment_analysis(x)['label'])
    df['sentiment_score'] = df['summary'].apply(lambda x: perform_sentiment_analysis(x)['score'])

    # Display results
    print(df[['title', 'clean_description', 'summary', 'sentiment', 'sentiment_score']])

if __name__ == '__main__':
    main()
