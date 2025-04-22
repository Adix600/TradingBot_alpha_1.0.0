import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def load_finnbert_pipeline():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def analyze_sentiment(texts, pipe=None):
    if pipe is None:
        pipe = load_finnbert_pipeline()
    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    results = []
    for text in texts:
        try:
            result = pipe(text[:512])[0]
            results.append(sentiment_map.get(result['label'].lower(), 0))
        except Exception as e:
            print(f"[Błąd sentymentu] {e}")
            results.append(0)
    return results

def scrape_latest_news():
    url = "https://www.forexlive.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return [item.get_text(strip=True) for item in soup.select(".article__title")[:10]]
    except Exception as e:
        print(f"[Błąd scrapowania newsów] {e}")
        return []
