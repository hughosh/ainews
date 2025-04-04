import os
import requests
import datetime
from typing import List, Dict, Any

def get_news_api_key() -> str:
    """
    Retrieves the NewsAPI key from the environment variable.
    """
    key = os.environ.get('NEWS_API_KEY')
    if not key:
        raise ValueError("Environment variable NEWS_API_KEY is not set.")
    return key

def fetch_top_headlines(sources: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetches top headlines from the specified sources published within the past `hours` hours.
    """
    api_key = get_news_api_key()
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "sources": ",".join(sources),
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error making request: {e}")
    
    if response.status_code == 429:
        raise RuntimeError("API rate limit exceeded. Please try again later.")
    elif response.status_code != 200:
        raise RuntimeError(f"Error fetching news: HTTP {response.status_code}")
    
    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Error in API response: {data.get('message', 'Unknown error')}")
    
    articles = data.get("articles", [])
    
    now = datetime.datetime.utcnow()
    cutoff_time = now - datetime.timedelta(hours=hours)
    
    filtered_articles = []
    for article in articles:
        published_at_str = article.get("publishedAt", "")
        try:
            published_at = datetime.datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        
        if published_at >= cutoff_time:
            filtered_articles.append(article)
    
    return filtered_articles

def print_articles(articles: List[Dict[str, Any]]) -> None:
    """
    Prints articles with enriched metadata if available.
    """
    if not articles:
        print("No articles found for the specified criteria.")
        return
    
    for article in articles:
        title = article.get("title", "No Title")
        # Try to get source from two possible locations.
        source = (article.get("source", {}).get("name") if isinstance(article.get("source"), dict)
                  else article.get("source", "Unknown Source"))
        # Use 'timestamp' if available; otherwise fall back to 'publishedAt'
        timestamp = article.get("timestamp", article.get("publishedAt", ""))
        topic = article.get("topic", "")
        keywords = article.get("keywords", [])
        description = article.get("description", "")
        
        print(f"Title: {title}")
        print(f"Source: {source}")
        print(f"Timestamp: {timestamp}")
        if topic:
            print(f"Topic: {topic}")
        if keywords:
            print(f"Keywords: {', '.join(keywords)}")
        if description:
            print(f"Description: {description}")
        print("-" * 80)

