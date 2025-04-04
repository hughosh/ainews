import os
import requests
import datetime
from typing import List, Dict, Any

def get_news_api_key() -> str:
    """
    Retrieves the NewsAPI key from the environment variable.
    Raises an error if the key is not found.
    """
    key = os.environ.get('NEWS_API_KEY')
    if not key:
        raise ValueError("Environment variable NEWS_API_KEY is not set.")
    return key

def fetch_top_headlines(sources: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetches top headlines from the specified sources that were published within the past `hours` hours.
    
    Parameters:
      sources (List[str]): List of news source identifiers (e.g. "cnn", "fox-news", "the-new-york-times").
      hours (int): Timeframe in hours to filter articles (default is 24 hours).
    
    Returns:
      List[Dict[str, Any]]: A list of dictionaries, each representing an article with its title,
                            source, publication date, and description.
    
    Raises:
      RuntimeError: If there is an error in the API request or the API rate limit is exceeded.
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
            continue  # Skip articles with unexpected date format
        
        if published_at >= cutoff_time:
            filtered_articles.append({
                "title": article.get("title", "No Title"),
                "source": article.get("source", {}).get("name", "Unknown Source"),
                "publishedAt": published_at.strftime("%Y-%m-%d %H:%M:%S"),
                "description": article.get("description", "No Description")
            })
    
    return filtered_articles

def print_articles(articles: List[Dict[str, Any]]) -> None:
    """
    Prints the list of articles in a formatted manner.
    
    Parameters:
      articles (List[Dict[str, Any]]): The list of articles to print.
    """
    for article in articles:
        print(f"Title: {article.get('title')}")
        print(f"Source: {article.get('source')}")
        print(f"Publication Date: {article.get('publishedAt')}")
        print(f"Description: {article.get('description')}")
        print("-" * 80)

