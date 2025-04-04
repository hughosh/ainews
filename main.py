from news_api_client import fetch_top_headlines, print_articles
from vector_db import VectorDatabase

def main():
    # Specify the news sources and timeframe (in hours)
    sources = ["cnn", "fox-news", "the-new-york-times"]
    timeframe_hours = 168

    try:
        # Fetch news articles with debug output enabled
        articles = fetch_top_headlines(sources, hours=timeframe_hours)
    except Exception as e:
        print(f"An error occurred while fetching articles: {e}")
        return

    if not articles:
        print("No articles to index.")
        return

    print("\nNews from the past 24 hours:\n")
    print_articles(articles)

    # Build FAISS vector database from the articles
    vector_db = VectorDatabase()
    vector_db.build_index(articles)
    print("\nFAISS index built with", len(vector_db.documents), "documents.")

    # Gut check query against the FAISS vector database
    query_text = "Latest political news"
    print(f"\nQuerying FAISS index for: '{query_text}'")
    results = vector_db.query(query_text, k=3)
    print("\nQuery results:")
    print_articles(results)

if __name__ == '__main__':
    main()

