from news_api_client import fetch_top_headlines, print_articles
from vector_db import VectorDatabase
from openai_synthesizer import synthesize_story

def main():
    # Specify the news sources and the timeframe (in hours)
    sources = ["cnn", "fox-news", "the-new-york-times"]
    timeframe_hours = 48

    try:
        # Fetch news articles
        articles = fetch_top_headlines(sources, hours=timeframe_hours)
    except Exception as e:
        print(f"An error occurred while fetching articles: {e}")
        return

    if not articles:
        print("No articles to index.")
        return

    print("\nNews from the past 24 hours (Raw Articles):\n")
    print_articles(articles)

    # Build the FAISS vector database with enriched metadata.
    vector_db = VectorDatabase()
    vector_db.build_index(articles)
    print("\nFAISS index built with", len(vector_db.documents), "documents.")

    # Query the FAISS vector database.
    query_text = "What are the latest political updates?"
    print(f"\nQuerying FAISS index for: '{query_text}'")
    results = vector_db.query(query_text, k=3)
    print("\nQuery results:")
    print_articles(results)

    # Synthesize a story using OpenAI API based on the queried articles.
    prompt_file = "prompt.txt"  # This file should contain your synthesis prompt template.
    try:
        synthesized_story = synthesize_story(query=query_text, articles=results, prompt_file=prompt_file)
    except Exception as e:
        print(f"An error occurred during synthesis: {e}")
        return

    print("\nSynthesized Story:\n")
    print(synthesized_story)

if __name__ == '__main__':
    main()

