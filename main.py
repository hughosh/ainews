from news_api_client import fetch_top_headlines, print_articles
from vector_db import VectorDatabase
from openai_synthesizer import synthesize_story
from elevenlabs_client import synthesize_tts, get_audio_length, save_audio
import os

def main():
    # Define news sources and timeframe.
    sources = ["cnn", "fox-news", "the-new-york-times"]
    timeframe_hours = 24

    try:
        # Fetch news articles.
        articles = fetch_top_headlines(sources, hours=timeframe_hours)
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return

    if not articles:
        print("No articles to index.")
        return

    print("Fetched Articles:")
    print_articles(articles)

    # Build the FAISS vector database.
    vector_db = VectorDatabase()
    vector_db.build_index(articles)
    print(f"FAISS index built with {len(vector_db.documents)} documents.")

    # Query the FAISS index.
    query_text = "What are the latest political updates?"
    print(f"Querying FAISS index for: '{query_text}'")
    results = vector_db.query(query_text, k=3)
    print("Query Results:")
    print_articles(results)

    # Synthesize a story using OpenAI.
    prompt_file = "prompt.txt"  # Make sure this file exists in your directory.
    try:
        synthesized_story = synthesize_story(query=query_text, articles=results, prompt_file=prompt_file)
    except Exception as e:
        print(f"Error during story synthesis: {e}")
        return

    print("Synthesized Story:")
    print(synthesized_story)

    # Retrieve the ElevenLabs voice ID from environment.
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    if not voice_id:
        print("ELEVENLABS_VOICE_ID environment variable is not set.")
        return

    # Convert the synthesized story to speech.
    try:
        audio_bytes = synthesize_tts(text=synthesized_story, voice_id=voice_id, stability=0.5, similarity_boost=0.5)
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return

    # Check the length of the audio.
    try:
        duration = get_audio_length(audio_bytes)
        print(f"Audio length: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting audio length: {e}")

    # Optionally save the audio output.
    output_file = "output.mp3"
    try:
        save_audio(audio_bytes, output_file)
        print(f"Audio saved to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")

if __name__ == '__main__':
    main()

