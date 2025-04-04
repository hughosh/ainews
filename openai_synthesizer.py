import os
from openai import OpenAI
from typing import List, Dict, Any

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_prompt(prompt_file: str) -> str:
    """
    Loads the prompt template from a text file.
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template

def format_articles_for_prompt(articles: List[Dict[str, Any]]) -> str:
    """
    Formats the list of articles into a single string to be inserted into the prompt.
    """
    article_strings = []
    for article in articles:
        title = article.get("title", "No Title")
        source = article.get("source", "Unknown Source")
        timestamp = article.get("timestamp", "")
        topic = article.get("topic", "")
        keywords = article.get("keywords", [])
        description = article.get("description", "")
        article_str = (
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Timestamp: {timestamp}\n"
            f"Topic: {topic}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Description: {description}\n"
        )
        article_strings.append(article_str)
    return "\n---\n".join(article_strings)

def synthesize_story(query: str, articles: List[Dict[str, Any]], prompt_file: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Synthesizes a story using OpenAI's API by combining the query with the given articles.

    Parameters:
      query (str): The query or topic for the synthesis.
      articles (List[Dict[str, Any]]): The list of articles to incorporate.
      prompt_file (str): The path to the prompt template file.
      model (str): The OpenAI model to use.

    Returns:
      str: The synthesized story.
    """
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Load the prompt template from file.
    prompt_template = load_prompt(prompt_file)
    # Format articles for insertion.
    articles_text = format_articles_for_prompt(articles)

    # Format the prompt with the query and articles.
    prompt = prompt_template.format(query=query, articles=articles_text)

    messages = [
        {"role": "system", "content": "You are an experienced journalist and story synthesizer."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

    synthesized_story = response.choices[0].message.content.strip()
    return synthesized_story

