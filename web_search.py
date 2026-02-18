"""
===========================================================================
web_search.py — Web Search Utility Module
===========================================================================

PURPOSE:
    This module provides web search capability using DuckDuckGo.
    It allows the LLM to access real-time web information before
    answering user questions.

HOW IT WORKS:
    1. Takes the user's question as a search query
    2. Searches DuckDuckGo (free, no API key needed!)
    3. Returns the top results as formatted text
    4. This text is then injected into the LLM prompt as context

USED BY:
    routes/chat_routes.py (when web search is enabled)
===========================================================================
"""

from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Args:
        query       : The search query (usually the user's question)
        max_results : Maximum number of results to return (default: 5)

    Returns:
        list[dict]: Each dict has keys: 'title', 'href', 'body'
                    - title : The page title
                    - href  : The URL
                    - body  : A snippet/summary of the page content

    Example:
        results = search_web("What is Python?")
        # [{"title": "Python.org", "href": "https://...", "body": "Python is..."}]
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def format_search_context(results: list[dict]) -> str:
    """
    Format search results into a text block that the LLM can understand.

    This creates a structured context string that gets prepended to the
    LLM conversation, giving it access to real-time web information.

    Args:
        results : List of search result dicts from search_web()

    Returns:
        str: Formatted text block with numbered results.

    Example output:
        [Web Search Results]
        1. "Python.org" (https://python.org)
           Python is a programming language that lets you work quickly...
        2. ...
    """
    if not results:
        return ""

    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("href", "")
        snippet = r.get("body", "No description")
        lines.append(f"{i}. \"{title}\" ({url})")
        lines.append(f"   {snippet}")
        lines.append("")  # blank line between results

    return "\n".join(lines)
