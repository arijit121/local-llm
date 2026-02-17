"""
===========================================================================
schemas.py — Pydantic Data Models (Request/Response Schemas)
===========================================================================

PURPOSE:
    This file defines the "shape" of data that the API expects to receive
    and send back. Think of these as TEMPLATES or BLUEPRINTS for data.

WHAT IS PYDANTIC?
    Pydantic is a Python library that validates data automatically.
    When someone sends a request to our API, Pydantic checks:
    ✓ Are all required fields present?
    ✓ Are the data types correct? (string, number, etc.)
    ✓ If a field is missing, use the default value.

    If the data is INVALID, FastAPI automatically returns a clear error
    message — we don't have to write that error handling ourselves!

WHY USE SCHEMAS?
    Without schemas:
        - We'd have to manually check every field in every request
        - Typos in field names would silently pass through
        - No auto-generated API documentation

    With schemas:
        - Data is validated automatically
        - FastAPI generates interactive docs at /docs
        - Our code is cleaner and safer

USED BY:
    routes/api_routes.py (to validate incoming API requests)
===========================================================================
"""

from typing import List, Optional  # For type hints (List = list, Optional = can be None)
from pydantic import BaseModel      # Base class for all our data models


# ===========================================================================
# SECTION 1: Our Custom API Schemas
# ===========================================================================

class ChatRequest(BaseModel):
    """
    Schema for the /api/chat endpoint.

    When a user sends a message in the chat, the frontend sends this data:

    Example JSON:
        {
            "conversation_id": "abc-123-def",
            "message": "Hello, how are you?",
            "model": "default",
            "mode": "text"
        }

    Fields:
        conversation_id (str) : Which conversation this message belongs to
        message         (str) : The actual text the user typed
        model           (str) : Which AI model to use (default = "default")
        mode            (str) : What type of generation:
                                 "text"  = generate a text reply
                                 "image" = generate an image
                                 "video" = generate a video (not yet implemented)
    """
    conversation_id: str          # Required — no default value
    message: str                  # Required — no default value
    model: str = "default"        # Optional — defaults to "default"
    mode: str = "text"            # Optional — defaults to "text"


class ConversationCreate(BaseModel):
    """
    Schema for creating a new conversation.

    Example JSON:
        {
            "title": "My Chat About Python"
        }

    Fields:
        title (str, optional): Name for the new conversation.
                                Defaults to "New Chat" if not provided.
    """
    title: Optional[str] = "New Chat"


# ===========================================================================
# SECTION 2: OpenAI-Compatible Schemas
# ===========================================================================
# These schemas make our API COMPATIBLE with tools like "Continue.dev"
# that expect an OpenAI-style API (like ChatGPT's API).
#
# By mimicking OpenAI's API format, any tool designed for ChatGPT
# can also work with our local LLM!

class OpenAIMessage(BaseModel):
    """
    A single message in the OpenAI chat format.

    Example JSON:
        {
            "role": "user",
            "content": "What is Python?"
        }

    Fields:
        role    (str): Who sent this message — "user", "assistant", or "system"
        content (str): The message text
    """
    role: str       # "user", "assistant", or "system"
    content: str    # The actual message text


class OpenAIChatCompletionRequest(BaseModel):
    """
    Schema for the OpenAI-compatible /v1/chat/completions endpoint.

    This matches the format that OpenAI's ChatGPT API uses, so tools
    like Continue.dev can connect to our local server instead of OpenAI.

    Example JSON:
        {
            "model": "MyLocalModel",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain Python in one sentence."}
            ],
            "stream": true,
            "max_tokens": 500,
            "temperature": 0.7
        }

    Fields:
        model       (str)            : Which model to use
        messages    (list)           : The conversation history
        stream      (bool, optional) : If True, send response word-by-word
                                        (like ChatGPT's typing effect)
        max_tokens  (int, optional)  : Maximum length of the response
        temperature (float, optional): Controls randomness (0=focused, 1=creative)
    """
    model: str                              # Required — which model to use
    messages: List[OpenAIMessage]           # Required — conversation history
    stream: Optional[bool] = False          # Optional — stream response?
    max_tokens: Optional[int] = None        # Optional — limit response length
    temperature: Optional[float] = 0.7      # Optional — creativity level (0.0 - 1.0)
