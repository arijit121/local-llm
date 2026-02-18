"""
===========================================================================
routes/chat_routes.py — Chat & Text Generation Routes
===========================================================================

PURPOSE:
    This file contains the HEART of the application — the routes that
    handle sending messages and generating AI responses.

    It has TWO main routes:

    1. POST /api/chat              — Our custom chat endpoint
       Used by our frontend. Supports text, image, and video modes.

    2. POST /v1/chat/completions   — OpenAI-compatible chat endpoint
       Used by external tools like Continue.dev.

HOW THE CHAT FLOW WORKS:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │ User sends   │ ──→ │ Save message  │ ──→ │ Generate    │
    │ a message    │     │ to database   │     │ AI response │
    └─────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
    ┌─────────────┐     ┌──────────────┐             │
    │ Return      │ ←── │ Auto-generate│ ←───────────┘
    │ response    │     │ chat title   │
    └─────────────┘     └──────────────┘

USED BY:
    main.py (included via the router)
===========================================================================
"""

import uuid    # For generating unique filenames for saved images
import os      # For file path operations
import json    # For converting data to JSON (used in streaming)

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Import our custom modules
import models_loader  # Access to global model variables (llm, pipe)
from database import OUTPUTS_DIR, get_db_connection
from models_loader import load_text_model, load_image_model
from schemas import ChatRequest, OpenAIChatCompletionRequest
from web_search import search_web, format_search_context

# ---------------------------------------------------------------------------
# Create the router for chat routes
# ---------------------------------------------------------------------------
router = APIRouter()


# ===========================================================================
# ROUTE 1: OpenAI-Compatible Chat Completions
# ===========================================================================

@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest):
    """
    Generate a chat completion in OpenAI-compatible format.

    This is the endpoint that tools like Continue.dev use to chat
    with our local AI. It supports both:
    - Regular mode  : returns the complete response at once
    - Streaming mode: sends response word-by-word (like ChatGPT's typing)

    WHAT IS STREAMING?
        Instead of waiting for the ENTIRE response to be generated,
        streaming sends each piece as it's generated.
        This feels much more responsive to the user.

    WHAT IS SSE (Server-Sent Events)?
        SSE is a web standard for sending a stream of events from
        server to client. Each event looks like:
            data: {"choices": [{"delta": {"content": "Hello"}}]}
        The client reads these events one by one and displays the text.

    Steps:
    1. Load the requested model (or error if not found)
    2. Format messages for llama.cpp
    3. Generate response (streaming or regular)
    4. Return in OpenAI-compatible format
    """
    # Step 1: Load the requested model
    target_model = request.model
    llm_instance = load_text_model(target_model)

    if not llm_instance:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{target_model}' not found or failed to load."
        )

    # Step 2: Convert our message objects to simple dictionaries
    # llama.cpp expects: [{"role": "user", "content": "Hi"}]
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Step 3: Generate response
    if request.stream:
        # --- STREAMING MODE ---
        # We use a Python "async generator" to send data piece by piece
        async def event_generator():
            # Ask llama.cpp to stream the response
            stream = llm_instance.create_chat_completion(
                messages=messages,
                stream=True,            # Enable streaming
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            # Send each chunk as a Server-Sent Event (SSE)
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"

            # Signal that the stream is complete
            yield "data: [DONE]\n\n"

        # Return a StreamingResponse (sends data as it's generated)
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"  # Standard SSE content type
        )
    else:
        # --- REGULAR MODE ---
        # Generate the full response at once and return it
        return llm_instance.create_chat_completion(
            messages=messages,
            stream=False,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )


# ===========================================================================
# ROUTE 2: Our Custom Chat Endpoint (the heart of the app!)
# ===========================================================================

@router.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handle a chat message — the MAIN route that powers our chat interface.

    This is the most complex route in the app! Here's what it does:

    MODES:
    1. TEXT  → Uses Llama model to generate a text reply
    2. IMAGE → Uses Stable Diffusion to generate an image
    3. VIDEO → Not implemented yet (returns a placeholder message)

    STEP-BY-STEP:
    1. Save the user's message to the database
    2. Generate an AI response based on the selected mode
    3. Save the AI's response to the database
    4. Auto-generate a title for new conversations
    5. Return the response to the frontend
    """
    conn = get_db_connection()
    c = conn.cursor()

    # -----------------------------------------------------------------------
    # Step 1: Save the user's message to the database
    # -----------------------------------------------------------------------
    c.execute(
        "INSERT INTO messages (conversation_id, role, content, type) "
        "VALUES (?, ?, ?, ?)",
        (request.conversation_id, "user", request.message, "text")
    )

    # Also save which mode and model the user is currently using
    c.execute(
        "UPDATE conversations SET last_mode = ?, last_model = ? WHERE id = ?",
        (request.mode, request.model, request.conversation_id)
    )
    conn.commit()

    # Initialize response variables
    response_content = ""   # Will hold the AI's response text or image URL
    response_type = "text"  # Will be "text" or "image"

    # -----------------------------------------------------------------------
    # Step 2: Generate the AI response based on mode
    # -----------------------------------------------------------------------

    if request.mode == "image":
        # ===== IMAGE GENERATION MODE =====
        response_content, response_type = _generate_image_response(
            request.message, request.model
        )

    elif request.mode == "video":
        # ===== VIDEO GENERATION MODE (NOT YET IMPLEMENTED) =====
        response_content = "Video generation is not yet implemented."
        response_type = "text"

    else:
        # ===== TEXT GENERATION MODE =====
        response_content, response_type = _generate_text_response(
            request, c
        )

    # -----------------------------------------------------------------------
    # Step 3: Save the AI's response to the database
    # -----------------------------------------------------------------------
    c.execute(
        "INSERT INTO messages (conversation_id, role, content, type) "
        "VALUES (?, ?, ?, ?)",
        (request.conversation_id, "assistant", response_content, response_type)
    )
    conn.commit()

    # -----------------------------------------------------------------------
    # Step 4: Auto-generate a title for new conversations
    # -----------------------------------------------------------------------
    new_title = _auto_generate_title(
        c, conn, request, response_content, response_type
    )

    conn.close()

    # -----------------------------------------------------------------------
    # Step 5: Return the response to the frontend
    # -----------------------------------------------------------------------
    return {
        "role": "assistant",
        "content": response_content,
        "type": response_type,
        "new_title": new_title  # None if title wasn't changed
    }


# ===========================================================================
# HELPER FUNCTIONS (used by the chat route above)
# ===========================================================================
# These are private helper functions (prefixed with _) that break down
# the chat route into smaller, easier-to-understand pieces.

def _generate_image_response(prompt: str, model_name: str) -> tuple:
    """
    Generate an image from a text prompt using Stable Diffusion.

    Args:
        prompt     : The text description of the image to generate
        model_name : Which image model to use

    Returns:
        tuple: (response_content, response_type)
               - On success: ("/outputs/filename.png", "image")
               - On failure: ("Error message...", "text")
    """
    # Load the image model
    models_loader.pipe = load_image_model(model_name)

    if models_loader.pipe is None:
        raise HTTPException(
            status_code=500,
            detail="Image generation model not loaded. Check server logs."
        )

    clean_prompt = prompt.strip()  # Remove extra whitespace

    try:
        # Generate the image! This is where the magic happens.
        # pipe(prompt) runs the entire Stable Diffusion pipeline:
        #   text → encoder → diffusion → decoder → image
        image = models_loader.pipe(clean_prompt).images[0]

        # Save the generated image with a unique filename
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(OUTPUTS_DIR, filename)
        image.save(image_path)

        # Return the URL path (not file path) for the frontend
        return f"/outputs/{filename}", "image"

    except Exception as e:
        # If image generation fails, return the error as text
        return f"Error generating image: {str(e)}", "text"


def _generate_text_response(request: ChatRequest, cursor) -> tuple:
    """
    Generate a text response using the Llama LLM.

    If web_search is enabled, this function will:
    1. Search the web for the user's question
    2. Inject the search results as context into the prompt
    3. Let the LLM generate an answer grounded in web data

    Args:
        request : The chat request containing the user's message
        cursor  : Database cursor to fetch conversation history

    Returns:
        tuple: (response_content, response_type)
               - On success: ("AI's response text...", "text")
               - On failure: ("Error message...", "text")
    """
    # Load the text model
    models_loader.llm = load_text_model(request.model)

    if models_loader.llm is None:
        return (
            "Error: Llama model not loaded. "
            "Please check config.json and models directory.",
            "text"
        )

    # Retrieve the FULL conversation history from the database
    # This gives the AI "memory" of what was said before
    cursor.execute(
        "SELECT role, content FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (request.conversation_id,)
    )
    history_rows = cursor.fetchall()

    # Convert to the format llama.cpp expects:
    # [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    messages = [{"role": r[0], "content": r[1]} for r in history_rows]

    # -----------------------------------------------------------------------
    # WEB SEARCH: If enabled, search the web and inject results as context
    # -----------------------------------------------------------------------
    if request.web_search:
        print(f"🌐 Web search enabled — searching for: {request.message}")
        search_results = search_web(request.message, max_results=5)
        search_context = format_search_context(search_results)

        if search_context:
            # Prepend a system message with web search results
            # This gives the LLM real-time web information to use
            web_system_message = {
                "role": "system",
                "content": (
                    "The user has enabled web search. Below are relevant "
                    "web search results for their latest question. Use this "
                    "information to provide an accurate, up-to-date answer. "
                    "Cite sources when possible.\n\n"
                    f"{search_context}"
                )
            }
            # Insert web context at the beginning of messages
            messages.insert(0, web_system_message)

    try:
        # Ask the LLM to generate a response based on the conversation
        response = models_loader.llm.create_chat_completion(messages=messages)

        # Extract just the text from the response object
        return response['choices'][0]['message']['content'], "text"

    except Exception as e:
        return f"Error communicating with Llama: {str(e)}", "text"


def _auto_generate_title(cursor, conn, request, response_content, response_type) -> str:
    """
    Auto-generate a title for new conversations.

    If the conversation is still called "New Chat", we give it a real name:
    - For TEXT mode:  Ask the LLM to generate a short title
    - For IMAGE/VIDEO mode: Use the first 50 characters of the user's prompt

    Args:
        cursor           : Database cursor
        conn             : Database connection (for committing changes)
        request          : The original chat request
        response_content : The AI's response
        response_type    : "text" or "image"

    Returns:
        str or None: The new title if generated, None if not changed.
    """
    # Check the current title
    cursor.execute(
        "SELECT title FROM conversations WHERE id = ?",
        (request.conversation_id,)
    )
    row = cursor.fetchone()
    current_title = row[0] if row else "New Chat"

    # Only generate a title if it's still the default "New Chat"
    if current_title != "New Chat":
        return None  # Title already set, nothing to do

    new_title = None

    if response_type == "text" and models_loader.llm:
        # --- Use the LLM to generate a clever title ---
        try:
            title_prompt = [
                {
                    "role": "user",
                    "content": (
                        f"Generate a short, concise title (max 5 words) "
                        f"for this chat conversation based on the following "
                        f"exchange:\nUser: {request.message}\n"
                        f"AI: {response_content}\nTitle:"
                    )
                }
            ]
            title_response = models_loader.llm.create_chat_completion(
                messages=title_prompt,
                max_tokens=10  # Keep the title short
            )
            generated_title = (
                title_response['choices'][0]['message']['content']
                .strip()        # Remove whitespace
                .strip('"')     # Remove surrounding quotes
            )

            if generated_title:
                cursor.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (generated_title, request.conversation_id)
                )
                conn.commit()
                new_title = generated_title

        except Exception as e:
            print(f"Error generating title: {e}")
    else:
        # --- For image/video: use the beginning of the user's prompt ---
        generated_title = request.message[:50].strip()
        if len(request.message) > 50:
            generated_title += "..."  # Add ellipsis if truncated

        cursor.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (generated_title, request.conversation_id)
        )
        conn.commit()
        new_title = generated_title

    return new_title
