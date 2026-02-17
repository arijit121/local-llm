"""
===========================================================================
routes/model_routes.py — Model Listing Routes
===========================================================================

PURPOSE:
    This file provides endpoints to list available AI models.
    It has TWO sets of routes:

    1. Our custom API      — GET /api/models?type=text
       Used by our own frontend to show model dropdowns.

    2. OpenAI-compatible   — GET /v1/models
       Used by external tools like Continue.dev, LangChain, etc.
       These tools expect the SAME format that OpenAI's API returns.

WHY TWO FORMATS?
    Our frontend uses a simple list of names: ["Model1", "Model2"]
    But tools like Continue.dev expect OpenAI's format:
    {
        "object": "list",
        "data": [
            {"id": "Model1", "object": "model", "created": ..., "owned_by": "local"}
        ]
    }
    By supporting both, our app works with our UI AND external tools!

USED BY:
    main.py (included via the router)
===========================================================================
"""

from datetime import datetime  # For generating timestamps

from fastapi import APIRouter

# Import our config (model settings from config.json)
from config_loader import config

# ---------------------------------------------------------------------------
# Create the router for model listing routes
# ---------------------------------------------------------------------------
router = APIRouter()


# ===========================================================================
# ROUTE 1: Our custom model listing (for our frontend)
# ===========================================================================

@router.get("/api/models")
async def get_models(type: str = "text"):
    """
    Get a list of available model names, filtered by type.

    QUERY PARAMETER:
        type (str): "text", "image", or "video"
                    Defaults to "text" if not specified.

    WHAT IS A QUERY PARAMETER?
        It's the part after the "?" in a URL.
        Example: /api/models?type=image
        FastAPI automatically extracts "type=image" and passes
        it to this function as the 'type' argument.

    Returns:
        list[str]: Just a list of model names (simple format).

    Example: GET /api/models?type=image → ["StableDiffusion", "DALL-E"]
    """
    if type == "image":
        models = config.get("image_models", [])
    elif type == "video":
        models = config.get("video_models", [])
    else:
        models = config.get("text_models", [])

    # Return only the names, not the full config objects
    return [m["name"] for m in models]


# ===========================================================================
# ROUTE 2: OpenAI-compatible model listing (for external tools)
# ===========================================================================

@router.get("/v1/models")
async def list_models():
    """
    List all available models in OpenAI-compatible format.

    This endpoint mimics OpenAI's model listing API so that external
    tools (like Continue.dev) can discover our local models.

    Returns data in the exact format OpenAI uses:
    {
        "object": "list",
        "data": [
            {
                "id": "ModelName",
                "object": "model",
                "created": 1234567890,
                "owned_by": "local"
            }
        ]
    }
    """
    text_models = config.get("text_models", [])

    data = []
    for m in text_models:
        data.append({
            "id": m["name"],                              # Model identifier
            "object": "model",                             # Always "model"
            "created": int(datetime.now().timestamp()),    # Unix timestamp
            "owned_by": "local"                            # We own this model!
        })

    return {"object": "list", "data": data}
