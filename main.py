"""
===========================================================================
main.py — Application Entry Point (The Starting Point of the App!)
===========================================================================

PURPOSE:
    This is the MAIN file that starts the entire application.
    Think of it as the "control center" that connects all the pieces.

    Previously, ALL the code was in this one file (450+ lines!).
    Now it's been split into smaller, focused modules:

    PROJECT STRUCTURE:
    ┌─────────────────────────────────────────────────────────┐
    │  main.py                  ← YOU ARE HERE (entry point)      │
    │  config_loader.py         ← Loads settings from config.json │
    │  database.py              ← SQLite database setup & helpers │
    │  models_loader.py         ← Loads AI models (text & image)  │
    │  schemas.py               ← Pydantic data validation models │
    │  routes/                                                    │
    │    ├── __init__.py         ← Makes "routes" a Python package│
    │    ├── page_routes.py      ← Serves HTML pages              │
    │    ├── conversation_routes.py ← Conversation CRUD endpoints │
    │    ├── model_routes.py     ← Model listing endpoints        │
    │    └── chat_routes.py      ← Chat & AI generation endpoints │
    │  templates/                                                 │
    │    └── index.html          ← The chat interface (frontend)  │
    │  static/                   ← CSS, JS, images for frontend   │
    │  data/                                                      │
    │    ├── history.db          ← SQLite database file            │
    │    └── outputs/            ← Generated images are saved here│
    │  models/                   ← AI model files go here         │
    │  config.json               ← App configuration file         │
    └─────────────────────────────────────────────────────────────┘

HOW TO RUN:
    Option 1 (recommended for development):
        uvicorn main:app --reload

    Option 2:
        python main.py

WHAT IS FastAPI?
    FastAPI is a modern Python web framework for building APIs.
    It's fast, easy to use, and automatically generates documentation.
    Visit http://localhost:8000/docs to see interactive API documentation!

WHAT IS Uvicorn?
    Uvicorn is an ASGI server — it's what actually runs our FastAPI app
    and handles incoming web requests. Think of it as the "engine" that
    powers the web server.
===========================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Import FastAPI framework
# ═══════════════════════════════════════════════════════════════════════════
import asyncio                                   # For running blocking tasks in background
from contextlib import asynccontextmanager       # For the lifespan handler

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  # For serving CSS, JS, images

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Import our custom modules
# ═══════════════════════════════════════════════════════════════════════════
# These imports also TRIGGER the initialization code in each module:
#   - config_loader  → loads config.json into memory
#   - database       → creates the SQLite tables if they don't exist
#   - models_loader  → loads the AI models into memory (can take a moment!)

from config_loader import config       # App configuration (from config.json)
from database import init_db           # Database initialization
from models_loader import (
    load_text_model,                    # Function to load text AI models
    load_image_model,                   # Function to load image AI models
)
from routes.page_routes import router as page_router              # HTML page routes
from routes.conversation_routes import router as conversation_router  # Conversation CRUD
from routes.model_routes import router as model_router              # Model listing
from routes.chat_routes import router as chat_router                # Chat & AI generation


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Define lifespan handler (background model loading)
# ═══════════════════════════════════════════════════════════════════════════
# Previously, load_text_model() and load_image_model() were called at module
# level, which BLOCKED the server from starting until all models were loaded.
# Now, we load them in BACKGROUND THREADS so the server starts immediately
# and the web page is accessible right away. Models will be ready once
# loading finishes in the background.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — runs background tasks when the app starts up.

    WHAT CHANGED:
        Before: Models loaded synchronously → server blocked → blank page
        After:  Models load in background threads → server starts instantly

    asyncio.to_thread() runs the blocking model-loading functions in a
    separate thread so they don't block the main async event loop.
    """
    import os
    if os.getenv("TESTING") == "1":
        print("🧪 Test mode: Skipping background model loading.")
        yield
        return

    # Fire-and-forget: start model loading in background threads
    asyncio.create_task(asyncio.to_thread(load_text_model))
    asyncio.create_task(asyncio.to_thread(load_image_model))
    print("🚀 Server started! Models are loading in the background...")

    yield  # Server is running and accepting requests

    # Cleanup code (if needed) would go here after yield
    print("Server shutting down...")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Create the FastAPI application
# ═══════════════════════════════════════════════════════════════════════════
# This creates the main "app" object that handles all web requests.
# The lifespan handler ensures models load in the background on startup.
app = FastAPI(lifespan=lifespan)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Mount static file directories
# ═══════════════════════════════════════════════════════════════════════════
# "Mounting" means telling FastAPI to serve files from a specific folder
# when a certain URL is requested.
#
# Example: A request to /static/style.css will serve the file at
#          static/style.css from disk.

# Serve CSS, JavaScript, and other static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve generated images/outputs so they can be displayed in the chat
app.mount("/outputs", StaticFiles(directory="data/outputs"), name="outputs")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Include all API routes from our route modules
# ═══════════════════════════════════════════════════════════════════════════
# Each router handles a different part of the API:
#   - page_router         → serves the HTML chat page
#   - conversation_router → create, list, delete conversations
#   - model_router        → list available AI models
#   - chat_router         → send messages & get AI responses
app.include_router(page_router)
app.include_router(conversation_router)
app.include_router(model_router)
app.include_router(chat_router)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Run the server (only when running this file directly)
# ═══════════════════════════════════════════════════════════════════════════
# The "if __name__ == '__main__'" check means:
#   - If you run "python main.py"     → this code RUNS
#   - If you run "uvicorn main:app"   → this code is SKIPPED
#     (because uvicorn imports the file instead of running it directly)
#
# This is a common Python pattern to separate "importable module" from
# "runnable script".

if __name__ == "__main__":
    import uvicorn
    # Start the web server on all network interfaces (0.0.0.0) at port 8000
    # This makes the app accessible at: http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
