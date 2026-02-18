"""
===========================================================================
routes/page_routes.py — HTML Page Routes
===========================================================================

PURPOSE:
    This file serves the HTML pages of the application.
    Right now we only have ONE page (the chat interface), but if
    you add more pages later (settings, about, etc.), they go here.

WHAT IS A "TEMPLATE"?
    A template is an HTML file with special placeholders that get
    filled in with data before being sent to the browser.
    We use Jinja2 — a popular Python templating engine.
    Our templates live in the "templates/" folder.

USED BY:
    main.py (included via the router)
===========================================================================
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Create the router for page routes
# ---------------------------------------------------------------------------
router = APIRouter()

# Set up Jinja2 to look for HTML files in the "templates" folder
templates = Jinja2Templates(directory="templates")


# ===========================================================================
# ROUTES
# ===========================================================================

@router.get("/")
async def read_root(request: Request):
    """
    Serve the main chat page (index.html).

    When someone visits http://localhost:8000/ in their browser,
    this function sends them the HTML page.

    The 'request' parameter is required by Jinja2Templates —
    it contains info about the incoming HTTP request.
    """
    return templates.TemplateResponse(request, "index.html")


@router.get("/chat/{chat_id}")
async def read_chat(request: Request, chat_id: str):
    """
    Serve the chat page for a specific conversation.

    When someone visits http://localhost:8000/chat/abc-123 (directly
    or after a page refresh), this serves the same index.html.
    The frontend JavaScript then reads the URL and auto-loads
    the correct conversation.
    """
    return templates.TemplateResponse(request, "index.html")

