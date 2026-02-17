"""
===========================================================================
routes/conversation_routes.py — Conversation CRUD Routes
===========================================================================

PURPOSE:
    This file handles all operations related to CONVERSATIONS:
    - Creating a new conversation
    - Listing all conversations
    - Deleting a conversation
    - Getting messages for a conversation

WHAT IS CRUD?
    CRUD stands for Create, Read, Update, Delete — the four basic
    operations you can do on data in a database.

    In this file:
    - CREATE → POST   /api/conversations       (make a new chat)
    - READ   → GET    /api/conversations       (list all chats)
    - READ   → GET    /api/conversations/{id}  (get messages for a chat)
    - DELETE → DELETE /api/conversations/{id}  (remove a chat)

USED BY:
    main.py (included via the router)
===========================================================================
"""

import uuid    # For generating unique conversation IDs
import sqlite3  # For database access

from fastapi import APIRouter, HTTPException

# Import our custom modules
from database import get_db_connection
from schemas import ConversationCreate

# ---------------------------------------------------------------------------
# Create the router for conversation routes
# ---------------------------------------------------------------------------
router = APIRouter()


# ===========================================================================
# ROUTE 1: Create a new conversation
# ===========================================================================

@router.post("/api/conversations")
async def create_conversation(conv: ConversationCreate):
    """
    Create a new conversation.

    How it works:
    1. Generate a unique ID (UUID) for the new conversation
    2. Insert it into the database with the given title
    3. Return the new conversation's ID and title

    Called by the frontend when the user clicks "New Chat".

    WHAT IS A UUID?
        UUID = Universally Unique Identifier
        It's a random string like "550e8400-e29b-41d4-a716-446655440000"
        that is practically guaranteed to be unique worldwide.
    """
    # Generate a unique ID for this conversation
    conv_id = str(uuid.uuid4())

    # Save to database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (id, title) VALUES (?, ?)",
        (conv_id, conv.title)
    )
    conn.commit()
    conn.close()

    # Return the new conversation info as JSON
    return {"id": conv_id, "title": conv.title}


# ===========================================================================
# ROUTE 2: List all conversations
# ===========================================================================

@router.get("/api/conversations")
async def get_conversations():
    """
    Get a list of ALL conversations, sorted by newest first.

    Returns a list of conversation objects, each containing:
    - id         : unique identifier
    - title      : conversation name
    - created_at : when it was created
    - last_mode  : last used mode (text/image/video)
    - last_model : name of the last used model

    Called by the frontend to populate the sidebar with chat history.
    """
    conn = get_db_connection()
    c = conn.cursor()

    # ORDER BY created_at DESC = newest conversations first
    c.execute(
        "SELECT id, title, created_at, last_mode, last_model "
        "FROM conversations ORDER BY created_at DESC"
    )
    rows = c.fetchall()
    conn.close()

    # Convert database rows into a list of dictionaries (JSON-friendly)
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "last_mode": r[3] or "text",   # Default to "text" if NULL
            "last_model": r[4] or ""       # Default to "" if NULL
        }
        for r in rows
    ]


# ===========================================================================
# ROUTE 3: Delete a conversation
# ===========================================================================

@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.

    The {conversation_id} in the URL is a PATH PARAMETER — FastAPI
    automatically extracts it from the URL and passes it to this function.

    Steps:
    1. Delete all messages that belong to this conversation
    2. Delete the conversation itself
    3. Return a success status

    WHY DELETE MESSAGES FIRST?
        Because messages have a FOREIGN KEY that references conversations.
        If we delete the conversation first, the messages would become
        "orphaned" (pointing to nothing). Deleting messages first is cleaner.

    Example: DELETE /api/conversations/abc-123 → deletes conversation "abc-123"
    """
    conn = get_db_connection()
    c = conn.cursor()

    try:
        # Delete messages FIRST (they reference the conversation)
        c.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        # Then delete the conversation itself
        c.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
    except Exception as e:
        conn.close()
        # Return HTTP 500 (Internal Server Error) with the error details
        raise HTTPException(status_code=500, detail=str(e))

    conn.close()
    return {"status": "success", "id": conversation_id}


# ===========================================================================
# ROUTE 4: Get messages for a conversation
# ===========================================================================

@router.get("/api/conversations/{conversation_id}")
async def get_messages(conversation_id: str):
    """
    Get all messages for a specific conversation.

    Returns:
    - messages   : list of all messages in chronological order
    - last_mode  : what mode this conversation was last using
    - last_model : what model this conversation was last using

    Called by the frontend when the user clicks on a conversation
    in the sidebar to load its chat history.
    """
    conn = get_db_connection()
    c = conn.cursor()

    # Get conversation metadata (mode and model info)
    c.execute(
        "SELECT last_mode, last_model FROM conversations WHERE id = ?",
        (conversation_id,)
    )
    conv_row = c.fetchone()
    last_mode = conv_row[0] if conv_row else "text"
    last_model = conv_row[1] if conv_row else ""

    # Get all messages, ordered by ID (chronological order)
    c.execute(
        "SELECT role, content, type FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,)
    )
    rows = c.fetchall()
    conn.close()

    return {
        "messages": [
            {"role": r[0], "content": r[1], "type": r[2]}
            for r in rows
        ],
        "last_mode": last_mode or "text",
        "last_model": last_model or ""
    }
