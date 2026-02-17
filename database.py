"""
===========================================================================
database.py — Database Setup & Helper Functions
===========================================================================

PURPOSE:
    This file handles everything related to the SQLite database.

WHAT IS SQLite?
    SQLite is a lightweight database that stores all data in a single file
    (in our case: "data/history.db"). Unlike big databases like PostgreSQL
    or MySQL, SQLite doesn't need a separate server — it's built right
    into Python!

WHAT WE STORE:
    1. Conversations — Each chat session the user starts
    2. Messages      — Individual messages within each conversation
                       (both user messages and AI responses)

DATABASE TABLES:
    ┌─────────────────────────────────────────────────┐
    │  conversations                                  │
    ├─────────────────────────────────────────────────┤
    │  id          (TEXT)    — Unique ID (UUID)        │
    │  title       (TEXT)    — Chat title              │
    │  last_mode   (TEXT)    — "text" / "image" / ...  │
    │  last_model  (TEXT)    — Name of last used model │
    │  created_at  (TIME)    — When chat was created   │
    └─────────────────────────────────────────────────┘
            │
            │  one conversation has many messages
            ▼
    ┌─────────────────────────────────────────────────┐
    │  messages                                       │
    ├─────────────────────────────────────────────────┤
    │  id              (INT)   — Auto-incremented ID  │
    │  conversation_id (TEXT)  — Links to conversation │
    │  role            (TEXT)  — "user" or "assistant" │
    │  content         (TEXT)  — The actual message    │
    │  type            (TEXT)  — "text" / "image" / .. │
    │  timestamp       (TIME)  — When it was sent      │
    └─────────────────────────────────────────────────┘

USED BY:
    routes/api_routes.py (to save and retrieve conversations/messages)
===========================================================================
"""

import os       # For creating directories
import sqlite3  # Python's built-in SQLite database library

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
DB_PATH = "data/history.db"   # Where the database file will be stored
OUTPUTS_DIR = "data/outputs"  # Where generated images/videos will be saved

# ---------------------------------------------------------------------------
# Create necessary directories if they don't exist yet
# ---------------------------------------------------------------------------
# 'exist_ok=True' means "don't throw an error if the folder already exists"
os.makedirs("data", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a new database connection.

    WHY A FUNCTION?
        Instead of sharing one connection everywhere (which can cause
        problems with multiple requests), we create a fresh connection
        each time we need to talk to the database.

    Returns:
        sqlite3.Connection: A connection object that you can use to
                            run SQL queries.

    Usage Example:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations")
        rows = cursor.fetchall()
        conn.close()  # Always close when done!
    """
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Initialize the database by creating tables if they don't exist.

    This function runs once when the app starts. It:
    1. Creates the 'conversations' table (if not already there)
    2. Creates the 'messages' table (if not already there)
    3. Runs "migrations" — adds new columns to existing tables
       (this is safe because we use "try/except" to ignore errors
       if the columns already exist)

    WHAT IS A MIGRATION?
        When you update your app and need to add new columns to an
        existing database, you "migrate" it. For example, we added
        'last_mode' and 'last_model' columns later, so we use
        ALTER TABLE to add them without losing existing data.
    """

    # Open a connection to the database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()  # A "cursor" lets us execute SQL commands

    # -----------------------------------------------------------------------
    # Create the 'conversations' table
    # -----------------------------------------------------------------------
    # IF NOT EXISTS = only create if the table doesn't already exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            last_mode TEXT DEFAULT 'text',
            last_model TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # -----------------------------------------------------------------------
    # Migration: Safely add new columns to existing tables
    # -----------------------------------------------------------------------
    # We wrap each ALTER TABLE in try/except because:
    # - If the column already exists, SQLite throws an error
    # - We just ignore that error with 'pass' (do nothing)
    try:
        c.execute("ALTER TABLE conversations ADD COLUMN last_mode TEXT DEFAULT 'text'")
    except Exception:
        pass  # Column already exists — that's fine!

    try:
        c.execute("ALTER TABLE conversations ADD COLUMN last_model TEXT DEFAULT ''")
    except Exception:
        pass  # Column already exists — that's fine!

    # -----------------------------------------------------------------------
    # Create the 'messages' table
    # -----------------------------------------------------------------------
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            type TEXT DEFAULT 'text',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')
    # FOREIGN KEY = ensures every message belongs to a valid conversation

    # Save changes and close
    conn.commit()   # Write changes to disk
    conn.close()    # Release the database connection


# ---------------------------------------------------------------------------
# Run database initialization when this module is first imported
# ---------------------------------------------------------------------------
init_db()
