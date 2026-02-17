"""
===========================================================================
config_loader.py — Configuration Loading Module
===========================================================================

PURPOSE:
    This file is responsible for loading the application's settings
    from a JSON file called "config.json".

    The config file tells the app which AI models to use for:
    - Text generation (e.g., chatbots)
    - Image generation (e.g., Stable Diffusion)
    - Video generation (future feature)

HOW IT WORKS:
    1. We check if "config.json" exists in the project folder.
    2. If it exists  → we read it and return the settings as a dictionary.
    3. If it doesn't → we return default (fallback) settings so the app
       can still start without crashing.

USED BY:
    main.py, models_loader.py, routes/api_routes.py
===========================================================================
"""

import os   # 'os' lets us interact with the file system (check if files exist, etc.)
import json  # 'json' lets us read/write JSON files (a common data format)

# ---------------------------------------------------------------------------
# Path to the configuration file
# ---------------------------------------------------------------------------
# This is the filename the app will look for in the project root folder.
CONFIG_PATH = "config.json"


def load_config() -> dict:
    """
    Load and return the application configuration from config.json.

    Returns:
        dict: A dictionary containing model configurations with these keys:
              - "text_models"  : list of text AI models  (for chatting)
              - "image_models" : list of image AI models (for generating images)
              - "video_models" : list of video AI models (not yet implemented)

    Example of what config.json looks like:
        {
            "text_models": [
                {"name": "MyModel", "path": "models/mymodel.gguf", "n_ctx": 2048}
            ],
            "image_models": [
                {"name": "SD", "path": "models/stable-diffusion", "device": "auto"}
            ],
            "video_models": []
        }
    """

    # Step 1: Check if the config file exists
    if os.path.exists(CONFIG_PATH):
        # Step 2a: File found! Open it and parse the JSON data
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    # Step 2b: File NOT found — return safe default settings
    # This prevents the app from crashing if config.json is missing
    return {
        "text_models": [{"name": "Default", "path": "", "n_ctx": 2048}],
        "image_models": [{"name": "Default", "path": "models", "device": "auto"}],
        "video_models": []
    }


# ---------------------------------------------------------------------------
# Load config once when this module is first imported
# ---------------------------------------------------------------------------
# By doing this at module level, every other file that imports 'config'
# gets the SAME configuration object — no need to reload from disk.
config = load_config()
