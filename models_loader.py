"""
===========================================================================
models_loader.py — AI Model Loading Module
===========================================================================

PURPOSE:
    This file handles loading AI models into memory so they can be used
    to generate text or images.

TWO TYPES OF MODELS:
    1. TEXT MODELS  — Use "llama.cpp" (a fast C++ library for running LLMs)
                      These models generate text responses in chat.
                      File format: .gguf files

    2. IMAGE MODELS — Use "Stable Diffusion" (via the 'diffusers' library)
                      These models generate images from text descriptions.
                      They are stored as folders with model weights.

KEY CONCEPTS:
    - "Loading a model" means reading its file from disk into RAM/GPU memory.
      This can take several seconds, so we only do it once and reuse it.
    - We use GLOBAL VARIABLES (llm, pipe) to keep the loaded models in
      memory so every request can access them without reloading.
    - "n_ctx" = context length = how much text the model can "remember"
    - "n_gpu_layers" = how many layers to run on GPU (0 = CPU only)

USED BY:
    main.py (initial load), routes/api_routes.py (for generating responses)
===========================================================================
"""

import os     # For checking if model files/folders exist

# NOTE: Heavy imports (torch, llama_cpp, diffusers) are NOT imported here!
# They are imported LAZILY inside the functions that need them.
# This is because importing these libraries can take MINUTES, and we don't
# want to block the server from starting while they load.

# Import our config (the settings we loaded from config.json)
from config_loader import config


# ===========================================================================
# SECTION 1: TEXT MODEL (Llama / LLM)
# ===========================================================================

# ---------------------------------------------------------------------------
# Global variables to keep the loaded text model in memory
# ---------------------------------------------------------------------------
llm = None                    # The loaded Llama model object (None = not loaded)
current_text_model_name = None  # Name of the currently loaded model


def load_text_model(model_name: str = None):
    """
    Load a text generation model (LLM) using llama.cpp.

    This function:
    1. Looks up the model by name in config.json
    2. If no name given, uses the FIRST model in the list
    3. Skips loading if the same model is already loaded (saves time!)
    4. Loads the .gguf model file into memory

    Args:
        model_name (str, optional): Name of the model to load.
                                     If None, loads the first available model.

    Returns:
        Llama object if successful, None if loading fails.

    Example:
        model = load_text_model("MyModel")
        if model:
            response = model.create_chat_completion(messages=[...])
    """
    global llm, current_text_model_name

    # Step 1: Get the list of text models from config
    models = config.get("text_models", [])
    if not models:
        print("Warning: No text models configured.")
        return None

    # Step 2: Find the requested model by name
    selected_model = None
    if model_name:
        for m in models:
            if m["name"] == model_name:
                selected_model = m
                break  # Found it! Stop searching.

    # Step 3: If not found (or no name given), use the first model as default
    if not selected_model:
        selected_model = models[0]

    # Step 4: Skip if this model is already loaded (optimization!)
    # This avoids wasting time reloading the same model on every request
    if llm is not None and current_text_model_name == selected_model["name"]:
        return llm

    # Step 5: Actually load the model from disk
    try:
        # Lazy import: only load llama_cpp when we actually need it
        from llama_cpp import Llama

        path = selected_model["path"]

        # Check if the model file exists
        if os.path.exists(path):
            print(f"Loading Llama model: {selected_model['name']} from {path}")

            # Create the Llama model object
            # - model_path : where the .gguf file is located
            # - n_ctx      : context window size (how much text it can process)
            # - n_gpu_layers: number of model layers to offload to GPU (0 = CPU)
            # - verbose    : False = don't print extra debug info
            llm = Llama(
                model_path=path,
                n_ctx=selected_model.get("n_ctx", 2048),
                n_gpu_layers=selected_model.get("n_gpu_layers", 0),
                verbose=False
            )
            current_text_model_name = selected_model["name"]
            return llm
        else:
            print(f"Error: Model path not found: {path}")
            return None

    except Exception as e:
        # If anything goes wrong, print the error and return None
        print(f"Error loading model {selected_model['name']}: {e}")
        return None


# ===========================================================================
# SECTION 2: IMAGE MODEL (Stable Diffusion)
# ===========================================================================

# ---------------------------------------------------------------------------
# Global variables to keep the loaded image model in memory
# ---------------------------------------------------------------------------
pipe = None                     # The loaded Stable Diffusion pipeline (None = not loaded)
current_image_model_name = None  # Name of the currently loaded image model


def load_image_model(model_name: str = None):
    """
    Load an image generation model (Stable Diffusion) using diffusers.

    This function:
    1. Looks up the model by name in config.json
    2. If no name given, uses the FIRST model in the list
    3. Skips loading if the same model is already loaded
    4. Automatically detects whether to use GPU (CUDA) or CPU
    5. Uses float16 on GPU (faster, less memory) and float32 on CPU

    Args:
        model_name (str, optional): Name of the image model to load.
                                     If None, loads the first available.

    Returns:
        StableDiffusionPipeline object if successful, None if loading fails.

    WHAT IS A "PIPELINE"?
        In machine learning, a "pipeline" is a chain of processing steps.
        For Stable Diffusion, the pipeline handles:
        text prompt → text encoder → diffusion process → image decoder → final image
    """
    global pipe, current_image_model_name

    # Step 1: Get the list of image models from config
    models = config.get("image_models", [])
    if not models:
        print("Warning: No image models configured.")
        return None

    # Step 2: Find the requested model by name
    selected_model = None
    if model_name:
        for m in models:
            if m["name"] == model_name:
                selected_model = m
                break

    # Step 3: Default to first model if none specified
    if not selected_model:
        selected_model = models[0]

    # Step 4: Skip if already loaded
    if pipe is not None and current_image_model_name == selected_model["name"]:
        return pipe

    # Step 5: Load the model
    try:
        # Lazy imports: only load torch and diffusers when we actually need them
        import torch
        from diffusers import StableDiffusionPipeline

        image_model_path = selected_model["path"]
        device_config = selected_model.get("device", "auto")

        # Fallback to default path if configured path doesn't exist
        if not os.path.exists(image_model_path):
            image_model_path = "./models"

        # Check if the model directory exists and has files in it
        if os.path.exists(image_model_path) and (
            os.path.isdir(image_model_path) and os.listdir(image_model_path)
        ):
            # Decide whether to use GPU or CPU
            # "auto" = use GPU if available, otherwise CPU
            use_cuda = device_config == "cuda" or (
                device_config == "auto" and torch.cuda.is_available()
            )

            if use_cuda:
                # GPU mode: uses float16 for faster processing and less memory
                print(f"Loading Diffusers model: {selected_model['name']} "
                      f"from {image_model_path} (CUDA, float16)")
                new_pipe = StableDiffusionPipeline.from_pretrained(
                    image_model_path,
                    torch_dtype=torch.float16,    # Half precision = faster on GPU
                    local_files_only=True          # Don't download from internet
                )
                new_pipe = new_pipe.to("cuda")     # Move model to GPU
            else:
                # CPU mode: uses float32 (full precision, slower but works everywhere)
                print(f"Loading Diffusers model: {selected_model['name']} "
                      f"from {image_model_path} (CPU, float32)")
                new_pipe = StableDiffusionPipeline.from_pretrained(
                    image_model_path,
                    torch_dtype=torch.float32,     # Full precision for CPU
                    local_files_only=True
                )
                new_pipe = new_pipe.to("cpu")      # Explicitly use CPU

            pipe = new_pipe
            current_image_model_name = selected_model["name"]
            return pipe
        else:
            print(f"Warning: Image model directory invalid: {image_model_path}")
            return None

    except Exception as e:
        print(f"Warning: Could not load Diffusers pipeline: {e}")
        return None
