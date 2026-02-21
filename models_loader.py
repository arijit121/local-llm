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
# SECTION 1: TEXT MODEL (Llama / LLM  OR  HuggingFace Transformers)
# ===========================================================================

# ---------------------------------------------------------------------------
# Global variables to keep the loaded text model in memory
# ---------------------------------------------------------------------------
llm = None                      # The loaded model object (None = not loaded)
current_text_model_name = None  # Name of the currently loaded model


# ---------------------------------------------------------------------------
# HuggingFace Adapter
# ---------------------------------------------------------------------------
# This adapter wraps a HuggingFace transformers pipeline so it exposes
# the SAME interface as llama.cpp's Llama object.
# That means chat_routes.py can call model.create_chat_completion(messages=...)
# on BOTH llama.cpp models AND HuggingFace models without any changes.

class HuggingFaceModelAdapter:
    """
    Wraps a HuggingFace text-generation pipeline to look like a llama.cpp Llama object.

    WHY?
        llama.cpp models are called like:
            llm.create_chat_completion(messages=[...])
        HuggingFace pipelines work differently, so we wrap them here
        to keep the rest of the app unchanged.
    """

    def __init__(self, pipeline, tokenizer):
        self.pipeline = pipeline
        self.tokenizer = tokenizer

    def create_chat_completion(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> dict:
        """
        Generate a chat response — same signature as llama.cpp.

        Args:
            messages    : List of {"role": ..., "content": ...} dicts
            max_tokens  : Maximum new tokens to generate
            temperature : Sampling temperature (0 = deterministic, 1 = creative)
            stream      : Ignored for HuggingFace models (not supported here)

        Returns:
            dict in OpenAI-compatible format:
            {
                "choices": [
                    {"message": {"content": "..."}}
                ]
            }
        """
        # Apply the model's chat template to format messages correctly
        # (each model has its own special format for system/user/assistant turns)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation if chat template fails
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            ) + "\nASSISTANT:"

        # Run the pipeline
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens if max_tokens else 512,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,   # Only return the NEW tokens, not the prompt
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = outputs[0]["generated_text"].strip()

        # Return in the same format as llama.cpp
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    }
                }
            ]
        }


def _is_huggingface_model(path: str) -> bool:
    """
    Returns True if the path is a HuggingFace model folder
    (detected by the presence of a config.json inside it).
    Returns False if it's a .gguf file path.
    """
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def _load_huggingface_model(model_config: dict) -> "HuggingFaceModelAdapter | None":
    """
    Load a HuggingFace model from a local folder using the transformers library.

    Args:
        model_config : Dict from config.json with 'path', 'n_ctx', etc.

    Returns:
        HuggingFaceModelAdapter on success, None on failure.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        path = model_config["path"]
        n_ctx = model_config.get("n_ctx", 4096)
        n_gpu_layers = model_config.get("n_gpu_layers", 0)

        # Decide device: use CUDA if n_gpu_layers != 0 and CUDA is available
        if n_gpu_layers != 0 and torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16   # Efficient on modern GPUs
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"Loading HuggingFace model: {model_config['name']} "
              f"from {path} ({device.upper()})")

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
            device_map=device if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to("cpu")

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=n_ctx,
            device=0 if device == "cuda" else -1,  # 0 = first GPU, -1 = CPU
        )

        print(f"✅ HuggingFace model loaded: {model_config['name']}")
        return HuggingFaceModelAdapter(hf_pipeline, tokenizer)

    except Exception as e:
        print(f"Error loading HuggingFace model {model_config['name']}: {e}")
        return None


def load_text_model(model_name: str = None):
    """
    Load a text generation model — either llama.cpp (GGUF) or HuggingFace.

    AUTO-DETECTION:
        - If the model path is a .gguf FILE  → use llama.cpp (fast, quantized)
        - If the model path is a FOLDER with config.json → use HuggingFace transformers

    This function:
    1. Looks up the model by name in config.json
    2. If no name given, uses the FIRST model in the list
    3. Skips loading if the same model is already loaded (saves time!)
    4. Routes to the correct backend based on the model format

    Args:
        model_name (str, optional): Name of the model to load.
                                     If None, loads the first available model.

    Returns:
        Model adapter object if successful, None if loading fails.
        The returned object always has a .create_chat_completion() method.
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
    if llm is not None and current_text_model_name == selected_model["name"]:
        return llm

    # Step 5: Route to the correct loader based on model format
    path = selected_model["path"]

    if _is_huggingface_model(path):
        # ── HuggingFace model (folder with config.json) ──
        llm = _load_huggingface_model(selected_model)
    else:
        # ── llama.cpp model (.gguf file) ──
        llm = _load_gguf_model(selected_model)

    if llm is not None:
        current_text_model_name = selected_model["name"]

    return llm


def _load_gguf_model(model_config: dict):
    """
    Load a GGUF model using llama.cpp.

    Args:
        model_config : Dict from config.json with 'path', 'n_ctx', 'n_gpu_layers'.

    Returns:
        Llama object on success, None on failure.
    """
    try:
        from llama_cpp import Llama
        import jinja2
        import re

        # Monkey-patch Jinja2 Environment to strip HuggingFace's custom 'generation' tags.
        # This prevents llama-cpp-python crashes when loading newer GGUF models.
        if not hasattr(jinja2.Environment, '_patched_for_generation'):
            _orig_from_string = jinja2.Environment.from_string
            def _patched_from_string(self, source, globals=None, template_class=None):
                if isinstance(source, str):
                    source = re.sub(r'\{%-?\s*generation\s*-?%\}', '', source)
                    source = re.sub(r'\{%-?\s*endgeneration\s*-?%\}', '', source)
                return _orig_from_string(self, source, globals, template_class)
            jinja2.Environment.from_string = _patched_from_string
            jinja2.Environment._patched_for_generation = True

        path = model_config["path"]
        chat_format = model_config.get("chat_format", None)

        if os.path.exists(path):
            print(f"Loading Llama model: {model_config['name']} from {path}"
                  f"{' (chat_format=' + chat_format + ')' if chat_format else ''}")

            # Build kwargs for the Llama constructor
            llama_kwargs = {
                "model_path": path,
                "n_ctx": model_config.get("n_ctx", 2048),
                "n_gpu_layers": model_config.get("n_gpu_layers", 0),
                "verbose": False,
            }

            # If chat_format is specified, handle it. We explicitly pass a basic chat_handler template override
            # to prevent it from auto-parsing the GGUF metadata template which contains unsupported Jinja tags.
            if chat_format:
                from llama_cpp.llama_chat_format import get_chat_completion_handler
                llama_kwargs["chat_handler"] = get_chat_completion_handler(chat_format)
            else:
                # If no chat format explicitly passed, we pass a basic template string so the internal gguf load
                # parser does not crash on the unsupported tag {% generation %} inside the metadata string.
                llama_kwargs["chat_format"] = "chatml"

            loaded = Llama(**llama_kwargs)
            return loaded
        else:
            print(f"Error: Model path not found: {path}")
            return None

    except Exception as e:
        print(f"Error loading model {model_config['name']}: {e}")
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
