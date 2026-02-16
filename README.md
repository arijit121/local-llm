# Local AI Studio

A ChatGPT-like local AI interface combining Ollama for text generation and Diffusers for image generation.

## Features

- **Text Generation**: Powered by llama.cpp (GGUF models).
- **Image Generation**: Powered by Hugging Face Diffusers (Stable Diffusion).
- **Persistent History**: Chat history stored in a local SQLite database.
- **Modern UI**: ChatGPT-style interface with dark mode and markdown support.

## Prerequisites

- Python 3.8+
- **GGUF Model**: A `.gguf` model file (e.g., Llama 3) placed in `local-ai-studio/models`.
- (Optional) NVIDIA GPU for faster image generation.

## Setup

1.  **Run the setup script**:
    Double-click `setup.bat` or run:
    ```bash
    setup.bat
    ```
    This will create a virtual environment and install all dependencies.

2.  **Download GGUF Model**:
    - Download a GGUF model (e.g., [Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)) from Hugging Face.
    - Save the `.gguf` file into `local-ai-studio/models`.

3.  **Download Diffusers Model**:
    The application requires a Stable Diffusion model to be present in the `models` directory.
    - Download a model (e.g., [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)) from Hugging Face.
    - Save the model files into `local-ai-studio/models`. The directory should contain `model_index.json`, `vae`, `unet`, etc.

## Configuration

You can configure the model paths and execution devices in `config.json`:

```json
{
    "text_model": {
        "path": "models/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf",
        "n_ctx": 2048,
        "n_gpu_layers": -1
    },
    "image_model": {
        "path": "models",
        "device": "auto" 
    }
}
```

- **text_model**: Path to your `.gguf` file. `n_gpu_layers` set to -1 offloads all layers to GPU.
- **image_model**: Path to Diffusers model directory. `device` can be "auto", "cuda", or "cpu".

## Running the Application

1.  Activate the virtual environment:
    ```bash
    venv\Scripts\activate
    ```

2.  Start the server:
    ```bash
    uvicorn main:app --reload
    ```

3.  Open your browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

## Usage

- **Text Chat**: Just type your message.
- **Image Generation**: Type `/image <prompt>` or "Generate an image of <prompt>".

## Project Structure

- `main.py`: FastAPI backend application.
- `templates/index.html`: Main frontend UI.
- `static/`: CSS and JavaScript files.
- `data/history.db`: SQLite database for chat history.
- `data/outputs/`: Generated images directory.
