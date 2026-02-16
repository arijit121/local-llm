import os
from llama_cpp import Llama

models_to_test = [
    "models/Aramis-2B-BitNet-b1.58-i2s-GGUF/aramis-ggml-model-i2_s.gguf",
    "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
]

for model_path in models_to_test:
    abs_path = os.path.abspath(model_path)
    print(f"\n--- Testing model: {model_path} ---")
    print(f"File exists: {os.path.exists(abs_path)}")
    if os.path.exists(abs_path):
        print(f"File size: {os.path.getsize(abs_path)}")

    try:
        print("Attempting to load Llama model with verbose=True...")
        llm = Llama(
            model_path=abs_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=True
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
