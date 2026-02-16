import os
import shutil
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# Load Configuration
import json
from diffusers import StableDiffusionPipeline
import torch

CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "text_models": [{"name": "Default", "path": "", "n_ctx": 2048}],
        "image_models": [{"name": "Default", "path": "models", "device": "auto"}],
        "video_models": []
    }

config = load_config()

# Initialize FastAPI
app = FastAPI()

# Database Setup (SQLite)
import sqlite3

DB_PATH = "data/history.db"
OUTPUTS_DIR = "data/outputs"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            last_mode TEXT DEFAULT 'text',
            last_model TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migration: add columns if they don't exist
    try:
        c.execute("ALTER TABLE conversations ADD COLUMN last_mode TEXT DEFAULT 'text'")
    except:
        pass
    try:
        c.execute("ALTER TABLE conversations ADD COLUMN last_model TEXT DEFAULT ''")
    except:
        pass
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            type TEXT DEFAULT 'text', -- 'text', 'image', 'video'
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Llama.cpp Setup
from llama_cpp import Llama

llm = None
current_text_model_name = None

def load_text_model(model_name=None):
    global llm, current_text_model_name
    
    # Get available models
    models = config.get("text_models", [])
    if not models:
        print("Warning: No text models configured.")
        return None

    # Determine which model to load
    selected_model = None
    if model_name:
        for m in models:
            if m["name"] == model_name:
                selected_model = m
                break
    
    if not selected_model:
        selected_model = models[0] # Default to first
    
    # If already loaded, skip
    if llm is not None and current_text_model_name == selected_model["name"]:
        return llm

    # Load the model
    try:
        path = selected_model["path"]
        if os.path.exists(path):
            print(f"Loading Llama model: {selected_model['name']} from {path}")
            # Unload previous if any (Python GC handles it if we drop reference)
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
        print(f"Error loading model {selected_model['name']}: {e}")
        return None

# Initial load
load_text_model()


# Diffusers Setup
pipe = None
current_image_model_name = None

def load_image_model(model_name=None):
    global pipe, current_image_model_name
    
    models = config.get("image_models", [])
    if not models:
         print("Warning: No image models configured.")
         return None
         
    selected_model = None
    if model_name:
        for m in models:
            if m["name"] == model_name:
                selected_model = m
                break
    
    if not selected_model:
        selected_model = models[0]

    if pipe is not None and current_image_model_name == selected_model["name"]:
        return pipe

    try:
        image_model_path = selected_model["path"]
        device_config = selected_model.get("device", "auto")
        
        if not os.path.exists(image_model_path):
             image_model_path = "./models"
        
        if os.path.exists(image_model_path) and (os.path.isdir(image_model_path) and os.listdir(image_model_path)):
            # Determine device first
            use_cuda = device_config == "cuda" or (device_config == "auto" and torch.cuda.is_available())
            
            if use_cuda:
                print(f"Loading Diffusers model: {selected_model['name']} from {image_model_path} (CUDA, float16)")
                new_pipe = StableDiffusionPipeline.from_pretrained(image_model_path, torch_dtype=torch.float16, local_files_only=True)
                new_pipe = new_pipe.to("cuda")
            else:
                print(f"Loading Diffusers model: {selected_model['name']} from {image_model_path} (CPU, float32)")
                new_pipe = StableDiffusionPipeline.from_pretrained(image_model_path, torch_dtype=torch.float32, local_files_only=True)
                new_pipe = new_pipe.to("cpu")
            
            pipe = new_pipe
            current_image_model_name = selected_model["name"]
            return pipe
        else:
            print(f"Warning: Image model directory invalid: {image_model_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load Diffusers pipeline: {e}")
        return None

# Initial load
load_image_model()


# Pydantic Models
class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    model: str = "default" 
    mode: str = "text" # text, image, video

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"

# Routes

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="data/outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/conversations")
async def create_conversation(conv: ConversationCreate):
    conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (conv_id, conv.title))
    conn.commit()
    conn.close()
    return {"id": conv_id, "title": conv.title}

@app.get("/api/conversations")
async def get_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at, last_mode, last_model FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "last_mode": r[3] or "text", "last_model": r[4] or ""} for r in rows]

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    conn.close()
    return {"status": "success", "id": conversation_id}

@app.get("/api/conversations/{conversation_id}")
async def get_messages(conversation_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get conversation metadata
    c.execute("SELECT last_mode, last_model FROM conversations WHERE id = ?", (conversation_id,))
    conv_row = c.fetchone()
    last_mode = conv_row[0] if conv_row else "text"
    last_model = conv_row[1] if conv_row else ""
    
    c.execute("SELECT role, content, type FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return {
        "messages": [{"role": r[0], "content": r[1], "type": r[2]} for r in rows],
        "last_mode": last_mode or "text",
        "last_model": last_model or ""
    }

@app.get("/api/models")
async def get_models(type: str = "text"):
    if type == "image":
        models = config.get("image_models", [])
    elif type == "video":
        models = config.get("video_models", [])
    else:
        models = config.get("text_models", [])
    return [m["name"] for m in models]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Save user message
    c.execute("INSERT INTO messages (conversation_id, role, content, type) VALUES (?, ?, ?, ?)",
              (request.conversation_id, "user", request.message, "text"))
    # Save last used mode and model
    c.execute("UPDATE conversations SET last_mode = ?, last_model = ? WHERE id = ?",
              (request.mode, request.model, request.conversation_id))
    conn.commit()

    response_content = ""
    response_type = "text"

    if request.mode == "image":
        # Image Generation
        global pipe
        pipe = load_image_model(request.model)

        if pipe is None:
             # Try to reload or just error
             raise HTTPException(status_code=500, detail="Image generation model not loaded. Check server logs.")
        
        prompt = request.message.strip()
        try:
            image = pipe(prompt).images[0]
            filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(OUTPUTS_DIR, filename)
            image.save(image_path)
            
            response_content = f"/outputs/{filename}"
            response_type = "image"
        except Exception as e:
             response_content = f"Error generating image: {str(e)}"
             response_type = "text" 
    
    elif request.mode == "video":
        response_content = "Video generation is not yet implemented."
        response_type = "text"

    else:
        # Text Generation
        # Ensure correct model is loaded
        global llm
        llm = load_text_model(request.model)

        if llm is None:
             response_content = "Error: Llama model not loaded. Please check config.json and models directory."
             response_type = "text"
        else:
            # Retrieve context (last few messages)
            c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (request.conversation_id,))
            history_rows = c.fetchall()
            
            messages = [{"role": r[0], "content": r[1]} for r in history_rows]
            
            try:
                response = llm.create_chat_completion(
                    messages=messages
                )
                response_content = response['choices'][0]['message']['content']
                response_type = "text"
            except Exception as e:
                response_content = f"Error communicating with Llama: {str(e)}"
                response_type = "text"
    
    # Save assistant response
    c.execute("INSERT INTO messages (conversation_id, role, content, type) VALUES (?, ?, ?, ?)",
              (request.conversation_id, "assistant", response_content, response_type))
    conn.commit()

    # Auto-Titling Logic
    new_title = None
    c.execute("SELECT title FROM conversations WHERE id = ?", (request.conversation_id,))
    row = c.fetchone()
    current_title = row[0] if row else "New Chat"

    if current_title == "New Chat":
        if response_type == "text" and llm:
            # Use LLM to generate a title for text chats
            try:
                title_prompt = [
                    {"role": "user", "content": f"Generate a short, concise title (max 5 words) for this chat conversation based on the following exchange:\nUser: {request.message}\nAI: {response_content}\nTitle:"}
                ]
                title_response = llm.create_chat_completion(messages=title_prompt, max_tokens=10)
                generated_title = title_response['choices'][0]['message']['content'].strip().strip('"')
                
                if generated_title:
                    c.execute("UPDATE conversations SET title = ? WHERE id = ?", (generated_title, request.conversation_id))
                    conn.commit()
                    new_title = generated_title
            except Exception as e:
                print(f"Error generating title: {e}")
        else:
            # For image/video modes, use the user's prompt as the title
            generated_title = request.message[:50].strip()
            if len(request.message) > 50:
                generated_title += "..."
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (generated_title, request.conversation_id))
            conn.commit()
            new_title = generated_title

    conn.close()

    return {"role": "assistant", "content": response_content, "type": response_type, "new_title": new_title}

if __name__ == "__main__":
    import uvicorn
    # Trigger reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
