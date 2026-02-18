"""
===========================================================================
test_app.py — Unit Tests for the Local AI Studio Application
===========================================================================

PURPOSE:
    This file tests the API endpoints to make sure they work correctly.
    We use "mocks" to simulate AI models so tests run fast without
    needing actual model files.

    python test_app.py

WHAT IS MOCKING?
    Mocking means creating fake versions of things (like AI models)
    so we can test our code without needing the real dependencies.
    For example, instead of loading a 4GB AI model, we create a fake
    one that returns pre-defined answers.

    We also mock 'web_search' to avoid real network calls during tests.
===========================================================================
"""



import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set testing environment variable BEFORE importing main
os.environ["TESTING"] = "1"

# Add current directory to sys.path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Mock external libraries BEFORE importing our code
# ---------------------------------------------------------------------------
# We need to mock these BEFORE "import main" because main.py imports
# modules that try to load real AI models. By mocking first, we prevent
# that from happening during tests.

# Mock the diffusers library (Stable Diffusion)
mock_diffusers = MagicMock()
mock_pipe_cls = MagicMock()
mock_diffusers.StableDiffusionPipeline = mock_pipe_cls
sys.modules["diffusers"] = mock_diffusers

# Mock the ollama library
mock_ollama = MagicMock()
sys.modules["ollama"] = mock_ollama

# Mock the llama_cpp library
mock_llama_cpp = MagicMock()
mock_llama_cls = MagicMock()
mock_llama_cpp.Llama = mock_llama_cls
sys.modules["llama_cpp"] = mock_llama_cpp

# Configure the Stable Diffusion mock pipeline
mock_pipeline_instance = MagicMock()
mock_pipeline_instance.to.return_value = mock_pipeline_instance
mock_pipe_cls.from_pretrained.return_value = mock_pipeline_instance

# Configure the Llama mock model
mock_llm_instance = MagicMock()
mock_llama_cls.return_value = mock_llm_instance

# Now it's safe to import our app
from main import app

# Create a test client (simulates HTTP requests without a real server)
client = TestClient(app)


class TestLocalAIStudio(unittest.TestCase):

    def setUp(self):
        """Runs before each test — setup any test data here."""
        pass

    def test_read_root(self):
        """Test that the homepage loads successfully."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Local AI Studio", response.text)

    def test_create_conversation(self):
        """Test creating a new conversation."""
        response = client.post("/api/conversations", json={"title": "Test Chat"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["title"], "Test Chat")
        return data["id"]

    def test_get_conversations(self):
        """Test listing all conversations."""
        self.test_create_conversation()
        response = client.get("/api/conversations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_get_messages(self):
        """Test getting messages for a conversation."""
        conv_id = self.test_create_conversation()
        response = client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_chat_text(self):
        print("\n--- Running test_chat_text ---")
        """Test sending a text message to the chat API."""
        mock_response = {'choices': [{'message': {'content': 'Hello from mocked Llama!'}}]}

        # Patch load_text_model where it's used (in chat_routes)
        with patch("routes.chat_routes.load_text_model") as mock_load:
            # Configure the mock LLM
            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = mock_response
            mock_load.return_value = mock_llm

            conv_id = self.test_create_conversation()
            response = client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": "Hello AI",
                "model": "gemma-2b"
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["role"], "assistant")
            self.assertEqual(data["content"], 'Hello from mocked Llama!')
            self.assertEqual(data["type"], "text")

    def test_chat_with_web_search(self):
        print("\n--- Running test_chat_with_web_search ---")
        """Test sending a message with web search enabled."""
        mock_response = {'choices': [{'message': {'content': 'Here is the answer based on web results.'}}]}
        
        # Mock web search results
        mock_web_results = [
            {"title": "Test Result", "href": "http://example.com", "body": "This is a test snippet."}
        ]

        # Patch:
        # 1. load_text_model (in chat_routes) -> returns our mock LLM
        # 2. search_web (in chat_routes) -> returns our mock results
        
        with patch("routes.chat_routes.load_text_model") as mock_load:
            # Configure the mock LLM
            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = mock_response
            mock_load.return_value = mock_llm

            with patch("routes.chat_routes.search_web", return_value=mock_web_results) as mock_search:
                
                conv_id = self.test_create_conversation()
                response = client.post("/api/chat", json={
                    "conversation_id": conv_id,
                    "message": "What is the weather?",
                    "web_search": True
                })
                
                # Check that web search was called
                mock_search.assert_called_once()
                self.assertEqual(mock_search.call_args[0][0], "What is the weather?")

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["content"], 'Here is the answer based on web results.')


    def test_chat_image(self):
        print("\n--- Running test_chat_image ---")
        """Test generating an image via chat."""
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        
        # We need the pipe to return an object with an 'images' attribute
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.return_value.images = [mock_image]

        # Patch load_image_model where it's used (in chat_routes)
        with patch("routes.chat_routes.load_image_model") as mock_load:
            mock_load.return_value = mock_pipe_instance

            # Also patch models_loader.pipe because _generate_image_response uses it
            with patch("models_loader.pipe", mock_pipe_instance):
                conv_id = self.test_create_conversation()
                response = client.post("/api/chat", json={
                    "conversation_id": conv_id,
                    "message": "/image a cat",
                    "mode": "image"
                })

                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["type"], "image")
                self.assertTrue(data["content"].startswith("/outputs/"))


if __name__ == "__main__":
    unittest.main()
