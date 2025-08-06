
from dotenv import load_dotenv
import os
from typing import List

from knowledge_base.config.settings import get_settings
from openai import OpenAI

# Load the .env file from the root directory
load_dotenv()

# Initialize with settings from settings.py
settings = get_settings()
model = settings.api.embedding_model

class EmbeddingClient:
    def __init__(self, model: str = model):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        self.client = OpenAI(api_key=self.api_key)

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
        input=text,
        model=self.model,
    )
        return response.data[0].embedding


    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        "Get embeddings from a batch of texts"
        response = self.client.embeddings.create(
            input = texts,
            model = self.model
        )
        return [embedding for embedding in response.data]