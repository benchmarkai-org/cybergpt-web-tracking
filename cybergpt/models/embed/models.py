from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from tqdm import tqdm
import os
import openai
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import dotenv

dotenv.load_dotenv()


SENTENCE_TRANSFORMERS = {
    "minilm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "e5": SentenceTransformer("intfloat/e5-small-v2"),
}
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        pass


class OpenAIEmbedder(EmbeddingModel):
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str = os.getenv("OPENAI_API_KEY"),
    ):
        self.model = model
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> np.ndarray:
        """Get OpenAI embeddings for multiple texts in batches."""
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)


class SentenceEmbedder(EmbeddingModel):
    def __init__(self, model: str = "minilm"):
        self.model = SENTENCE_TRANSFORMERS[model]

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))


class TfidfEmbedder(EmbeddingModel):
    def __init__(self):
        pass

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()


MODELS = {
    "openai": OpenAIEmbedder(),
    "openai-large": OpenAIEmbedder(model="text-embedding-3-large"),
    "minilm": SentenceEmbedder("minilm"),
    "e5": SentenceEmbedder("e5"),
    "tfidf": TfidfEmbedder(),
}
