import wikipedia
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def fetch_wiki(topic: str) -> str:
    wikipedia.set_lang("en")
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        return page.content[:5000]  # use more than just summary for better coverage
    except wikipedia.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=10)
        except:
            return "No information found"
    except Exception:
        return "No information found"

def chunk_text(text: str, chunk_size: int = 2) -> list[str]:
    sentences = text.split(". ")
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

class VectorSearch:
    def __init__(self, texts: list[str]):
        self.texts = texts
        model = get_embed_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query: str, k: int = 2) -> list[str]:
        model = get_embed_model()
        query_emb = model.encode([query], show_progress_bar=False)
        _, indices = self.index.search(np.array(query_emb).astype("float32"), k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]