"""The "R" in RAG: chunk the policies, embed them, store them, search them.

Flow:
    policy .md files -> chunks (one per paragraph) -> embedding vectors
    -> Qdrant vector index (each point keeps its source metadata)
    question -> embedding -> nearest chunks + cosine similarity scores

Both the baseline RAG system and the hybrid RAG + LoRA system use the SAME
Retriever object, so any difference between them comes from the model only.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from .config import Settings
from .data import PolicyDocument, load_documents

COLLECTION = "northstar_policies"


@dataclass
class Chunk:
    chunk_id: str  # e.g. "shipping-1"
    document_id: str  # e.g. "shipping" -> this is what answers cite
    title: str  # e.g. "Shipping Policy"
    position: int  # 0 = first paragraph of the document
    text: str


@dataclass
class SearchHit(Chunk):
    score: float  # cosine similarity: 1.0 = same meaning, ~0 = unrelated


def chunk_documents(documents: list[PolicyDocument]) -> list[Chunk]:
    """Split each policy into paragraphs.

    The policies are short and each paragraph covers one rule, so a paragraph
    is a natural, self-contained unit of evidence.
    """

    chunks = []
    for document in documents:
        paragraphs = [p.replace("\n", " ").strip() for p in document.text.split("\n\n")]
        for position, paragraph in enumerate(p for p in paragraphs if p):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{position}",
                    document_id=document.document_id,
                    title=document.title,
                    position=position,
                    text=paragraph,
                )
            )
    return chunks


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Real semantic embeddings: similar meaning -> nearby vectors."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        # Imported here so the fast tests never load this heavy library.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, revision=revision)
        self.dimension = self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes every vector length 1, so the dot
        # product equals the cosine similarity.
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class HashEmbedder:
    """TEST-ONLY bag-of-words embedder: no download, fully deterministic.

    It only matches shared words, not meaning, so it must never be used for
    the submitted evaluation.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                matrix[row, bucket] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class Retriever:
    """Builds the Qdrant index once and answers nearest-neighbour queries."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_revision
        )
        self.chunks = chunk_documents(load_documents(settings.knowledge_base_dir))

        # ":memory:" = Qdrant runs inside this Python process; nothing to install
        # or start. The index is rebuilt (in about a second) on every run.
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        # The title is embedded with the text to give short paragraphs more topic signal.
        vectors = self.embedder.encode([f"{c.title}\n{c.text}" for c in self.chunks])
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                # The "payload" is the metadata stored next to each vector. This
                # is how the source ID survives into the search results.
                models.PointStruct(id=index, vector=vector.tolist(), payload=asdict(chunk))
                for index, (chunk, vector) in enumerate(zip(self.chunks, vectors, strict=True))
            ],
        )

    def search(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the top-k chunks, best first, WITHOUT applying the threshold."""

        query_vector = self.embedder.encode([question])[0].tolist()
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k or self.settings.top_k,
            with_payload=True,
        )
        return [SearchHit(**point.payload, score=float(point.score)) for point in response.points]

    def evidence_for(self, question: str) -> tuple[list[SearchHit], list[SearchHit]]:
        """Return (all top-k hits, hits that pass the evidence threshold).

        If the second list is empty the question is treated as unsupported and
        the systems must abstain instead of using the nearest weak match.
        """

        hits = self.search(question)
        return hits, [hit for hit in hits if hit.score >= self.settings.min_score]
