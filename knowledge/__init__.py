"""Conocimiento: ingesta, chunking, embeddings, índice vectorial y RAG (Clase 6); búsqueda
híbrida, reescritura y re-ranking (Clase 7)."""

from knowledge.chunking import (
    Chunk,
    chunk_document,
    chunk_file,
    chunk_fixed,
    chunk_markdown,
    chunk_recursive,
    split_por_encabezados,
    split_recursivo,
)
from knowledge.embeddings import (
    EmbeddingCache,
    EmbeddingProvider,
    FakeEmbedder,
    GeminiEmbedder,
    doc_text,
    make_embedder,
    query_text,
)
from knowledge.ingest import Document, IngestError, clean_text, load_dir, load_path
from knowledge.lexical import BM25Index, LexHit, tokenize
from knowledge.rag import UMBRAL, CitaInvalida, RagAnswer, RagResult, answer, rag_messages
from knowledge.rerank import Reranking, rerank
from knowledge.retrieval import MODOS, Retriever, rrf
from knowledge.rewrite import Reescritura, rewrite_query
from knowledge.store import EmbeddingMismatch, Hit, VectorStore, build_where, open_store

__all__ = [
    "MODOS",
    "UMBRAL",
    "BM25Index",
    "Chunk",
    "CitaInvalida",
    "Document",
    "EmbeddingCache",
    "EmbeddingMismatch",
    "EmbeddingProvider",
    "FakeEmbedder",
    "GeminiEmbedder",
    "Hit",
    "IngestError",
    "LexHit",
    "RagAnswer",
    "RagResult",
    "Reescritura",
    "Reranking",
    "Retriever",
    "VectorStore",
    "answer",
    "build_where",
    "chunk_document",
    "chunk_file",
    "chunk_fixed",
    "chunk_markdown",
    "chunk_recursive",
    "clean_text",
    "doc_text",
    "load_dir",
    "load_path",
    "make_embedder",
    "open_store",
    "query_text",
    "rag_messages",
    "rerank",
    "rewrite_query",
    "rrf",
    "split_por_encabezados",
    "split_recursivo",
    "tokenize",
]
