"""Embeddings: texto → vector, con Gemini o con un embedder falso para tests y sin cuota.

`gemini-embedding-2` no usa `task_type`: la tarea va en el texto, con los prefijos de
`doc_text` y `query_text`. Cada vector se guarda en una caché SQLite por modelo,
dimensión y texto: un chunk que no cambió no se vuelve a pagar.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx

from core.errors import PermanentProviderError, ProviderError, TransientProviderError

if TYPE_CHECKING:
    from core.logger import CallLogger
    from knowledge.chunking import Chunk

CACHE_PATH = Path(".cache") / "embeddings.sqlite"
QUOTA_BACKOFF_S = 10  # Cuota agotada (429): esperas de 10 s, 20 s, 40 s.
NORM_TOLERANCE = 1e-3
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}

_WORD = re.compile(r"\w+")
# Palabras que no ayudan a distinguir documentos en el embedder falso.
_STOPWORDS = frozenset(
    "que con por para los las del una uno unos unas como más pero sus este esta esto ese esa "
    "son está hay muy sin sobre entre cuando también desde hasta donde cada qué cómo cuál "
    "cuándo dónde quién".split()
)


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]: ...

    def embed_query(self, pregunta: str) -> list[float]: ...


def doc_text(chunk: Chunk) -> str:
    """Texto que se embebe por chunk (S30): `title: {titulo} > {seccion} | text: {texto}`."""
    titulo = chunk.metadatos.get("titulo")
    seccion = chunk.metadatos.get("seccion")
    if not titulo:
        title = "none"
    elif seccion:
        title = f"{titulo} > {seccion}"
    else:
        title = titulo
    return f"title: {title} | text: {chunk.texto}"


def query_text(pregunta: str) -> str:
    """Texto que se embebe por pregunta (S30)."""
    return f"task: search result | query: {pregunta}"


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class EmbeddingCache:
    """Caché SQLite: `sha256(model|dim|texto_con_prefijo)` → vector."""

    def __init__(self, path: Path | str = CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def key(model: str, dim: int, texto: str) -> str:
        return hashlib.sha256(f"{model}|{dim}|{texto}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        found: dict[str, list[float]] = {}
        with self._connect() as db:
            for start in range(0, len(keys), 500):  # Límite de variables de SQLite.
                part = keys[start : start + 500]
                marks = ",".join("?" * len(part))
                rows = db.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", part
                )
                found.update({key: json.loads(vector) for key, vector in rows})
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(key, json.dumps(vector)) for key, vector in items.items()],
            )


class GeminiEmbedder:
    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dim: int = 768,
        batch_size: int = 32,
        cache: EmbeddingCache | None = None,
        *,
        client=None,
        logger: CallLogger | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.cache = cache
        self.max_retries = max_retries
        self._client = client
        self._logger = logger
        self._sleep = sleep

    def _setup(self) -> None:
        """Crea el cliente y el logger con la llave y el log de la Clase 1 (`.env`)."""
        if self._client is not None and self._logger is not None:
            return
        from google import genai

        from core.config import Settings
        from core.logger import CallLogger

        settings = Settings(primary="gemini", fallbacks=[])  # Falla claro si no hay llave.
        if self._client is None:
            self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        if self._logger is None:
            self._logger = CallLogger(settings.log_path)

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        return self._embed_texts([doc_text(chunk) for chunk in chunks])

    def embed_query(self, pregunta: str) -> list[float]:
        return self._embed_texts([query_text(pregunta)])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        keys = [EmbeddingCache.key(self.model, self.dim, t) for t in texts]
        found = self.cache.get_many(keys) if self.cache else {}
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in found))
        new: dict[str, list[float]] = {}
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            for texto, vector in zip(batch, self._embed_batch(batch), strict=True):
                new[EmbeddingCache.key(self.model, self.dim, texto)] = vector
        if self.cache and new:
            self.cache.put_many(new)
        found.update(new)
        return [found[k] for k in keys]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Una llamada por lote. Cada texto va en su propio `Content`: con una lista de
        strings, `gemini-embedding-2` regresa un solo embedding agregado."""
        from google.genai import types

        self._setup()
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.dim)
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as e:
                error = _classify(e)
                self._log(texts, start, success=False, error_type=type(error).__name__)
                if isinstance(error, TransientProviderError) and attempt < self.max_retries:
                    base = QUOTA_BACKOFF_S if error.status_code == 429 else 1
                    self._sleep(base * 2**attempt + random.uniform(0, 0.25))
                    continue
                raise error from e
            vectors = [list(embedding.values) for embedding in response.embeddings]
            if len(vectors) != len(texts):
                raise PermanentProviderError(
                    "gemini", f"se pidieron {len(texts)} embeddings y llegaron {len(vectors)}"
                )
            vectors = [
                v if abs(math.sqrt(sum(x * x for x in v)) - 1) <= NORM_TOLERANCE else _normalize(v)
                for v in vectors
            ]
            self._log(texts, start, success=True, error_type=None)
            return vectors
        raise AssertionError("inalcanzable")  # pragma: no cover

    def _log(
        self, texts: list[str], start: float, *, success: bool, error_type: str | None
    ) -> None:
        self._logger.log(
            provider="gemini",
            model=self.model,
            tokens_in=0,  # La respuesta de embed_content no trae conteo de tokens.
            tokens_out=0,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
            fallback=False,
            success=success,
            error_type=error_type,
            prompt_chars=sum(len(t) for t in texts),
            embed_model=self.model,
            embed_batch=len(texts),
        )


def _classify(error: Exception) -> ProviderError:
    """Taxonomía de la Clase 1: transitorio (se reintenta) o permanente (falla ya)."""
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, httpx.TransportError):
        return TransientProviderError("gemini", f"error de red: {error}")
    code = getattr(error, "code", None)
    if code == 429:
        return TransientProviderError(
            "gemini",
            "cuota de embeddings agotada; espera un minuto o usa --embedder fake",
            429,
        )
    if code in TRANSIENT_STATUS:
        return TransientProviderError("gemini", str(error), code)
    return PermanentProviderError("gemini", str(error), code if isinstance(code, int) else None)


def _words(texto: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [w for w in _WORD.findall(plain) if len(w) > 2 and w not in _STOPWORDS]


class FakeEmbedder:
    """Embedder determinista y sin red: bolsa de palabras con `hashlib`, normalizada.

    Textos que comparten palabras quedan más cerca. No usa `hash()` de Python, que cambia
    entre procesos.
    """

    model = "fake"

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _vector(self, texto: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _words(texto):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return _normalize(vector)

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        return [
            self._vector(
                " ".join(
                    [
                        chunk.metadatos.get("titulo", ""),
                        chunk.metadatos.get("seccion", ""),
                        chunk.texto,
                    ]
                )
            )
            for chunk in chunks
        ]

    def embed_query(self, pregunta: str) -> list[float]:
        return self._vector(pregunta)


def make_embedder(name: str) -> EmbeddingProvider:
    """`"gemini"` (con caché en .cache/) o `"fake"`."""
    if name == "gemini":
        return GeminiEmbedder(cache=EmbeddingCache())
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"embedder desconocido: {name!r}. Usa gemini o fake.")
