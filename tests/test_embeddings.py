from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors

from core.errors import PermanentProviderError, TransientProviderError
from knowledge.chunking import Chunk
from knowledge.embeddings import (
    EmbeddingCache,
    FakeEmbedder,
    GeminiEmbedder,
    doc_text,
    make_embedder,
    query_text,
)
from tests.conftest import read_events

ROOT = Path(__file__).resolve().parents[1]


def chunk(texto: str, **metadatos) -> Chunk:
    return Chunk(f"doc#{len(texto)}", texto, metadatos)


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1 - sum(x * y for x, y in zip(a, b, strict=True))


class FakeGenaiClient:
    """Imita `client.models.embed_content`: regresa (o lanza) lo de `script` en orden."""

    def __init__(self, dim: int = 4, script: list | None = None, norm: float = 1.0) -> None:
        self.dim = dim
        self.script = list(script or [])
        self.norm = norm
        self.calls: list[dict] = []
        self.models = self

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
        value = self.norm / math.sqrt(self.dim)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[value] * self.dim) for _ in contents]
        )


@pytest.fixture
def embedder_factory(tmp_path, logger, fake_sleep):
    def factory(client, **kwargs) -> GeminiEmbedder:
        return GeminiEmbedder(
            dim=client.dim, client=client, logger=logger, sleep=fake_sleep, **kwargs
        )

    return factory


def test_doc_text_prefixes():
    assert doc_text(
        chunk("Texto", titulo="Escalamiento", seccion="Escalamiento > Prioridad alta")
    ) == ("title: Escalamiento > Escalamiento > Prioridad alta | text: Texto")
    assert doc_text(chunk("Texto", titulo="Contrato de servicio PagosRed")) == (
        "title: Contrato de servicio PagosRed | text: Texto"
    )
    assert doc_text(chunk("Texto")) == "title: none | text: Texto"


def test_query_text_prefix():
    assert query_text("¿Cuándo escalo?") == "task: search result | query: ¿Cuándo escalo?"


def test_fake_embedder_is_normalized_and_meaningful():
    fake = FakeEmbedder()
    cola = fake.embed_query("reinicio de la cola de pedidos")
    assert math.isclose(math.sqrt(sum(v * v for v in cola)), 1.0)
    [cercano, lejano] = fake.embed_documents(
        [
            chunk("Detén la cola de pedidos y reinicia el consumidor."),
            chunk("Renueva el certificado."),
        ]
    )
    assert cosine_distance(cola, cercano) < cosine_distance(cola, lejano)


def test_fake_embedder_is_deterministic_across_processes():
    """Otro proceso, con otra semilla de `hash()`, calcula el mismo vector."""
    code = (
        "from knowledge.embeddings import FakeEmbedder; "
        "print(FakeEmbedder(dim=16).embed_query('Prioridad alta y escalamiento'))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    other = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert other == str(FakeEmbedder(dim=16).embed_query("Prioridad alta y escalamiento"))


def test_batches_and_output_dimensionality(embedder_factory):
    client = FakeGenaiClient(dim=4)
    embedder = embedder_factory(client, batch_size=2)

    vectors = embedder.embed_documents([chunk(f"texto {n}", titulo="t") for n in range(5)])

    assert len(vectors) == 5
    assert [len(call["contents"]) for call in client.calls] == [2, 2, 1]
    assert all(call["config"].output_dimensionality == 4 for call in client.calls)
    assert client.calls[0]["contents"][0].parts[0].text == "title: t | text: texto 0"
    assert client.calls[0]["model"] == "gemini-embedding-2"


def test_vectors_are_normalized_if_needed(embedder_factory):
    vector = embedder_factory(FakeGenaiClient(dim=4, norm=2.0)).embed_query("hola")
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


def test_cache_avoids_second_call(tmp_path, embedder_factory):
    client = FakeGenaiClient(dim=4)
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    chunks = [chunk("uno", titulo="t"), chunk("dos", titulo="t")]

    first = embedder_factory(client, cache=cache).embed_documents(chunks)
    second = embedder_factory(client, cache=cache).embed_documents(chunks)

    assert first == second
    assert len(client.calls) == 1


def test_transient_error_is_retried(embedder_factory, fake_sleep, log_path):
    unavailable = errors.ServerError(503, {"error": {"message": "sobrecargado"}})
    client = FakeGenaiClient(dim=4, script=[unavailable])

    vector = embedder_factory(client).embed_query("hola")

    assert len(vector) == 4
    assert len(client.calls) == 2
    assert len(fake_sleep.waits) == 1 and fake_sleep.waits[0] < 2
    events = read_events(log_path)
    assert [e["success"] for e in events] == [False, True]
    assert events[-1]["embed_model"] == "gemini-embedding-2" and events[-1]["embed_batch"] == 1


def test_quota_error_waits_longer_and_explains(embedder_factory, fake_sleep):
    quota = errors.ClientError(429, {"error": {"message": "quota"}})
    client = FakeGenaiClient(dim=4, script=[quota] * 4)

    with pytest.raises(TransientProviderError, match="cuota de embeddings agotada"):
        embedder_factory(client).embed_query("hola")

    assert len(client.calls) == 4
    assert all(wait >= 10 for wait in fake_sleep.waits)


def test_validation_error_fails_immediately(embedder_factory, fake_sleep):
    invalid = errors.ClientError(400, {"error": {"message": "dimensión inválida"}})
    client = FakeGenaiClient(dim=4, script=[invalid])

    with pytest.raises(PermanentProviderError):
        embedder_factory(client).embed_query("hola")

    assert len(client.calls) == 1
    assert fake_sleep.waits == []


def test_make_embedder():
    assert isinstance(make_embedder("fake"), FakeEmbedder)
    with pytest.raises(ValueError, match="embedder desconocido"):
        make_embedder("otro")
