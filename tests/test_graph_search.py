from __future__ import annotations

import networkx as nx
import pytest

from knowledge.graph import KnowledgeGraph
from knowledge.graph_build import build_graph, load_chunks
from knowledge.graph_search import hechos, ppr, rank_chunks
from knowledge.retrieval import Retriever
from knowledge.store import Hit
from scripts.eval_retrieval import ROOT
from tests.test_rewrite import StubStore

M1 = "El T-1043 lleva 3 horas sin respuesta del proveedor, ¿ya incumplió su contrato?"


@pytest.fixture(scope="module")
def kg() -> KnowledgeGraph:
    return build_graph(load_chunks(ROOT / "data" / "docs"))


def fuentes(ranking: list[tuple[str, float]]) -> list[str]:
    return [chunk_id.split("#")[0] for chunk_id, _ in ranking]


def test_ppr_sums_to_one_and_fades_with_distance():
    p = ppr(nx.path_graph(["a", "b", "c", "d"]), ["a"])
    assert sum(p.values()) == pytest.approx(1)
    assert p["a"] >= 0.15  # al menos 1 − α vuelve a la semilla en cada paso
    assert p["b"] > p["c"] > p["d"]


def test_the_node_between_two_seeds_collects_more():
    g = nx.Graph([("a", "x"), ("x", "b"), ("a", "y"), ("b", "z")])
    p = ppr(g, ["a", "b"])
    assert p["x"] > p["y"] == pytest.approx(p["z"])


def test_an_isolated_seed_keeps_all_the_probability():
    g = nx.Graph()
    g.add_node("s")
    assert ppr(g, ["s"]) == {"s": pytest.approx(1)}


def test_a_ticket_reaches_its_provider_contract_in_two_hops(kg):
    top5 = fuentes(rank_chunks(kg, M1, k=5))
    assert top5[0] == "runbooks/portal-proveedores.md"
    assert "contratos/nubehost.pdf" in top5
    assert max(top5.count(fuente) for fuente in top5) <= 2  # tope por documento


def test_without_entities_there_is_nothing_to_walk(kg):
    assert rank_chunks(kg, "¿Cómo restablezco mi contraseña?") == []


def test_where_filters_the_graph_like_the_other_searches(kg):
    contratos = fuentes(rank_chunks(kg, M1, where={"tipo_doc": "contrato"}))
    assert contratos[0] == "contratos/nubehost.pdf"
    assert all(fuente.startswith("contratos/") for fuente in contratos)


def test_facts_start_at_the_seed_and_cite_their_source(kg):
    primeros = hechos(kg, "¿Qué pasa con el T-1043?")[:2]
    assert "T-1043 depende_de NubeHost [tickets_db.json]" in primeros


def test_the_retriever_uses_the_graph_as_one_more_ranking():
    kg = KnowledgeGraph()
    ticket = kg.add_entidad("ticket", "T-1043")
    proveedor = kg.add_entidad("proveedor", "NubeHost")
    kg.add_relacion(ticket, "depende_de", proveedor, "tickets_db.json")
    for chunk_id, tipo in (("runbook.md#0", "runbook"), ("contrato.pdf#0", "contrato")):
        kg.add_entidad("chunk", chunk_id, metadatos={"tipo_doc": tipo}, ref=chunk_id)
    kg.add_relacion("chunk:runbook.md#0", "menciona", ticket, "runbook.md#0")
    kg.add_relacion("chunk:contrato.pdf#0", "contrato_de", proveedor, "contrato.pdf#0")
    store = StubStore(
        [Hit(c, "...", {"fuente": c}, 0.5) for c in ("runbook.md#0", "contrato.pdf#0")]
    )

    hits = Retriever(store, modo="grafo", grafo=kg).search("¿Y el T-1043?", k=5)

    assert [h.chunk_id for h in hits] == ["runbook.md#0", "contrato.pdf#0"]
    assert store.consultas == []  # el modo grafo no hace búsqueda vectorial
    with pytest.raises(ValueError, match="necesita un KnowledgeGraph"):
        Retriever(store, modo="grafo")
