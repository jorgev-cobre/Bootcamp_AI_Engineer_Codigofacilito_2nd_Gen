from __future__ import annotations

from knowledge.chunking import Chunk
from knowledge.graph import KnowledgeGraph
from knowledge.graph_build import add_chunk, build_graph, from_datos, load_chunks, menciones
from scripts.build_graph import main as build_main
from scripts.eval_retrieval import ROOT

TICKETS = [
    {"id": "T-1043", "asunto": "Portal caído", "prioridad": "alta", "estado": "abierto",
     "equipo": "TI", "proveedor": "NubeHost"},
    {"id": "T-1022", "asunto": "Doble captura", "prioridad": "baja", "estado": "abierto",
     "equipo": "TI", "proveedor": None},
]  # fmt: skip
PROVEEDORES = [{"nombre": "NubeHost", "servicio": "Hospedaje del portal", "sla_horas": 2}]


def datos() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    from_datos(kg, TICKETS, PROVEEDORES)
    return kg


def test_structured_data_links_tickets_to_providers_and_teams():
    kg = datos()
    assert kg.g.has_edge("ticket:T-1043", "proveedor:NubeHost", key="depende_de")
    assert kg.g.has_edge("ticket:T-1022", "equipo:TI", key="atendido_por")
    assert list(kg.g.successors("ticket:T-1022")) == ["equipo:TI"]  # sin proveedor
    assert kg.entidades("guardia") == ["guardia:N1", "guardia:N2", "guardia:N3"]


def test_mentions_find_ids_names_and_guards_but_not_unknown_tickets():
    texto = "El T-1043 sigue caído; T-9999 no existe. Avisa a la Guardia N2 y a nubehost (TI)."
    assert menciones(datos(), texto) == [
        "ticket:T-1043",
        "guardia:N2",
        "proveedor:NubeHost",
        "equipo:TI",
    ]


def test_contract_pages_belong_to_their_provider_even_without_its_name():
    kg = datos()
    metadatos = {"fuente": "contratos/nubehost.pdf", "pagina": 2, "proveedor": "NubeHost"}
    nodo = add_chunk(kg, Chunk("contratos/nubehost.pdf#1", "El Proveedor responde.", metadatos))

    assert kg.g.nodes[nodo]["ref"] == "contratos/nubehost.pdf · p. 2"
    assert kg.g.has_edge(nodo, "proveedor:NubeHost", key="contrato_de")
    assert not kg.g.has_edge(nodo, "proveedor:NubeHost", key="menciona")


def test_builds_the_whole_corpus_without_llm(tmp_path, capsys):
    kg = build_graph(load_chunks(ROOT / "data" / "docs"))
    resumen = kg.resumen()
    assert (resumen["ticket"], resumen["proveedor"], resumen["chunk"]) == (50, 6, 54)
    portal = "chunk:runbooks/portal-proveedores.md#0"
    assert {"ticket:T-1043", "proveedor:NubeHost"} <= set(kg.g.successors(portal))

    out = tmp_path / "grafo.json"
    assert build_main(["--out", str(out)]) == 0
    assert out.exists() and "54 chunk" in capsys.readouterr().out
