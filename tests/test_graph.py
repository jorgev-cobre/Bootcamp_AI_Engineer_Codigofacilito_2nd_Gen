from __future__ import annotations

import pytest

from knowledge.graph import KnowledgeGraph, clave, nodo_id


def small_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    ticket = kg.add_entidad("ticket", "T-1043", prioridad="alta")
    proveedor = kg.add_entidad("proveedor", "NubeHost")
    kg.add_relacion(ticket, "depende_de", proveedor, "tickets_db.json")
    return kg


def test_node_ids_carry_the_type():
    assert nodo_id("ticket", "T-1043") == "ticket:T-1043"
    assert small_graph().entidades("proveedor") == ["proveedor:NubeHost"]


def test_names_compare_without_case_accents_or_stopwords():
    kg = small_graph()
    kg.add_entidad("equipo", "Logística")
    assert clave("Transportes Sierra") == " transportes sierra "
    assert kg.buscar("equipo", "logistica") == "equipo:Logística"
    assert kg.buscar("proveedor", "nubehost") == "proveedor:NubeHost"
    assert kg.buscar("proveedor", "PagosRed") is None


def test_adding_an_existing_entity_merges_its_attributes():
    kg = small_graph()
    nodo = kg.add_entidad("ticket", "T-1043", estado="en revisión")
    assert kg.g.nodes[nodo] == {
        "tipo": "ticket",
        "nombre": "T-1043",
        "prioridad": "alta",
        "estado": "en revisión",
    }
    assert kg.resumen()["ticket"] == 1


def test_the_same_fact_from_two_sources_is_one_edge():
    kg = small_graph()
    origen, destino = "ticket:T-1043", "proveedor:NubeHost"
    kg.add_relacion(origen, "depende_de", destino, "runbooks/portal-proveedores.md#0")
    kg.add_relacion(origen, "depende_de", destino, "tickets_db.json")

    assert kg.g.number_of_edges() == 1
    assert kg.g.edges[origen, destino, "depende_de"]["fuentes"] == [
        "tickets_db.json",
        "runbooks/portal-proveedores.md#0",
    ]


def test_unknown_types_and_nodes_are_rejected():
    kg = small_graph()
    with pytest.raises(ValueError, match="tipo desconocido"):
        kg.add_entidad("cliente", "Acme")
    with pytest.raises(KeyError, match="nodo desconocido"):
        kg.add_relacion("ticket:T-1043", "afecta", "servicio:portal", "x")


def test_save_and_load_keep_attributes_and_sources(tmp_path):
    kg = small_graph()
    path = tmp_path / "grafo.json"
    kg.save(path)

    cargado = KnowledgeGraph.load(path)
    assert cargado.g.is_multigraph() and cargado.g.is_directed()
    assert cargado.g.nodes["ticket:T-1043"]["prioridad"] == "alta"
    assert cargado.g.edges["ticket:T-1043", "proveedor:NubeHost", "depende_de"]["fuentes"] == [
        "tickets_db.json"
    ]
    assert cargado.resumen() == kg.resumen()
