from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge.chunking import Chunk
from knowledge.extract import (
    Entidad,
    Extraccion,
    ExtractionCache,
    add_extraccion,
    extraer,
    resolver,
)
from knowledge.graph_build import build_graph
from tests.test_graph_build import datos
from tests.test_rewrite import client_for, reply

PORTAL = {
    "entidades": [
        {"nombre": "NubeHost", "tipo": "proveedor"},
        {"nombre": "portal de proveedores", "tipo": "servicio"},
        {"nombre": "T-1043", "tipo": "ticket"},
    ],
    "relaciones": [
        {"origen": "NubeHost", "relacion": "presta", "destino": "portal de proveedores"},
        {"origen": "T-1043", "relacion": "afecta", "destino": "portal de proveedores"},
    ],
}


def test_each_relation_must_join_declared_entities_of_its_types():
    malo = {
        "entidades": [{"nombre": "T-1043", "tipo": "ticket"}],
        "relaciones": [{"origen": "T-1043", "relacion": "presta", "destino": "portal"}],
    }
    with pytest.raises(ValidationError, match="debe ir de un proveedor a un servicio"):
        Extraccion.model_validate(malo)
    assert len(Extraccion.model_validate(PORTAL).relaciones) == 2


def test_resolution_reuses_known_nodes_and_drops_invented_ones():
    kg = datos()
    assert resolver(kg, Entidad(nombre="t-1043", tipo="ticket")) == "ticket:T-1043"
    assert resolver(kg, Entidad(nombre="T-9999", tipo="ticket")) is None
    assert resolver(kg, Entidad(nombre="nubehost", tipo="proveedor")) == "proveedor:NubeHost"
    assert resolver(kg, Entidad(nombre="Acme Pagos", tipo="proveedor")) is None
    assert resolver(kg, Entidad(nombre="Guardia N2", tipo="guardia")) == "guardia:N2"

    nuevo = resolver(kg, Entidad(nombre="El Portal de  Proveedores", tipo="servicio"))
    assert nuevo == "servicio:portal de proveedores"
    assert resolver(kg, Entidad(nombre="portal de proveedores", tipo="servicio")) == nuevo


def test_extraction_adds_services_and_relations_with_their_source(logger, fake_sleep):
    kg = datos()
    chunk = kg.add_entidad("chunk", "runbooks/portal-proveedores.md#0", texto="...")
    _, client = client_for([reply(**PORTAL)], logger, fake_sleep)

    add_extraccion(kg, chunk, extraer(client, kg, "Lo hospeda NubeHost..."))

    portal = "servicio:portal de proveedores"
    fuentes = kg.g.edges["proveedor:NubeHost", portal, "presta"]["fuentes"]
    assert fuentes == ["runbooks/portal-proveedores.md#0"]
    assert kg.g.has_edge("ticket:T-1043", portal, key="afecta")
    assert set(kg.g.successors(chunk)) == {"proveedor:NubeHost", portal, "ticket:T-1043"}


def test_a_relation_with_the_wrong_types_is_repaired_once(logger, fake_sleep):
    malo = {**PORTAL, "relaciones": [{"origen": "T-1043", "relacion": "presta", "destino": "x"}]}
    provider, client = client_for([reply(**malo), reply(**PORTAL)], logger, fake_sleep)

    salida = extraer(client, datos(), "texto")

    assert len(provider.calls) == 2
    assert "debe ir de un proveedor" in provider.calls[1]["messages"][-1]["content"]
    assert len(salida.relaciones) == 2


def test_the_cache_avoids_paying_twice(tmp_path, logger, fake_sleep):
    provider, client = client_for([reply(**PORTAL)], logger, fake_sleep)
    cache = ExtractionCache(tmp_path / "extracciones.json")

    primera = extraer(client, datos(), "Lo hospeda NubeHost...", cache)
    segunda = extraer(client, datos(), "Lo hospeda NubeHost...", ExtractionCache(cache.path))

    assert primera == segunda
    assert len(provider.calls) == 1
    assert (cache.nuevas, cache.aciertos) == (1, 0)


def test_rules_run_after_the_llm_and_find_the_new_services(logger, fake_sleep):
    colas = {"entidades": [{"nombre": "cola de facturas", "tipo": "servicio"}], "relaciones": []}
    vacio = {"entidades": [], "relaciones": []}
    _, client = client_for([reply(**colas), reply(**vacio)], logger, fake_sleep)
    chunks = [
        Chunk("runbooks/reinicio-colas.md#0", "Reinicia la cola de facturas.", {"fuente": "a"}),
        Chunk("politicas/guardias.md#1", "N2 reinicia la cola de facturas.", {"fuente": "b"}),
    ]

    kg = build_graph(chunks, client)

    cola = "servicio:cola de facturas"
    assert kg.g.has_edge("chunk:politicas/guardias.md#1", cola, key="menciona")
    assert kg.g.has_edge("chunk:politicas/guardias.md#1", "guardia:N2", key="menciona")
