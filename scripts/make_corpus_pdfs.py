"""Genera los PDF de contratos del corpus de la Clase 6 (se corre una vez; no es runtime).

    uv run scripts/make_corpus_pdfs.py

Escribe data/docs/contratos/{pagosred,nubehost,transportes-sierra}.pdf a partir del texto
de este archivo. Los PDF traen defectos a propósito, como un PDF real: encabezado y pie
repetidos en cada página ("Página N de M") y palabras partidas con guion al final de
línea. `knowledge.ingest.clean_text` es quien los limpia.

Contratos ficticios: ningún dato es de personas ni de empresas reales.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "data" / "docs" / "contratos"
HEADER = "Acme Operaciones · Contrato de servicio · Confidencial"
FONT, SIZE, LEADING = "Helvetica", 10.5, 15

# Un contrato es una lista de páginas; cada página, una lista de líneas ya cortadas.
# Los cortes con guion ("contra-" / "prestación") son el defecto que limpia clean_text.
CONTRATOS: dict[str, list[list[str]]] = {
    "pagosred": [
        [
            "CONTRATO DE PRESTACIÓN DE SERVICIOS DE PASARELA DE PAGOS",
            "",
            "Partes. Acme Operaciones (en adelante, Acme) y PagosRed (en adelante, el",
            "Proveedor) celebran este contrato para el procesamiento de cobros con tarjeta",
            "en la tienda en línea y en el portal de clientes de Acme.",
            "",
            "Cláusula 1. Objeto. El Proveedor procesa los cobros con tarjeta de crédito y",
            "débito, la reversión de cargos y la conciliación diaria de las transac-",
            "ciones, y entrega a Acme un reporte de conciliación cada día hábil.",
            "",
            "Cláusula 2. Vigencia. El contrato tiene una vigencia de dos años a partir de",
            "su firma y se renueva por periodos de un año, salvo aviso por escrito con",
            "sesenta días naturales de anticipación.",
        ],
        [
            "Cláusula 3. Obligaciones del Proveedor.",
            "",
            "a) Mantener la pasarela disponible al menos el 99.5% del tiempo de cada mes.",
            "b) Notificar a Acme cualquier incidente que afecte cobros en menos de una",
            "hora desde que lo detecte, por correo a la guardia de operaciones.",
            "c) Procesar las solicitudes de reversión de cargos duplicados y reflejarlas",
            "en el reporte de conciliación siguiente.",
            "d) Guardar la información de tarjetas conforme a las normas de la industria;",
            "Acme nunca recibe ni almacena el número completo de una tarjeta.",
            "",
            "Cláusula 4. Obligaciones de Acme. Acme abre los casos de soporte con el",
            "número de folio, la hora y los últimos cuatro dígitos de la tarjeta, y",
            "designa un dueño del servicio de pagos como contacto principal.",
        ],
        [
            "Cláusula 5. Niveles de servicio.",
            "",
            "Tiempo de respuesta. El Proveedor atenderá cada caso de soporte abierto por",
            "Acme con una primera respuesta con diagnóstico en un plazo máximo de 4 horas",
            "hábiles, contadas desde la apertura del caso. Una respuesta automática no",
            "cuenta como primera respuesta.",
            "",
            "Penalización por incumplimiento. Si el Proveedor no cumple el tiempo de res-",
            "puesta, Acme aplicará una penalización del 5% de la cuota mensual por cada",
            "caso fuera de plazo, descontada en la factura del mes siguiente. Tres incum-",
            "plimientos en un mismo trimestre permiten a Acme rescindir el contrato sin",
            "costo, con aviso por escrito de treinta días naturales.",
        ],
        [
            "Cláusula 6. Confidencialidad. Las partes mantendrán confidencial la informa-",
            "ción de clientes, cobros y conciliaciones durante la vigencia del contrato",
            "y cinco años después de su terminación.",
            "",
            "Cláusula 7. Contacto. Los casos de soporte se abren en",
            "soporte@pagosred.example. El dueño del servicio por parte de Acme es el",
            "área de Finanzas.",
            "",
            "Cláusula 8. Jurisdicción. Para la interpretación de este contrato, las",
            "partes se someten a los tribunales de la Ciudad de México.",
        ],
    ],
    "nubehost": [
        [
            "CONTRATO DE PRESTACIÓN DE SERVICIOS DE HOSPEDAJE",
            "",
            "Partes. Acme Operaciones (en adelante, Acme) y NubeHost (en adelante, el",
            "Proveedor) celebran este contrato para el hospedaje del portal de",
            "proveedores y del sitio web de Acme.",
            "",
            "Cláusula 1. Objeto. El Proveedor hospeda el portal de proveedores y el sitio",
            "web, administra sus certificados y balanceadores, y realiza un respaldo",
            "nocturno de las bases de datos con una retención de treinta días.",
            "",
            "Cláusula 2. Vigencia. Un año a partir de la firma, con renovación automá-",
            "tica por periodos iguales.",
        ],
        [
            "Cláusula 3. Niveles de servicio.",
            "",
            "Disponibilidad. El portal de proveedores estará disponible al menos el",
            "99.9% del tiempo de cada mes, sin contar las ventanas de mantenimiento",
            "avisadas con 72 horas de anticipación.",
            "",
            "Tiempo de respuesta. El Proveedor dará una primera respuesta con diagnóstico",
            "en un plazo máximo de 2 horas hábiles desde la apertura de cada caso.",
            "",
            "Certificados. El Proveedor renovará los certificados del sitio al menos",
            "quince días antes de su vencimiento y avisará a Acme de la renovación.",
        ],
        [
            "Cláusula 4. Penalizaciones. Por cada hora de indisponibilidad por debajo del",
            "nivel pactado, Acme recibirá un crédito del 2% de la cuota mensual. Si el",
            "Proveedor no cumple el tiempo de respuesta en más de dos casos en un mes,",
            "Acme podrá solicitar una revisión del servicio con la dirección del",
            "Proveedor.",
            "",
            "Cláusula 5. Contacto. Los casos se abren en mesa@nubehost.example. El dueño",
            "del servicio por parte de Acme es el área de TI.",
        ],
    ],
    "transportes-sierra": [
        [
            "CONTRATO DE PRESTACIÓN DE SERVICIOS DE TRANSPORTE",
            "",
            "Partes. Acme Operaciones (en adelante, Acme) y Transportes Sierra (en",
            "adelante, el Proveedor) celebran este contrato para fletes y rutas de",
            "distribución regional.",
            "",
            "Cláusula 1. Objeto. El Proveedor asigna unidades a las rutas de distribución",
            "programadas por Logística y atiende pedidos urgentes según disponibi-",
            "lidad, con unidades aseguradas y operadores certificados.",
        ],
        [
            "Cláusula 2. Niveles de servicio.",
            "",
            "Asignación de unidades. El Proveedor confirma la unidad de cada ruta",
            "programada con al menos 48 horas de anticipación.",
            "",
            "Tiempo de respuesta. Para pedidos urgentes y rutas sin unidad asignada, el",
            "Proveedor responde en un plazo máximo de 12 horas hábiles con una unidad",
            "disponible o con una alternativa.",
            "",
            "Penalización. Una ruta programada sin unidad asignada a tiempo genera un",
            "descuento del 10% del flete de esa ruta.",
        ],
        [
            "Cláusula 3. Suspensión. Acme puede suspender el contrato si el Proveedor",
            "deja sin unidad tres rutas en un mismo mes. Mientras el contrato esté",
            "suspendido, Logística contrata transportistas externos para las rutas",
            "comprometidas.",
            "",
            "Cláusula 4. Contacto. Las solicitudes se envían a",
            "trafico@transportessierra.example. El dueño del servicio por parte de Acme",
            "es el área de Logística.",
        ],
    ],
}


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    total = len(pages)
    for number, lines in enumerate(pages, start=1):
        pdf.setFont(FONT, 9)
        pdf.drawString(72, height - 50, HEADER)
        pdf.drawString(72, 40, f"Página {number} de {total}")
        pdf.setFont(FONT, SIZE)
        y = height - 95
        for line in lines:
            pdf.drawString(72, y, line)
            y -= LEADING
        pdf.showPage()
    pdf.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in CONTRATOS.items():
        path = OUT / f"{name}.pdf"
        write_pdf(path, pages)
        print(f"{path.relative_to(OUT.parents[2])}: {len(pages)} páginas")


if __name__ == "__main__":
    main()
