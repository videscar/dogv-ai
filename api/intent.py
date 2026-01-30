from __future__ import annotations

from datetime import datetime
from typing import Any

from .ollama import OllamaClient


INTENT_SYSTEM = (
    "Eres un analista de intencion para consultas del DOGV. "
    "Devuelve SOLO JSON con estos campos: "
    "language (es|ca), doc_kind, doc_subkind, keywords (lista), "
    "since_date (YYYY-MM-DD o null), until_date (YYYY-MM-DD o null), "
    "needs_online (true|false), entities (objeto con organismo, municipio, cuerpo)."
)

INTENT_USER = """Pregunta del usuario:
{question}

Categorias principales permitidas para doc_kind:
- Empleo Publico
- Ayudas
- Subvenciones
- Premios
- Becas
- Otros

Subcategorias frecuentes para doc_subkind (si aplica):
- OPE
- Convocatoria
- Listas
- Tribunales
- Resultados
- Correcciones
- Bases
- Adjudicaciones
"""


def analyze_intent(question: str) -> dict[str, Any]:
    client = OllamaClient()
    messages = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": INTENT_USER.format(question=question)},
    ]
    result = client.chat_json(messages, temperature=0.0)

    for key in ("since_date", "until_date"):
        value = result.get(key)
        if value:
            try:
                result[key] = datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                result[key] = None
    return result
