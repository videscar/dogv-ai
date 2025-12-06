from typing import Iterable, Mapping

import requests

from .config import get_settings

settings = get_settings()


def _build_prompt(question: str, language: str | None, hits: Iterable[Mapping]) -> str:
    """
    Build a single text prompt for the local LLM, including context from retrieved docs.
    """
    context_blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(
            f"[{i}] DOGV {h.get('issue_numero')} ({h.get('issue_date')})\n"
            f"Sección: {h.get('section')}\n"
            f"Conselleria / organismo: {h.get('conselleria')}\n"
            f"Título: {h.get('title')}\n"
            f"Fragmento:\n{h.get('snippet')}\n"
            f"PDF: {h.get('pdf_url')}\n"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "No se han encontrado documentos relevantes."

    lang = (language or "auto").lower()
    if "va" in lang or "valenci" in lang:
        answer_lang_instruction = (
            "Respon en valencià clar i precís. "
            "Si no tens prou informació als documents, digues-ho explícitament."
        )
    else:
        answer_lang_instruction = (
            "Responde en castellano claro y preciso. "
            "Si no tienes suficiente información en los documentos, dilo explícitamente."
        )

    system_instructions = (
        "Eres un asistente jurídico especializado en el Diari Oficial de la Generalitat "
        "Valenciana (DOGV). Solo puedes usar la información que aparece en los documentos "
        "que te proporciono a continuación. No inventes normas ni fechas. "
        "Si la pregunta no se puede responder con estos documentos, dilo claramente.\n\n"
        f"{answer_lang_instruction}\n\n"
    )

    user_part = (
        "Pregunta del usuario:\n"
        f"{question}\n\n"
        "Documentos relevantes:\n"
        f"{context_text}\n\n"
        "Tarea:\n"
        "- Analiza los documentos anteriores.\n"
        "- Contesta exactamente a la pregunta, citando cuando sea útil el número de DOGV o el órgano que dicta el acto.\n"
        "- Si la respuesta depende de otra normativa externa al DOGV, indica el límite de la información disponible.\n"
    )

    return system_instructions + user_part


def generate_answer_with_ollama(
    question: str,
    language: str | None,
    hits: Iterable[Mapping],
    model: str | None = None,
) -> str:
    """
    Call local Ollama (gpt-oss:20b or similar) and return the answer text.
    """

    base_url = settings.ollama_base_url.rstrip("/")
    model_name = model or settings.ollama_model

    prompt = _build_prompt(question, language, hits)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful legal assistant for DOGV."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        return (
            "Error al llamar al modelo local de Ollama: "
            f"{e}\n\n"
            "Documentos que se han encontrado:\n" +
            "\n".join(
                f"- DOGV {h.get('issue_numero')} ({h.get('issue_date')}), "
                f"{h.get('conselleria')}: {h.get('title')}"
                for h in hits
            )
        )

    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content") or ""

    return content.strip() if content else "El modelo no ha devuelto contenido."
