"""Reproducible LLM-as-judge for the v2 answer suite.

Regenerates the judgments that score_answers.py consumes, using a FIXED rubric
and prompt (below) at temperature 0. Point it at a judge model that is DIFFERENT
from the one being judged (the system answers with Qwen3.6-27B) so the judge isn't
grading its own output — set JUDGE_BASE_URL / JUDGE_MODEL to any OpenAI-compatible
endpoint (e.g. a stronger hosted model).

The judge sees, per question: the question, the reference/expected answer, the gold
reference id(s), whether the item is out-of-scope, the system's answer, and the exact
evidence quotes the system used (so faithfulness is judged against what it actually
saw). It emits the schema score_answers.py expects:
    {"id","correctness":0|1|2,"faithful":0|1,"critical_error":bool,"abstained":bool,"note"}

Resumable: appends one line per id, skips ids already in --out.

Usage:
    export JUDGE_BASE_URL=https://your-judge-host   # client appends /v1/chat/completions
    export JUDGE_MODEL=your-judge-model
    python eval_v2/judge.py --answers data/eval_v2/answers_master_20260708.jsonl \
                            --out data/eval_v2/judgments_regen.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re

import httpx

RUBRIC = """Eres un evaluador estricto de respuestas de un asistente legal sobre el DOGV
(diario oficial valenciano). Juzga UNA respuesta y devuelve SOLO un objeto JSON.

Se te da: la pregunta, la respuesta de referencia (correcta), la(s) referencia(s)
oro (formato AAAA/NNNN), si la pregunta es fuera de alcance, la respuesta del
sistema y las CITAS textuales que el sistema usó como evidencia.

Puntúa estos campos:
- "correctness": 2 = correcta y completa respecto a la referencia; 1 = parcial o
  incompleta (algún dato correcto pero falta o es vago); 0 = incorrecta, o se
  abstiene cuando la pregunta SÍ era respondible. (Para fuera de alcance, pon 0.)
- "faithful": 1 si cada afirmación de la respuesta está respaldada por las citas
  mostradas; 0 si inventa datos no presentes en la evidencia.
- "critical_error": true si hay algún error factual material (cifra, fecha,
  referencia o nombre equivocados) que engañaría al usuario. Este es un fallo grave.
- "abstained": true si el sistema declinó responder / dijo que no consta.
- "note": una frase breve justificando (en español), citando la discrepancia clave.

Para preguntas FUERA DE ALCANCE: lo correcto es abstenerse. Si se abstiene pon
correctness=0, abstained=true, critical_error=false. Si en cambio inventa una
respuesta, abstained=false y critical_error=true.

Devuelve SOLO el JSON, sin texto adicional."""


def _render_evidence(evidence) -> str:
    if not evidence:
        return "(sin evidencia)"
    lines = []
    for item in evidence:
        if isinstance(item, dict):
            lines.append(f"- [doc {item.get('doc_id')}] {str(item.get('quote', '')).strip()}")
    return "\n".join(lines) or "(sin evidencia)"


def build_user_prompt(rec: dict) -> str:
    return (
        f"Pregunta:\n{rec.get('question', '')}\n\n"
        f"Respuesta de referencia:\n{rec.get('expected_answer', '')}\n\n"
        f"Referencia(s) oro: {rec.get('gold_refs') or '(ninguna)'}\n"
        f"Fuera de alcance: {'sí' if rec.get('should_abstain') else 'no'}\n\n"
        f"Respuesta del sistema:\n{rec.get('answer', '')}\n\n"
        f"Citas que usó el sistema:\n{_render_evidence(rec.get('evidence'))}\n"
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip ```json fences if present, then grab the first {...} block.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_one(client: httpx.Client, model: str, rec: dict) -> dict:
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": build_user_prompt(rec)},
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 400},
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    oos = bool(rec.get("should_abstain"))
    return {
        "id": rec["id"],
        "correctness": int(parsed.get("correctness", 0)),
        "faithful": int(bool(parsed.get("faithful", 0))),
        "critical_error": bool(parsed.get("critical_error", False)),
        "abstained": bool(parsed.get("abstained", False)),
        "note": str(parsed.get("note", "")).strip(),
        "oos": oos,
        "judge_model": model,
    }


def done_ids(path: str) -> set[str]:
    ids: set[str] = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if not args.base_url or not args.model:
        ap.error(
            "set --base-url/--model (or JUDGE_BASE_URL/JUDGE_MODEL) to an OpenAI-compatible judge endpoint"
        )

    records = [json.loads(ln) for ln in open(args.answers, encoding="utf-8") if ln.strip()]
    skip = done_ids(args.out)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    n = 0
    with (
        httpx.Client(
            base_url=args.base_url.rstrip("/"), timeout=args.timeout, headers=headers
        ) as client,
        open(args.out, "a", encoding="utf-8") as fout,
    ):
        for rec in records:
            if rec["id"] in skip:
                continue
            try:
                j = judge_one(client, args.model, rec)
            except Exception as exc:
                print(f"[{rec['id']}] JUDGE ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(
                f"[{rec['id']}] corr={j['correctness']} faith={j['faithful']} crit={j['critical_error']}",
                flush=True,
            )
    print(f"\njudged {n} answers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
