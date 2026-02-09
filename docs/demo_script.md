# Demo Script (Single-Turn) - February 26, 2026

## Objective
Run 5 scripted prompts end-to-end in Chainlit with stable behavior and citations.

## Operator Rules
- Use a new chat for each prompt.
- Keep each prompt as a single-turn question.
- Do not reformulate prompts during rehearsal; if a prompt fails, log failure and continue.

## Success Criteria Per Prompt
- The answer is returned in the same language as the prompt.
- At least one citation is shown.
- At least one citation link is clickable (`html_url` preferred, fallback `pdf_url`).
- No timeout or backend error state is shown.

## Prompt 1 (VA)
`Quines ajudes es van aprovar per als vehicles sinistrats per la DANA, de quina quantia eren i qui les podia demanar?`

Expected checks:
- Mentions aid scope and eligibility conditions from DOGV evidence.
- Includes at least one citation with DOGV link.

## Prompt 2 (ES)
`¿Qué becas o ayudas para estudios universitarios se publicaron en octubre de 2025 y quién podía solicitarlas?`

Expected checks:
- Response is constrained to the requested period or clarifies the temporal window used.
- Includes eligibility details only if supported by citations.

## Prompt 3 (VA)
`Quin és el termini de presentació de sol·licituds de la convocatòria que cita la referència 2025/XXXX?`

Expected checks:
- If the exact reference is not found, answer must state uncertainty clearly.
- No invented deadline values.

## Prompt 4 (ES)
`¿Qué subvenciones de movilidad o transporte se publicaron esta semana?`

Expected checks:
- Uses temporal policy behavior correctly for relative dates.
- Returns citation-backed result or explicit constrained/no-results response.

## Prompt 5 (VA)
`Resumeix els requisits de beneficiaris i les quanties de la convocatòria més recent relacionada amb habitatge.`

Expected checks:
- Mentions beneficiary requirements and amounts only when present in evidence.
- If missing, states equivalent of "No consta" or insufficient evidence.

## Rehearsal Log Template
- Prompt ID:
- Pass/Fail:
- Response latency (approx.):
- Citation links working (yes/no):
- Notes:
