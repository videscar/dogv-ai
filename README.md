# DOGV AI (local)

Asistente local para consultas sobre el DOGV (empleo publico, ayudas/subvenciones/premios y becas),
con ingesta automatizada, busqueda hibrida y respuestas citadas. Funciona 100% en local con Ollama
y PostgreSQL (pgvector + tsvector).

Estado actual (importante):
- `scripts/build_chunks.py` esta ejecutandose para 2024-01-01 .. 2024-12-31.
- El eval set y las curvas de recall todavia no se han ejecutado con los embeddings nuevos.

## Estado actual (lo que ya funciona)
- Ingesta de sumarios DOGV ES/VA y extraccion de PDFs completos.
- Limpieza y extraccion de texto, clasificacion por doc_kind/doc_subkind.
- Indexado hibrido en Postgres: embeddings (pgvector) + BM25 (tsvector).
- Embeddings por titulo y a nivel documento (titulo + resumen + metadata).
- Pipeline de consulta multi‑paso con LangGraph:
  - analisis de intencion
  - recuperacion hibrida (vector + BM25 + titulos)
  - RRF + relajacion de filtros + backoff sin filtros
  - re‑ranking con LLM (recall‑first)
  - lectura de chunks + lectura de documentos completos (segun evidencia/confianza)
  - respuesta final con citas o respuesta negativa si no hay evidencia
- BM25 sin expansión (consulta literal).
- Top‑k adaptativo en retrieval/rerank/lectura si el RRF es plano.
- Idiomas: espanol y valenciano (mapeo ca -> va_va).

## Arquitectura actual (pipeline real)
### 1) Ingesta
- `sumario_ingest.py`: baja el sumario diario y upsert de issues.
- `extract_documents.py`: crea documentos (disposiciones) por issue.
- `download_assets.py` / `download_html.py`: cache local en `data/pdf_cache/` y `data/html_cache/`.
- `extract_text.py`: extrae texto completo de PDF/HTML.
- `classify_documents.py`: clasifica tipo/subtipo (doc_kind/doc_subkind).
- `build_chunks.py`: chunking + embeddings + BM25 tsvector.

### 2) Indexado y almacenamiento
Tablas clave:
- `dogv_issues`: issues por fecha/idioma.
- `dogv_documents`: disposiciones (title/ref/pdf_url/html_url/text).
- `rag_chunk`: chunks con embeddings y `tsvector`.
- `rag_title`: embeddings de titulos.

Ventanas:
- Warm index: ultimos 24 meses (se purga lo mas antiguo en `maintain_indices.py`).
- Hot index: concepto configurado (6 meses), aun no separado fisicamente.

### 3) Consulta (LangGraph)
Archivo: `agent/graph.py`
- Intent: LLM detecta idioma, doc_kind/doc_subkind, entidades y fechas.
- Auto‑ingest (opcional): si hay rango en la pregunta o faltan dias recientes.
- Recuperacion hibrida:
  - `vector_search` (embeddings)
  - `bm25_search` (tsvector, catalan para va_va, fallback a spanish)
  - `title_vector_search`
  - fusion RRF
  - expansion adaptativa cuando el margen de scores es bajo
- Re‑ranking por LLM (top 5).
- Lectura:
  - Top N docs como maximo (configurable, con expansion si RRF plano).
  - Chunks: 4 por doc (fallback por embeddings si faltan).
  - Full‑docs: max 2 docs y limite de chars por doc y total (solo si hay evidencia o alta confianza).
- Respuesta:
  - Si no hay evidencia, se responde con “no hay publicaciones” y se pide mas detalle.
  - Evidencia tiene fallback lexico si el extractor LLM no devuelve pruebas.
  - Evidencia puede incluir extractos numericos si la pregunta pide cuantias.

## Modelos (Ollama)
- LLM: `gpt-oss-20b-high` (Reasoning: high en Modelfile).
- Embeddings: `bge-m3`.
- Contexto: `OLLAMA_NUM_CTX` (actual 65536).
- Timeout: `OLLAMA_TIMEOUT` (segundos).

Modelfile: `ollama/gpt-oss-20b-high.Modelfile`

## Configuracion clave (.env)
Usa `.env.example` como plantilla. Ejemplos (valores reales en `api/config.py` y `.env`):
- `DOGV_DB_DSN`
- `OLLAMA_MODEL=gpt-oss-20b-high`
- `OLLAMA_EMBED_MODEL=bge-m3`
- `OLLAMA_NUM_CTX=65536`
- `OLLAMA_TIMEOUT=300`
- `ASK_LANES=vector,bm25,title`
- `ASK_MAX_DOCS=20`
- `ASK_MIN_DOCS=3`
- `ASK_READ_MAX_DOCS=3`
- `ASK_CHUNKS_PER_DOC=4`
- `ASK_CHUNK_MAX_CHARS=1200`
- `ASK_DOC_FALLBACK_CHARS=12000`
- `ASK_RERANK_TOP_N=5`
- `ASK_RERANK_MAX_CANDIDATES=10`
- `ASK_RERANK_EXPAND_CANDIDATES=10`
- `ASK_RERANK_EXPAND_TOP_N=2`
- `ASK_DOC_CONFIDENCE_MIN=0.06`
- `ASK_RRF_EXPAND_MARGIN_RATIO=0.12`
- `ASK_RRF_MARGIN_PROBE=5`
- `ASK_MAX_DOCS_EXPAND=20`
- `ASK_READ_EXPAND_DOCS=2`
- `FULL_DOC_MAX_DOCS=2`
- `FULL_DOC_MAX_CHARS=120000`
- `FULL_DOC_TOTAL_CHARS=200000`
- `AUTO_INGEST_ENABLED=true|false`
- `AUTO_INGEST_MAX_DAYS=15`
- `BACKFILL_ENABLED=true|false`
- `WARM_INDEX_MONTHS=24`
- `CHUNK_MIN_CHARS=200`
- `CHUNK_MAX_CHARS=500`
- `CHUNK_OVERLAP_CHARS=80`

## Scripts principales
- Bootstrap/diario: `scripts/maintain_indices.py`
- Pipeline completo: `api/ingest_pipeline.py`
- Rebuild BM25 tsvector: `scripts/rebuild_tsv.py`
- Reset BD: `scripts/reset_db.py`
- Eval set (v1): `scripts/build_eval_set.py`
- Audit manual (muestra): `scripts/audit_eval_set.py`
- Evaluacion/recall: `scripts/run_eval.py`
- Check de regresiones: `scripts/check_eval_regression.py`

## Operativa rapida
Bootstrap 2 anos (warm index):
```bash
.venv/bin/python scripts/maintain_indices.py --bootstrap
```

Ingesta diaria (ultimo N dias):
```bash
.venv/bin/python scripts/maintain_indices.py --daily
```

Rebuild BM25 para valenciano:
```bash
.venv/bin/python scripts/rebuild_tsv.py --language va_va --ts-config catalan --batch-size 5000
```

API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Evaluacion (recall)
- Schema eval set: `data/eval_schema_v1.json`
- Generar set v1: `python scripts/build_eval_set.py --size 200 --output data/eval_set_v1.json`
- Audit manual (20): `python scripts/audit_eval_set.py --input data/eval_set_v1.json --count 20`
- Run eval: `python scripts/run_eval.py --input data/eval_set_v1.json --include-nofilter`
- Regression gate (usa un baseline report en `data/eval_baseline.json`): `python scripts/check_eval_regression.py --report data/eval_reports/<run_id>.json`

Pendiente:
- Ejecutar eval con los nuevos embeddings de documento cuando termine `build_chunks`.

## Embeddings a nivel documento
- Tabla nueva: `rag_doc` (ver `sql/2026-03-doc-embeddings.sql`).
- `scripts/build_chunks.py` ahora genera embeddings de documento (titulo + resumen + metadata).
- Flag opcional: `--skip-doc-embeddings`.

## Endpoints
- `GET /health`
- `GET /issues`
- `GET /issues/{issue_id}/documents`
- `POST /ask`

## Limitaciones actuales
- Hot index no separado fisicamente (solo warm index real).
- BM25 depende de tsvector correcto por idioma.
- Full‑docs limitados por umbrales de chars.
- OCR no integrado para PDFs escaneados.

## Prioridades de mejora (corto plazo)
- Mejorar recall BM25 y mezcla ES/VA (fallbacks).
- Afinar lectura completa cuando el doc correcto ya fue localizado.
- Reducir latencia sin perder evidencia.
