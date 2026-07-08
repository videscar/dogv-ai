# Configuración — referencia completa

Todas las variables se leen de `.env` vía `pydantic-settings` (`api/config.py`, que es la
fuente de verdad de los valores por defecto — ante cualquier duda, gana el código).
Los nombres de entorno son los de campo en mayúsculas (p. ej. `ask_max_docs` → `ASK_MAX_DOCS`).

## Infraestructura (BD / LLM / embeddings)

| Variable | Default | Qué controla |
|---|---|---|
| `DOGV_BASE_URL` | `https://dogv.gva.es` | Portal DOGV origen de la ingesta |
| `DATABASE_URL` | `postgresql+psycopg2://…/dogv_ai` | PostgreSQL (SQLAlchemy) |
| `DOGV_DB_DSN` | — | DSN plano para psql/CLI |
| `LLM_BASE_URL` | `http://127.0.0.1:8000` | Servidor de chat OpenAI-compatible (vLLM) |
| `LLM_MODEL` | `qwen3.6-27b` | Nombre servido del modelo de chat |
| `LLM_TIMEOUT` / `LLM_MAX_TOKENS` | `300` / `8192` | Timeout y tope de generación |
| `EMBED_BASE_URL` | `http://127.0.0.1:8001` | Servidor de embeddings (llama.cpp) |
| `EMBED_MODEL` / `EMBED_TIMEOUT` | `bge-m3` / `60` | Modelo de embeddings |
| `EMBEDDING_DIM` | `1024` | Dimensión de los vectores (bge-m3) |

## Chunking (ingesta)

| Variable | Default | Qué controla |
|---|---|---|
| `CHUNK_MIN_TOKENS` / `CHUNK_MAX_TOKENS` | `300` / `500` | Tamaño de chunk en tokens reales del tokenizer de bge-m3 |
| `CHUNK_OVERLAP_TOKENS` | `80` | Solape entre chunks |
| `DOGV_CLASSIFY_WORKERS` | `2` | Paralelismo de la clasificación LLM |
| `DOGV_CLASSIFY_LLM_URLS` | — | Opcional: repartir la clasificación entre varios endpoints |

## Recuperación (carriles, RRF, BM25)

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_LANES` | `vector,bm25,title` | Carriles activos |
| `ASK_MAX_DOCS` / `ASK_MIN_DOCS` | `20` / `3` | Tamaño del pool fusionado / mínimo antes de relajar filtros |
| `ASK_MAX_FACETS` | `3` | Descomposición de la pregunta en facetas BM25 |
| `ASK_MAX_DOCS_EXPAND` | `20` | Ampliación del pool cuando el margen RRF es plano |
| `ASK_BM25_MAX_DOCS` | `50` | Límite por consulta BM25 |
| `ASK_DOC_CONFIDENCE_MIN` | `0.06` | Confianza mínima de documento |
| `ASK_RRF_EXPAND_MARGIN_RATIO` / `ASK_RRF_MARGIN_PROBE` | `0.12` / `5` | Umbral y profundidad del margen RRF que dispara expansiones |
| `ASK_RRF_WEIGHT_VECTOR` / `_BM25` / `_TITLE` / `_TITLE_LEXICAL` | `1.0` / `1.0` / `1.0` / `0.8` | Pesos RRF por carril |
| `ASK_FALLBACK_ALLOW_MARGIN` | `false` | Permitir relajar filtros también por margen bajo (no solo por pool corto) |
| `BM25_FUSE_WEIGHT_CHUNK` / `_STRICT` / `_TITLE` | `1.0` / `1.2` / `0.9` | Pesos de la fusión interna BM25 |
| `BM25_STRICT_PRIMARY_MIN` | `1` | Hits mínimos para que la consulta estricta sea primaria |
| `BM25_PRF_DOCS` / `BM25_PRF_TERMS` | `5` / `6` | Pseudo-relevance feedback |

### HyDE (con puerta de confianza)

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_HYDE_ENABLED` | `true` | Carril HyDE (documento hipotético embebido) |
| `ASK_HYDE_CONDITIONAL` | `true` | Saltar HyDE cuando la pregunta cita una norma concreta |
| `ASK_HYDE_CONFIDENCE_GATED` | `true` | Disparar HyDE solo con pool base de baja confianza |
| `ASK_HYDE_MARGIN_THRESHOLD` | `0.22` | Umbral de margen RRF que dispara HyDE |
| `ASK_RRF_WEIGHT_HYDE` | `3.0` | Peso RRF del carril HyDE |

### Anclas semánticas

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_SEMANTIC_ANCHOR_ENABLED` | `true` | Plaza garantizada en el pool para el top-N de los carriles semánticos |
| `ASK_SEMANTIC_ANCHOR_TOP` | `3` | Profundidad de ancla por carril |

## Conversación (multi-turno)

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_CONTEXTUALIZE_ENABLED` | `true` | Reescritura de turnos de seguimiento como consulta autónoma |
| `ASK_HISTORY_MAX_TURNS` | `6` | Turnos previos usados en contextualización y síntesis |

## Lectura y re-ranking

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_RERANK_TOP_N` / `ASK_RERANK_MAX_CANDIDATES` | `5` / `10` | Re-ranking LLM |
| `ASK_RERANK_EXPAND_CANDIDATES` / `_EXPAND_TOP_N` | `10` / `2` | Ampliación del rerank con margen plano |
| `ASK_RERANK_COVERAGE_KEEP` / `_RECENT_KEEP` | `4` / `2` | Cupos de cobertura/recencia en la selección |
| `ENUMERATION_AUGMENT_ENABLED` / `_MAX` | `true` / `20` | Consultas de enumeración: ampliar el pool con la serie mes+categoría vía SQL |
| `ASK_ENUMERATION_MAX_CANDIDATES` / `_TOP_N` | `25` / `15` | Presupuestos ampliados de rerank/lectura para enumeraciones |
| `ASK_EDITION_RECENCY_ENABLED` | `true` | Suprimir ediciones hermanas obsoletas (publicaciones recurrentes) |
| `ASK_EDITION_RECENCY_SIM` / `_SCAN_N` | `0.86` / `12` | Umbral de coseno doc-embedding / profundidad de escaneo |
| `ASK_READ_MAX_DOCS` | `3` | Documentos leídos por defecto |
| `ASK_CHUNKS_PER_DOC` | `10` | Chunks por documento en el payload |
| `ASK_READ_RETRIEVAL_CHUNKS` | `4` | Chunks provenientes de retrieval/BM25 conservados por documento |
| `ASK_READ_CITATION_FLOOR` / `_DOCS` | `true` / `5` | Todo documento seleccionado aporta al menos una cita usable |
| `ASK_CHUNK_MAX_CHARS` | `1200` | Truncado de chunk en el payload |
| `ASK_CHUNK_WINDOW_ENABLED` | `true` | Truncado por ventana de keywords (no por prefijo) |
| `ASK_QUOTE_REGROUND_ENABLED` | `true` | Re-anclar citas no literales del LLM al chunk fuente |
| `ASK_DOC_FALLBACK_CHARS` | `12000` | Fallback de texto de documento si faltan chunks |
| `ASK_READ_EXPAND_DOCS` / `_COVERAGE_DOCS` / `_ELIGIBILITY_DOCS` / `_AMOUNT_DOCS` | `2` / `2` / `1` / `1` | Cupos extra de lectura por tipo de pregunta |
| `FULL_DOC_MAX_DOCS` / `FULL_DOC_MAX_CHARS` / `FULL_DOC_TOTAL_CHARS` | `2` / `120000` / `200000` | Límites de lectura de documento completo |
| `ASK_SYNTHESIS_THINKING` | `false` | Thinking en la síntesis final (off = determinista) |
| `ASK_SYNTHESIS_TEMPERATURE` | `0.0` | Temperatura de la síntesis final |

## Temporal

| Variable | Default | Qué controla |
|---|---|---|
| `ASK_TEMPORAL_POLICY` | `filter` | `filter` o `reject` para marcos temporales |
| `TEMPORAL_TIMEZONE` / `TEMPORAL_WEEK_START` | `Europe/Madrid` / `monday` | Resolución de fechas relativas |
| `FEED_RECENT_DAYS` | `21` | Ventana de "publicado recientemente" |
| `ASK_LLM_EXPAND` | `true` | Expansión LLM de la consulta |

## Ingesta automática y backfill

| Variable | Default | Qué controla |
|---|---|---|
| `AUTO_INGEST_ENABLED` | `false` | Ingesta desde la API (OFF en producción: la cubre el timer diario) |
| `AUTO_INGEST_MAX_DAYS` / `AUTO_INGEST_LANGUAGES` | `15` / `es_es,va_va` | Alcance de la ingesta automática |
| `AUTO_INGEST_STARTUP_ENABLED` / `_BLOCKING` | `true` / `false` | Sincronización al arrancar la API |
| `AUTO_INGEST_STARTUP_PURGE_OLD` / `_REPAIR_GAPS` | `true` / `true` | Purga de la ventana + reparación de huecos |
| `AUTO_INGEST_STARTUP_LOCK_ID` | `190021` | Advisory lock de PostgreSQL (una sync a la vez) |
| `AUTO_INGEST_GAP_CHECK_RETRIES` / `_BACKOFF_SECONDS` | `3` / `1.5` | Reintentos de verificación de huecos contra el origen |
| `AUTO_INGEST_GAP_REPAIR_SCAN_MAX_DAYS` | `45` | Ventana máxima de escaneo de huecos |
| `BACKFILL_ENABLED` | `true` | Fetch histórico bajo demanda (norma citada fuera de ventana) |
| `INGEST_COMPLETE_BIS_EDITIONS` | `true` | Capturar la edición hermana en fechas con edición *bis* |
| `HOT_INDEX_MONTHS` / `WARM_INDEX_MONTHS` | `6` / `24` | Ventanas del índice (hot solo conceptual) |

## Pipeline de respuesta

| Variable | Default | Qué controla |
|---|---|---|
| `ANSWER_VALIDATOR_ENABLED` | `true` | Validador de la respuesta final |
| `ANSWER_CLAIM_GUARD_MODE` | `unit_aware_strict` | Guardia de cifras: solo cantidades monetarias/porcentuales deben existir en la fuente citada (`current_strict` — cualquier número — se evaluó y descartó por falsos positivos) |
| `ANSWER_REPAIR_ATTEMPTS` / `ANSWER_REPAIR_MODE` | `1` / `conditional` | Reintento de reparación |
| `ANSWER_FALLBACK_STYLE` / `_MAX_ITEMS` | `concise_summary` / `3` | Respuesta de repliegue |
| `ANSWER_NORM_TARGET_CITATION_ENABLED` | `true` | Citar siempre la norma nombrada si está en el conjunto de lectura |
| `INFER_NAMED_NORM_FROM_CORPUS_ENABLED` | `true` | Inferir el N/YYYY de una norma nombrada solo por tema a partir del corpus |
| `ANSWER_MUTATORS_ENABLED` / `ANSWER_MISSING_NOTES_ENABLED` | `false` / `false` | Post-procesos experimentales (off) |

## Demo / observabilidad / UI

| Variable | Default | Qué controla |
|---|---|---|
| `DEMO_ENFORCE_READY_GATE` | `true` | `/ask` devuelve 503 hasta que el índice está listo |
| `DEMO_REQUEST_TIMEOUT_SECONDS` | `60` | Timeout de la UI (0 = sin timeout) |
| `TRACE_ENABLED` | `false` | Persistencia de trazas por petición |
| `CHAINLIT_BACKEND_URL` | `http://127.0.0.1:8088` | URL de la API para la UI |
| `CHAINLIT_ENABLE_DATA_LAYER` | `false` | Persistencia de hilos en Chainlit |
| `DEMO_HISTORY_MAX_TURNS` | `6` | Solo UI: turnos que la sesión Chainlit reenvía como `history` |
