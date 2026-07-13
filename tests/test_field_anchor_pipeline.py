"""Pipeline-stage regression tests for multi-field grant-query anchoring.

Each test pins the deterministic slice of one verified failure mechanism:
- the second hop must never run the field-request clause as a retrieval query
  (T1/T3: it merged+pinned amount-dense credit transfers and tribunal minutes);
- the facet BM25 lanes must vote for the identity, not the field vocabulary;
- the reader must not select best-chunk-per-field across foreign documents
  (T4: a Social-Security law chunk became the "règim" answer);
- extract-length chunks must reach the reader whole (T3: the 1200-char window
  cut 'Quart. Import ... 1.865,05 euros', turning a stated amount into a false
  'No consta');
- the synthesis prompt must carry the single-document anchoring instruction.
"""

from __future__ import annotations

from types import SimpleNamespace

import agent.nodes.read as read_mod
import agent.nodes.second_hop as second_hop
import api.reader as reader
from agent.nodes.read_chunks import _window_chunk_text
from agent.nodes.retrieve import _build_facet_specs

Q1_QUIMICA = (
    "Estic acabant el grau de Química a la UJI i he superat més de la meitat dels "
    "crèdits. Hi ha alguna beca oberta per iniciar-me en la investigació sobre "
    "plaguicides o aigües? Si n'hi ha, dis-me qui hi pot optar, l'import, el "
    "termini i on són les bases."
)
Q3_MODALITAT = (
    "Vull una beca d'iniciació a la investigació de la UJI del projecte 'Modalitat "
    "A: ajudes complementàries…'. Quant es cobra al mes i a quin departament o "
    "institut és?"
)
Q4_TEN_FIELDS = (
    "Sóc estudianta del màster en Cervell i Conducta a la UJI. Digues-me d'aquella "
    "beca de Psicologia: 1) requisits, 2) finalitat, 3) retribució mensual, 4) hores "
    "setmanals, 5) durada, 6) import global, 7) règim de Seguretat Social, 8) "
    "termini, 9) on tramitar, 10) codi BDNS."
)

# A realistic UJI grant extract: beneficiaris up front, the asked fields
# (retribució/hores/durada, import, termini) past char 1200 — the shape that
# produced the T3 false-negative under the prefix-biased 1200-char window.
EXTRACT = (
    "EXTRACTE de la Resolució de 29 d'abril de 2026, del Vicerectorat "
    "d'Investigació, per la qual es convoca una beca d'iniciació a la "
    "investigació (projecte «Modalitat A: ajudes complementàries», codi 25I637). "
    "Institut Universitari de Plaguicides i Aigües. "
    "BDNS (identif.): 902932. "
    "De conformitat amb el que es preveu en els articles 17.3.b) i 20.8.a) de la "
    "Llei 38/2003, de 17 de novembre, general de subvencions, es publica "
    "l'extracte de la convocatòria. "
    "Primer. Persones beneficiàries. Poden participar en la present convocatòria "
    "les persones físiques que reunisquen els següents requisits: estudiantat de "
    "la Universitat Jaume I de grau en Química que haja superat almenys el 50 % "
    "dels crèdits de la titulació de grau i que no haja superat tots els crèdits "
    "de què consta la titulació, i tindrà la nacionalitat espanyola o serà "
    "nacional d'un estat membre de la Unió Europea o d'un altre país no "
    "pertanyent a la UE amb permís de residència o d'estudis a Espanya i amb la "
    "documentació identificativa en vigor en el moment de la incorporació a la "
    "beca, sense cap altra condició addicional que les establides en les bases "
    "de la convocatòria publicades en la seu electrònica de la universitat. "
    "Segon. Finalitat. Obtindre formació i col·laborar en el projecte "
    "d'investigació sobre plaguicides i aigües de l'institut universitari, "
    "d'acord amb el pla de treball aprovat per la comissió corresponent i sota "
    "la supervisió de la persona investigadora responsable del projecte. "
    "Tercer. Bases reguladores. Accessibles en la seu electrònica. "
    "Quart. Import. L'import global destinat a la concessió d'esta beca és de "
    "1.865,05 euros. "
    "Quint. Termini de presentació de sol·licituds. Deu dies hàbils comptadors "
    "des de l'endemà de la publicació d'este extracte en el DOGV."
)


class TestSecondHopNeverRunsFieldClauses:
    def test_facet_targets_empty_for_multi_field_query(self):
        # T1/T3 regression: the field-request clause became a hop query and its
        # per-field noise pool was merged AND pinned into the read set.
        assert second_hop._facet_targets(Q1_QUIMICA) == []
        assert second_hop._facet_targets(Q3_MODALITAT) == []
        assert second_hop._facet_targets(Q4_TEN_FIELDS) == []

    def test_facet_targets_kept_for_ordinary_questions(self):
        q = (
            "Quina taxa es paga per a la renovació del títol de família nombrosa? "
            "Quin és el termini per a sol·licitar la targeta sanitària?"
        )
        # Not a grant question: decompose/compound behaviour must be untouched.
        assert not second_hop.is_multi_field_grant_query(q)
        assert second_hop._facet_targets(q) != []


class TestFacetSpecsUseIdentity:
    def test_field_vocabulary_removed_from_extra_lanes(self):
        specs = _build_facet_specs(Q4_TEN_FIELDS, "unused-bm25", None, {}, max_facets=3)
        assert len(specs) >= 2  # main question + identity lane
        identity_specs = " ".join(q for q, _ in specs[1:])
        # Identity vocabulary present, field vocabulary absent.
        assert "cervell" in identity_specs.lower()
        assert "beca" in identity_specs.lower()
        assert "seguretat" not in identity_specs.lower()
        assert "bdns" not in identity_specs.lower()
        assert "retribuci" not in identity_specs.lower()


class TestReaderFieldQueryMode:
    def _docs(self):
        return [
            {
                "document_id": 1,
                "title": "EXTRACTE beca 25I637",
                "ref": "2026/13043",
                "issue_date": "2026-05-06",
                "doc_subkind": "convocatoria",
                "chunks": [EXTRACT],
            },
            {
                "document_id": 2,
                "title": "RESOLUCIÓ AVI pràctiques",
                "ref": "2024/7597",
                "issue_date": "2024-08-30",
                "doc_subkind": "convocatoria",
                "chunks": [
                    "Llei 27/2011, d'1 d'agost, sobre actualització, adequació i "
                    "modernització del sistema de la Seguretat Social. Els becaris "
                    "seran inclosos en el règim general amb una retribució de 1.000 "
                    "euros i un import global de 500.000 euros per a la convocatòria, "
                    "amb termini de deu dies i requisits de titulació universitària."
                ],
            },
        ]

    def test_no_cross_document_per_field_extras(self, monkeypatch):
        """T4 regression: with field_query=True the reader must not add
        best-chunk-per-keyword evidence from a foreign document (the SS-law
        chunk) when the LLM reader only quoted the target."""

        class _Client:
            def __init__(self, *a, **k):
                pass

            def chat_json(self, messages, temperature=0.0, **kwargs):
                return {
                    "evidence": [
                        {
                            "doc_id": 1,
                            "quote": "L'import global destinat a la concessió d'esta beca és de 1.865,05 euros.",
                            "detail": "Import",
                        }
                    ]
                }

        monkeypatch.setattr(reader, "LlmClient", _Client)
        out = reader.extract_evidence(Q4_TEN_FIELDS, self._docs(), field_query=True)
        assert out, "target evidence must survive"
        assert {item["doc_id"] for item in out} == {1}

    def test_default_mode_unchanged_adds_coverage_extras(self, monkeypatch):
        """Control: without the flag the legacy per-field coverage extras still
        fire (other query classes depend on them)."""

        class _Client:
            def __init__(self, *a, **k):
                pass

            def chat_json(self, messages, temperature=0.0, **kwargs):
                return {
                    "evidence": [
                        {
                            "doc_id": 1,
                            "quote": "L'import global destinat a la concessió d'esta beca és de 1.865,05 euros.",
                            "detail": "Import",
                        }
                    ]
                }

        monkeypatch.setattr(reader, "LlmClient", _Client)
        out = reader.extract_evidence(Q4_TEN_FIELDS, self._docs(), field_query=False)
        assert {item["doc_id"] for item in out} == {1, 2}


class TestExtractReachesReaderWhole:
    def test_t3_import_section_survives_payload(self):
        """T3 regression: with the widened field-query chunk cap the extract's
        'Quart. Import ... 1.865,05 euros' section reaches the reader payload."""
        docs_by_id = {
            1: (
                SimpleNamespace(
                    id=1,
                    title="EXTRACTE beca 25I637",
                    ref="2026/13043",
                    doc_kind="anuncio",
                    doc_subkind="convocatoria",
                    text=EXTRACT,
                ),
                SimpleNamespace(date=None),
            )
        }
        chunk = {"chunk_index": 0, "text": EXTRACT}
        payload = read_mod._build_docs_payload(
            [1],
            docs_by_id,
            {1: [chunk]},
            {},
            {},
            {},
            salient=["modalitat", "complementaries"],
            enumeration_query=False,
            chunk_max_chars=3000,
        )
        joined = " ".join(payload[0]["chunks"])
        assert "1.865,05" in joined
        assert "Termini de presentació" in joined

    def test_default_1200_window_reproduces_the_bug(self):
        """The failure the cap fixes: prefix-biased 1200-char window drops the
        Import section when the question's salient terms all hit the chunk head."""
        windowed = _window_chunk_text(EXTRACT, 1200, ["modalitat", "complementaries", "beca"])
        assert "1.865,05" not in windowed

    def test_field_cue_window_recovers_amount_without_cap(self):
        """Backstop for extracts longer than the widened cap: field cues steer
        the window onto the amount section."""
        long_extract = EXTRACT.replace(
            "Primer. Persones beneficiàries.",
            "Primer. Persones beneficiàries. " + ("Requisits addicionals. " * 60),
        )
        assert len(long_extract) > 3000
        windowed = _window_chunk_text(
            long_extract, 1200, ["modalitat", "complementaries", "import", "euros"]
        )
        assert "1.865,05" in windowed


class TestClaimGuardIgnoresListMarkers:
    """T4 regression: the ten-field answer is a numbered list, and the NEXT
    item's marker ("...440 euros.\n4) Hores...") lands inside the previous
    figure's currency window. The bare '4' then failed the source-presence
    check and the validator dumped a fully correct answer."""

    SOURCE = (
        "Beca amb dedicació de 20 hores setmanals i una retribució bruta mensual "
        "de 440 euros. La duració prevista és de 10 mesos. L'import global "
        "destinat a la concessió d'aquesta beca és de 4.508,40 euros. BDNS 918151."
    )
    LIST_ANSWER = (
        "3) Retribució mensual: 440 euros bruts.\n"
        "4) Hores setmanals: 20 hores.\n"
        "6) Import global: 4.508,40 euros.\n"
        "7) Règim: General.\n"
        "10) Codi BDNS: 918151."
    )

    def test_numbered_list_answer_passes(self):
        from api.answer_validator import _detect_unsupported_claim

        assert not _detect_unsupported_claim(
            answer_text=self.LIST_ANSWER, source_text=self.SOURCE, mode="unit_aware_strict"
        )

    def test_fabricated_figure_still_flagged(self):
        from api.answer_validator import _detect_unsupported_claim

        assert _detect_unsupported_claim(
            answer_text=self.LIST_ANSWER.replace("440", "999"),
            source_text=self.SOURCE,
            mode="unit_aware_strict",
        )


class TestSynthesisAnchoring:
    def test_field_query_prompt_carries_anchor_instruction(self, monkeypatch):
        import api.answer as answer

        captured: dict = {}

        class _Client:
            model = "qwen"

            def __init__(self, *a, **k):
                pass

            def chat_json(self, messages, temperature=0.0, **kwargs):
                captured["messages"] = messages
                return {"answer": "ok", "citations": [1]}

        monkeypatch.setattr(answer, "LlmClient", _Client)
        monkeypatch.setattr(answer.settings, "answer_validator_enabled", False)
        answer.build_answer(
            Q4_TEN_FIELDS,
            "va_va",
            evidence=[{"doc_id": 1, "quote": "beca"}],
            field_query=True,
        )
        system = captured["messages"][0]["content"]
        assert "UNA beca o convocatoria concreta" in system
        assert "NUNCA tomes el valor de otra convocatoria" in system

        answer.build_answer(
            "Quin és el termini de la convocatòria?",
            "va_va",
            evidence=[{"doc_id": 1, "quote": "beca"}],
            field_query=False,
        )
        system = captured["messages"][0]["content"]
        assert "UNA beca o convocatoria concreta" not in system
