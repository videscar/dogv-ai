"""Regression tests for multi-field grant-query anchoring (api/field_anchor.py).

Four verified failures (UJI beca extracts, 2026-07): near-identical template
siblings cross-contaminated because the pipeline retrieved and selected evidence
per FIELD (import, hores, termini...) instead of per DOCUMENT. Each test class
pins one failure's deterministic mechanism.
"""

from __future__ import annotations

from api.field_anchor import (
    asked_fields,
    field_cue_terms,
    identity_query,
    is_multi_field_grant_query,
)

# The four failing queries, verbatim.
Q1_QUIMICA = (
    "Estic acabant el grau de Química a la UJI i he superat més de la meitat dels "
    "crèdits. Hi ha alguna beca oberta per iniciar-me en la investigació sobre "
    "plaguicides o aigües? Si n'hi ha, dis-me qui hi pot optar, l'import, el "
    "termini i on són les bases."
)
Q2_MASTER = (
    "Faig el màster en Química Aplicada i Farmacològica a la UJI. Hi ha una beca "
    "d'iniciació a la investigació a la qual em puga presentar: requisits, quants "
    "diners, quant dura, quantes hores i quan acaba el termini?"
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


class TestGateFiresOnTheFourFailures:
    def test_t1_chemistry_degree(self):
        assert is_multi_field_grant_query(Q1_QUIMICA)
        assert {"import", "termini", "requisits", "bases"} <= asked_fields(Q1_QUIMICA)

    def test_t2_master_query(self):
        assert is_multi_field_grant_query(Q2_MASTER)
        assert {"import", "durada", "hores", "termini", "requisits"} <= asked_fields(Q2_MASTER)

    def test_t3_comparison_case(self):
        # Shared project title across near-identical siblings: the gate must
        # still fire so siblings are presented separately with their own values.
        assert is_multi_field_grant_query(Q3_MODALITAT)
        assert {"import", "departament"} <= asked_fields(Q3_MODALITAT)

    def test_t4_ten_field_single_document(self):
        assert is_multi_field_grant_query(Q4_TEN_FIELDS)
        fields = asked_fields(Q4_TEN_FIELDS)
        assert {
            "requisits",
            "finalitat",
            "import",
            "hores",
            "durada",
            "seguretat_social",
            "termini",
            "tramitar",
            "bdns",
        } <= fields


class TestIdentityQueryStripsFieldClauses:
    """The identity query is what retrieval anchors on: it must keep the
    document-identifying content and drop the field-request vocabulary that
    made field-matching noise docs (a Social-Security law, credit transfers)
    outrank the target extract."""

    def test_t4_numbered_field_list_stripped(self):
        identity = identity_query(Q4_TEN_FIELDS)
        assert "Cervell i Conducta" in identity
        assert "beca de Psicologia" in identity
        for leaked in ("retribució", "hores", "durada", "Seguretat Social", "BDNS", "termini"):
            assert leaked.lower() not in identity.lower()

    def test_t1_field_request_sentence_dropped(self):
        identity = identity_query(Q1_QUIMICA)
        assert "plaguicides" in identity
        assert "grau de Química" in identity
        for leaked in ("import", "termini", "bases", "qui hi pot optar"):
            assert leaked not in identity.lower()

    def test_t2_field_tail_dropped(self):
        identity = identity_query(Q2_MASTER)
        assert "Química Aplicada i Farmacològica" in identity
        assert "beca d'iniciació" in identity
        for leaked in ("quants diners", "quantes hores", "termini"):
            assert leaked not in identity.lower()

    def test_t3_keeps_quoted_project_title(self):
        identity = identity_query(Q3_MODALITAT)
        assert "Modalitat A" in identity
        assert "ajudes complementàries" in identity
        assert "cobra" not in identity.lower()

    def test_returns_empty_when_nothing_identity_like_survives(self):
        assert identity_query("import? termini?") == ""
        assert identity_query("") == ""


class TestGateStaysOffForOtherQueryClasses:
    """Preserve behaviour: existing pipelines (enumeration, comparison,
    single-field amount questions, non-grant retributive tables) are untouched."""

    def test_single_field_amount_question(self):
        q = "Quin és l'import global de la convocatòria de beques de la UJI?"
        assert not is_multi_field_grant_query(q)

    def test_non_grant_multi_field(self):
        # eval v2-001 style: retributive tables, no beca/ajuda/subvenció noun.
        q = (
            "¿Cuál es el importe del sueldo base mensual del subgrupo A1 según las "
            "tablas retributivas de la Generalitat de 2026, y qué subida salarial aplican?"
        )
        assert not is_multi_field_grant_query(q)

    def test_multi_reference_comparison(self):
        q = (
            "Compara l'import i el termini de les ajudes de l'Ordre 23/2026 i "
            "de l'Ordre 18/2026 de la Conselleria d'Educació."
        )
        assert not is_multi_field_grant_query(q)

    def test_enumeration_query(self):
        q = (
            "Cita todas las convocatorias de ayudas publicadas en mayo de 2026 "
            "con su importe y plazo."
        )
        assert not is_multi_field_grant_query(q)

    def test_eval_set_blast_radius_is_zero(self):
        import json
        from pathlib import Path

        eval_path = Path(__file__).resolve().parents[1] / "data/eval_v2/eval_set_v2.jsonl"
        if not eval_path.exists():
            return
        fired = [
            row["id"]
            for row in map(json.loads, eval_path.read_text().splitlines())
            if is_multi_field_grant_query(row["question"])
        ]
        assert fired == []


class TestFieldCueTerms:
    def test_cues_cover_asked_fields_and_are_folded(self):
        cues = field_cue_terms(asked_fields(Q4_TEN_FIELDS))
        assert cues, "ten-field query must produce window cues"
        joined = " ".join(cues)
        assert joined == joined.lower()
        # At least the amount cue must be present: the T3 false-negative came
        # from the chunk window cutting the 'Import' section of the extract.
        assert any("import" in c or "retribucio" in c for c in cues)
