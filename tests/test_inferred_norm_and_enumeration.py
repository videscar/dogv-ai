"""Q#1 (infer a no-number named law from corpus titles) + Q#30 (enumeration
parsing) — offline, no DB / network.

Q#1: a foundational law asked for by name only ("la Ley de Transparencia") is
recovered by reading how in-window norms cite it. Q#30: "list all dispositions of
mayo 2026" is detected as an enumeration so the SQL augment can fire.
"""

from __future__ import annotations

from api.dogv_resolver import _infer_principal_ref, infer_reference_from_corpus
from api.enumeration import parse_enumeration, is_enumeration_query
from datetime import date


# Real DOGV title shapes: the principal a doc modifies/develops is named right
# after its number, while the doc's own leading number is not. The generic term
# "publica" recurs across many law names, so IDF must let the distinctive term
# ("transparencia") single out the principal — exactly the trap that made a naive
# count pick the wrong law.
TITLES_TRANSPARENCIA = [
    "LEY 4/2024, de 26 de julio, de la Generalitat, de modificación de la Ley 1/2022, "
    "de 13 de abril, de Transparencia y Buen Gobierno de la Comunidad Valenciana, y de "
    "la Ley 8/2016, de 28 de octubre, de Incompatibilidades y Conflictos de Intereses",
    "ANUNCIO por el cual se somete a información a la propuesta de modificación de "
    "la Ley 1/2022, de 13 de abril, de la Generalitat, de transparencia y buen gobierno",
    "RESOLUCIÓN por la que se autoriza un crédito previsto en la Ley 1/2015, de 6 de "
    "febrero, de Hacienda Pública del sector público instrumental.",
    "ACUERDO en aplicación de la Ley 10/2014, de 29 de diciembre, de Salud Pública.",
    "RESOLUCIÓN en desarrollo de la Ley 7/2017, de 30 de marzo, de Función Pública.",
]


def test_infer_principal_picks_distinctive_topic_over_generic():
    # "transparencia" (low-df) must win over "publica" (high-df, in Hacienda/Salud/
    # Función Pública); a frequency-only score would wrongly pick a "publica" law.
    ref = _infer_principal_ref(
        TITLES_TRANSPARENCIA, "ley", ["transparencia", "publica"]
    )
    assert ref is not None
    assert ref.num_year == "1/2022"
    assert ref.tipo == "ley"


def test_infer_principal_none_without_topic_hit():
    ref = _infer_principal_ref(TITLES_TRANSPARENCIA, "ley", ["urbanismo", "costas"])
    assert ref is None


def test_infer_principal_bails_on_tie():
    titles = [
        "DECRETO 9/2020, de 3 de enero, de medidas de salud pública.",
        "DECRETO 7/2020, de 3 de enero, de medidas de salud pública.",
    ]
    # Two distinct numbers tie on the same single topic term -> ambiguous -> None.
    assert _infer_principal_ref(titles, "decreto", ["salud", "publica"]) is None


def test_infer_reference_from_corpus_skips_numbered_question():
    # A numbered question is handled by parse_reference, not inference.
    class _DB:
        def execute(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("should not query for a numbered question")

    assert infer_reference_from_corpus(_DB(), "¿Qué dice el Decreto 3/2020?") is None


def test_enumeration_detects_list_query():
    spec = parse_enumeration(
        "Necesito que me cites las disposiciones donde figuran las ofertas de empleo "
        "público del grupo A1 que se han publicado en mayo de 2026"
    )
    assert spec is not None
    assert spec.date_start == date(2026, 5, 1)
    assert spec.date_end == date(2026, 5, 31)
    assert spec.group_codes == ["A1"]


def test_enumeration_ignores_single_norm_query():
    assert parse_enumeration(
        "¿Qué dice la Ley de Transparencia sobre el acceso a la información pública?"
    ) is None
    # A month without a list cue is not an enumeration.
    assert not is_enumeration_query("¿Qué decreto regula las ayudas de mayo de 2026?")
