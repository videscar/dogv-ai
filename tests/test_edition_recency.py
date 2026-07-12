from __future__ import annotations

from datetime import date

from api.edition_recency import group_editions, stale_edition_ids, titles_veto_pair

# Real DOGV titles from the 2026-07-12 grounded-probe RC1 false-family cases.
TITLE_NOMBRA = (
    "RESOLUCIÓN de 10 de febrero de 2025, por la que se nombra director de la "
    "Oficina Autonómica de Auditoría e Inspección Sanitaria de la Comunitat Valenciana."
)
TITLE_CREA = (
    "RESOLUCIÓN de 5 de febrero de 2025, por la que se crea la Oficina Autonómica "
    "de Auditoría e Inspección Sanitaria de la Comunitat Valenciana."
)
TITLE_MODIFICA_ADSCRIPCION = (
    "RESOLUCIÓN de 5 de febrero de 2026, por la cual se modifica la adscripción de la "
    "Oficina Autonómica de Auditoría e Inspección Sanitaria de la Comunitat Valenciana."
)
TITLE_ADJUDICA_9_24 = (
    "RESOLUCIÓN de 22 de diciembre de 2025, de la Dirección General de Función Pública "
    "por la que se adjudican destinos a quienes han superado las pruebas selectivas de "
    "acceso al cuerpo superior facultativo de acción social, administración de servicios "
    "sociales y sociosanitarios, escala medicina A1-07-03, cuerpo especial, convocatoria "
    "9/24, turno libre y personas con discapacidad, por el sistema de oposición, "
    "correspondientes a la oferta de empleo público de 2024 para personal de la "
    "Administración de la Generalitat."
)
TITLE_ADJUDICA_7_24 = (
    "RESOLUCIÓN de 12 de mayo de 2026, de la Dirección General de Función Pública, por "
    "la que se adjudican destinos a las personas que han superado las pruebas selectivas "
    "de acceso al cuerpo superior facultativo de acción social, administración de "
    "servicios sociales y sociosanitarios, escala en psicología, A1-07-02, cuerpo "
    "especial, convocatoria 7/24, turno libre y personas con diversidad funcional, por "
    "el sistema de oposición, y convocatoria 8/24, promoción interna, por el sistema de "
    "concurso-oposición, correspondientes a la oferta de empleo público de 2024, para "
    "personal de la Administración de la Generalitat."
)
TITLE_MODIFICA_FECHA_29_24 = (
    "RESOLUCIÓN de 10 de junio de 2026, de la Dirección General de Función Pública, por "
    "la que se modifica la fecha de toma de posesión de la convocatoria 29/24, escala de "
    "atención sociosanitaria, C1-04-01, correspondiente a la oferta pública de empleo de "
    "2024 para personal de la Administración de la Generalitat."
)
TITLE_CODI_CONVOCA = (
    "RESOLUCIÓ de 7 d’agost de 2024, per la qual es convoca un programa de capacitació "
    "digital en el marc del programa C19.I01.P04, Competències digitals per a la "
    "infància (CODI), dirigit a l’alumnat entre 9 i 17 anys de centres docents de "
    "titularitat de la Generalitat, dins del Pla de recuperació, transformació i "
    "resiliència, per al curs 2024-2025."
)
TITLE_CODI_ASSIGNA = (
    "RESOLUCIÓ de 28 de febrer de 2025, de la Direcció General d'Innovació i Inclusió "
    "Educativa, per la qual s'assignen accions formatives addicionals per al "
    "desenrotllament del Programa de capacitació digital en el marc del programa "
    "C19.I01.P04, Competències digitals per a la infància (CODI)."
)
TITLE_BECAS_2026 = (
    "RESOLUCIÓ de 3 de febrer de 2026, de la Direcció General d'Universitats, per la "
    "qual es convoquen les beques per a la realització d'estudis universitaris, "
    "exempció de taxes, durant el curs acadèmic 2025-2026 en les universitats que "
    "integren el Sistema Universitari Valencià."
)
TITLE_BECAS_2025 = (
    "RESOLUCIÓ de 5 de febrer de 2025, de la Direcció General d'Universitats, per la "
    "qual es convoquen les beques per a la realització d'estudis universitaris, "
    "exempció de taxes, durant el curs acadèmic 2024-2025 en les universitats que "
    "integren el Sistema Universitari Valencià."
)


def test_group_editions_unions_transitively():
    groups = group_editions([(1, 2), (2, 3), (10, 11)])
    as_sets = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    assert as_sets == [[1, 2, 3], [10, 11]]


def test_group_editions_drops_singletons():
    # ids that never pair are not returned as groups
    assert group_editions([]) == []


def test_older_sibling_is_stale_newest_kept():
    pairs = [(112410, 37182)]  # 2026 vs 2025 edition of the same appointment
    dates = {112410: date(2026, 7, 1), 37182: date(2025, 6, 11)}
    assert stale_edition_ids(pairs, dates) == {37182}


def test_three_year_family_keeps_only_newest():
    pairs = [(112447, 39930), (39930, 92220), (112447, 92220)]
    dates = {112447: date(2026, 7, 1), 39930: date(2025, 9, 1), 92220: date(2024, 7, 4)}
    assert stale_edition_ids(pairs, dates) == {39930, 92220}


def test_members_sharing_newest_date_all_kept():
    # two concurrent same-day publications + one older -> only the older is stale
    pairs = [(1, 2), (2, 3), (1, 3)]
    dates = {1: date(2026, 7, 1), 2: date(2026, 7, 1), 3: date(2025, 1, 1)}
    assert stale_edition_ids(pairs, dates) == {3}


def test_unknown_date_never_marked_stale():
    pairs = [(1, 2)]
    dates = {1: date(2026, 7, 1), 2: None}
    assert stale_edition_ids(pairs, dates) == set()


def test_veto_act_type_conflict_nomination_family():
    # Nomination, creation, and adscription-change of the SAME office are three
    # different acts — none of these may pair, breaking the transitive chain that
    # suppressed the nomination doc (probe P03).
    assert titles_veto_pair(TITLE_NOMBRA, TITLE_MODIFICA_ADSCRIPCION)
    assert titles_veto_pair(TITLE_CREA, TITLE_MODIFICA_ADSCRIPCION)
    assert titles_veto_pair(TITLE_NOMBRA, TITLE_CREA)


def test_veto_code_conflict_parallel_convocatorias():
    # Same act verb ("adjudican"), near-identical boilerplate, but convocatoria
    # 9/24 vs 7/24 and scale A1-07-03 vs A1-07-02 are parallel processes (probe P09).
    assert titles_veto_pair(TITLE_ADJUDICA_9_24, TITLE_ADJUDICA_7_24)


def test_veto_act_type_conflict_follow_up_act():
    # "se adjudican destinos" vs "se modifica la fecha de toma de posesión":
    # a follow-up act about a different convocatoria must not suppress the
    # adjudication (probe P09's kept-vs-dropped pair).
    assert titles_veto_pair(TITLE_ADJUDICA_9_24, TITLE_MODIFICA_FECHA_29_24)


def test_veto_program_family_convoca_vs_assigna():
    # CODI: the convocatoria and a later assignment resolution of the same program
    # are different acts (probes P13/P14 — RC1 dropped 8 of 10 pool docs here).
    assert titles_veto_pair(TITLE_CODI_CONVOCA, TITLE_CODI_ASSIGNA)


def test_no_veto_for_annual_reeditions():
    # RC1's core case must keep working: same act verb, identifiers differ only in
    # course-year tokens -> still a sibling pair, stale year suppressed.
    assert not titles_veto_pair(TITLE_BECAS_2026, TITLE_BECAS_2025)


def test_no_veto_for_provisional_vs_definitiva():
    # RC2-adjacent precedence pairs share the act verb ("publica") and carry the
    # same process code; they must remain pairable.
    provisional = (
        "RESOLUCIÓN de 7 de abril de 2025, por la que se publica la relación "
        "provisional de personas aprobadas de la convocatoria 9/24, escala medicina "
        "A1-07-03."
    )
    definitiva = (
        "RESOLUCIÓN de 7 de julio de 2025, por la que se publica la relación "
        "definitiva de personas que han superado las pruebas selectivas de la "
        "convocatoria 9/24, escala medicina A1-07-03."
    )
    assert not titles_veto_pair(provisional, definitiva)


def test_damage_cap_skips_family_collapse(monkeypatch):
    # A "family" swallowing most of the pool is a thematic cluster, not editions:
    # suppression must back off entirely (probes P13/P14: 6-8 of 10 docs dropped).
    from api import edition_recency as er

    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    candidates = [{"document_id": i, "issue_date": date(2024, 1, i), "title": None} for i in ids]
    pairs = [(i, 10) for i in range(1, 10)]  # everything pairs with the newest
    monkeypatch.setattr(er, "edition_sibling_pairs", lambda *a, **k: pairs)
    kept, dropped = er.suppress_stale_editions(
        None, ids, candidates, sim_threshold=0.86, scan_n=12, max_drops=3
    )
    assert kept == ids and dropped == set()
    # Under the cap the same mechanism still prunes small true families.
    monkeypatch.setattr(er, "edition_sibling_pairs", lambda *a, **k: [(1, 10), (2, 10)])
    kept, dropped = er.suppress_stale_editions(
        None, ids, candidates, sim_threshold=0.86, scan_n=12, max_drops=3
    )
    assert dropped == {1, 2}


def test_no_veto_when_signal_missing():
    # Conservative default: unparseable/one-sided titles keep the old behavior.
    assert not titles_veto_pair(None, TITLE_NOMBRA)
    assert not titles_veto_pair("ANUNCI sense verb estandard", "Un altre anunci")
    # One side has codes, the other none -> ambiguous, no veto.
    assert not titles_veto_pair(TITLE_ADJUDICA_9_24, TITLE_BECAS_2026[:60] + ".")
