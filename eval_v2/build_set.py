"""Build the v2 hard eval suite (grounded in real DOGV corpus content).

Emits:
  data/eval_v2/eval_set_v2.jsonl   - full suite (100 items), human-readable record per line
  data/eval_v2/retrieval_input.json - answerable subset in run_eval.py format
  data/eval_v2/answer_input.json    - all items for the answer/abstention judge

Gold is scored at DOCUMENT level (the production pipeline ranks documents);
gold_chunk is recorded for traceability only.
"""
from __future__ import annotations

import json
import os
from collections import Counter

# ref -> (es_id, va_id), verified against the DB
TW = {
    "2025/23721": (15679, 37580), "2025/24021": (15685, 37586), "2025/34001": (17892, 39793),
    "2025/3669": (11627, 33528), "2025/40814": (18712, 40613), "2025/45563": (20033, 41934),
    "2025/50491": (21226, 43127), "2026/10726": (83751, 83987), "2026/10892": (84204, 84494),
    "2026/11981": (84414, 84704), "2026/12103": (86845, 89337), "2026/12145": (86084, 88576),
    "2026/12208": (86079, 88571), "2026/12314": (86180, 88672), "2026/12435": (86274, 88766),
    "2026/12483": (86200, 88692), "2026/12806": (86267, 88759), "2026/13383": (86371, 88863),
    "2026/13421": (86380, 88872), "2026/13949": (86667, 89159), "2026/14121": (86529, 89021),
    "2026/14441": (86609, 89101), "2026/14662": (86629, 89121), "2026/15721": (86763, 89255),
    "2026/16662": (86839, 89331), "2026/16864": (86981, 89473), "2026/16922": (87172, 89664),
    "2026/17382": (87141, 89633), "2026/17649": (87140, 89632), "2026/3145": (48017, 48415),
    "2026/4408": (48743, 49082), "2026/5962": (76628, 76674), "2026/6301": (76609, 76655),
    "2026/7581": (85055, 87547), "2026/8487": (85222, 87714), "2026/9064": (85457, 87949),
    "2026/9140": (85343, 87835), "2026/9192": (85603, 88095), "2026/9267": (85489, 87981),
}
KIND = {
    "2025/23721": "Premios", "2025/24021": "Premios", "2025/34001": "Becas", "2025/3669": "Ayudas",
    "2025/40814": "Becas", "2025/45563": "Becas", "2025/50491": "Premios", "2026/10726": "Otros",
    "2026/10892": "Empleo Publico", "2026/11981": "Ayudas", "2026/12103": "Otros", "2026/12145": "Becas",
    "2026/12208": "Ayudas", "2026/12314": "Empleo Publico", "2026/12435": "Empleo Publico",
    "2026/12483": "Empleo Publico", "2026/12806": "Ayudas", "2026/13383": "Otros", "2026/13421": "Subvenciones",
    "2026/13949": "Becas", "2026/14121": "Empleo Publico", "2026/14441": "Otros", "2026/14662": "Otros",
    "2026/15721": "Ayudas", "2026/16662": "Otros", "2026/16864": "Otros", "2026/16922": "Empleo Publico",
    "2026/17382": "Premios", "2026/17649": "Subvenciones", "2026/3145": "Empleo Publico", "2026/4408": "Subvenciones",
    "2026/5962": "Subvenciones", "2026/6301": "Subvenciones", "2026/7581": "Empleo Publico", "2026/8487": "Premios",
    "2026/9064": "Subvenciones", "2026/9140": "Otros", "2026/9192": "Subvenciones", "2026/9267": "Otros",
}
SRC = {  # text_source per ref (annex/pdf-fallback flag)
    "2026/15721": "pdf", "2026/12806": "pdf", "2025/3669": "pdf", "2026/12208": "html", "2026/11981": "pdf",
    "2026/17649": "html", "2026/13421": "pdf", "2026/6301": "pdf", "2026/5962": "html", "2026/9192": "html",
    "2026/4408": "html", "2026/9064": "html", "2025/40814": "html", "2026/13949": "html", "2025/34001": "pdf",
    "2025/45563": "html", "2026/12145": "pdf", "2025/23721": "pdf", "2025/50491": "pdf", "2026/17382": "html",
    "2025/24021": "html", "2026/8487": "html", "2026/12314": "pdf", "2026/12435": "pdf", "2026/16922": "html",
    "2026/14121": "html", "2026/3145": "pdf", "2026/12483": "pdf", "2026/10892": "pdf", "2026/7581": "pdf",
    "2026/12103": "html", "2026/16662": "html", "2026/14441": "html", "2026/14662": "html", "2026/16864": "html",
    "2026/9267": "html", "2026/10726": "html", "2026/9140": "html", "2026/13383": "html",
}

ITEMS: list[dict] = []


def gid(ref: str, lang: str) -> int:
    return TW[ref][0 if lang == "es" else 1]


def add(qid, lang, cat, pert, q, expected, refs, *, both=False, chunk="", abstain=False,
        kind=None, src=None):
    """refs: list of ref strings. both=True -> all required (one gold set); else any-of."""
    if abstain:
        gold_sets, doc_ids, dkind, dsrc = [], [], None, None
    else:
        ids = [gid(r, lang) for r in refs]
        if both:
            gold_sets = [ids]
        else:
            gold_sets = [[i] for i in ids]
        doc_ids = ids
        dkind = kind or KIND[refs[0]]
        dsrc = src or SRC[refs[0]]
    ITEMS.append({
        "id": qid, "language": lang, "category": cat, "perturbation": pert,
        "question": q, "expected_answer": None if abstain else expected,
        "should_abstain": abstain, "gold_sets": gold_sets, "doc_ids": doc_ids,
        "gold_refs": [] if abstain else refs, "gold_chunk": chunk,
        "doc_kind": dkind, "text_source": dsrc,
    })


# ============================ CLEAN / DIRECT (specific facts) ============================
add("v2-001", "es", "clean", "none",
    "¿Cuál es el importe del sueldo base mensual del subgrupo A1 según las tablas retributivas de la Generalitat de 2026, y qué subida salarial aplican?",
    "Sueldo base mensual del A1 = 1.366,74 €. La subida salarial consolidable es del 2,5% respecto a 31/12/2024.",
    ["2026/3145"], chunk="chunk 4 (tabla A1=1.366,74); chunk 1 (subida 2,5%)")
add("v2-002", "va", "clean", "none",
    "Quants llocs de treball té la relació de llocs de treball (RLT) de l'Entitat de Sanejament d'Aigües (EPSAR) per a l'exercici 2026?",
    "64 llocs de naturalesa laboral (es ratifica la RLT de 2025, sense modificacions).",
    ["2026/12435"], chunk="chunk 1 (64 puestos)")
add("v2-003", "es", "clean", "none",
    "¿Cuántas modalidades de competición y cuántos docentes expertos se van a seleccionar para los campeonatos de FP CVSkills 2027?",
    "33 modalidades de competición y, por tanto, 33 docentes expertos/as.",
    ["2026/12314"], chunk="chunk 2 (33 modalidades / 33 docentes)")
add("v2-004", "va", "clean", "none",
    "Quina dotació total tenen les subvencions per a la producció d'obres audiovisuals a la Comunitat Valenciana (anualitats 2026-2028) en la modalitat de ficció?",
    "Modalitat ficció: 4.860.000 €. Llargmetratges A: 1.000.000 € (màxim 500.000 € per ajuda). Videojocs: 30.000 €.",
    ["2026/5962"], chunk="chunk 4 (ficción 4.860.000)")
add("v2-005", "es", "clean", "none",
    "¿De cuánto es el premio individual de excelencia académica concedido en diciembre de 2025 al alumnado de grado del sistema universitario valenciano?",
    "3.000,00 € por persona premiada (Orden 12/2022).",
    ["2025/50491"], chunk="chunk 1 (premio 3.000 €)")
add("v2-006", "va", "clean", "none",
    "Quina és la quantia total de les subvencions concedides a les entitats dels XLIV Jocs Esportius de la Comunitat Valenciana en el nivell d'iniciació al rendiment per al curs 2025-2026?",
    "Import total de 500.000 € (línia de subvenció S0124).",
    ["2026/17649"], chunk="chunk 1 (500.000 euros)")
add("v2-007", "es", "clean", "none",
    "¿Qué importe global máximo tienen las becas GV-Talent para la excelencia académica convocadas en abril de 2026, y de cuánto es cada beca?",
    "Importe global máximo 1.600.000 €. La beca es de 800 € (renta entre umbral 1 y 2) o 400 € (renta superior al umbral 2).",
    ["2026/12145"], chunk="chunk 4 (1.600.000 €; 800/400)")
add("v2-008", "va", "clean", "none",
    "Quins són els drets d'examen (taxa) de les proves selectives dels subgrups A1 i A2 de l'oferta d'ocupació pública de 2026 per concurs de mèrits?",
    "A1: 30,57 €; A2: 25,48 €. Bonificació del 10% per presentació telemàtica.",
    ["2026/14121"], chunk="chunk 6 (30,57 / 25,48; bonif 10%)")
add("v2-009", "es", "clean", "none",
    "¿A quién se ha nombrado gerente de la Universitat de València en la resolución de abril de 2026?",
    "Daniel González Serisola.",
    ["2026/14662"], chunk="chunk 0 (Daniel González Serisola)")
add("v2-010", "va", "clean", "none",
    "Quin import global màxim tenen les beques per a l'alumnat que finalitze els estudis universitaris durant el curs 2025-2026, convocades el maig de 2026?",
    "180.000 € per a l'exercici 2026 (línia de subvenció S0225).",
    ["2026/13949"], chunk="chunk 1 (180.000 €)")

# ============================ ANNEX-SPECIFIC (PDF fallback) ============================
add("v2-011", "es", "annex", "annex_detail",
    "En la concesión de ayudas del programa Código Escuela 4.0 para robótica en centros concertados (mayo 2026), ¿qué importe se concedió al Centre Privat Pureza de María - Grao de València?",
    "36.065,80 € concedidos (de 45.600,00 € solicitados, 38 unidades concertadas).",
    ["2026/15721"], chunk="anexo I, chunk 4 (Pureza de María Grao 36.065,80)")
add("v2-012", "va", "annex", "annex_detail",
    "Segons la resolució d'assignació del Fons de Cooperació Municipal 2026, quina quantitat li correspon al municipi d'Alaquàs?",
    "261.560 € (Albaida 94.382 €, Albal 180.611 € apareixen al mateix annex).",
    ["2026/12806"], chunk="anexo, chunk 5 (Alaquàs 261.560 €)")
add("v2-013", "es", "annex", "annex_detail",
    "En los premios de la Generalitat para deportistas por su clasificación olímpica y paralímpica de 2025, ¿qué premio recibió Néstor Abad Sanjuán (gimnasia artística)?",
    "6.000 € (anexo I de la Resolución de 19 de junio de 2025).",
    ["2025/23721"], chunk="anexo I, chunk 2 (ABAD SANJUAN NESTOR 6.000 €)")
add("v2-014", "va", "annex", "annex_detail",
    "En les beques academicodeportives per a esportistes d'elit de 2025, quina quantia va rebre Jairo Agenjo Trigos (taekwondo)?",
    "2.250 € (annex de la Resolució de 19 de setembre de 2025).",
    ["2025/40814"], chunk="anexo, chunk 1 (AGENJO TRIGOS JAIRO TAEKWONDO 2.250 €)")
add("v2-015", "es", "annex", "annex_detail",
    "En las ayudas para instalaciones de recogida de residuos (NextGenerationEU) concedidas en abril de 2026, ¿qué porcentaje de ayuda se aplicó y qué importe se otorgó al primer expediente del Consorcio Plan Zonal de Residuos Zona I?",
    "Ayuda del 90%. Al expediente MAECOP20250001 (Consorcio Zona I): solicitado 758.679,57 €, otorgado 670.301,61 €.",
    ["2026/12208"], chunk="chunk 1 (90%; 758.679,57 -> 670.301,61)")
add("v2-016", "va", "annex", "annex_detail",
    "En la concessió d'ajudes DANA al lloguer de febrer de 2025, quin import total es va comprometre amb càrrec a la línia de subvenció habilitada?",
    "3.651.200,00 € (línia S1892, Decret 167/2024).",
    ["2025/3669"], chunk="chunk 3 (3.651.200,00 €)")
add("v2-017", "es", "annex", "annex_detail",
    "En las becas para estudiantes de la UNED en la Comunitat Valenciana del curso 2024-2025, ¿qué importe total se libró a favor de la UNED y cuál era el importe global máximo de la convocatoria?",
    "Importe global máximo 75.000 €; se libró a la UNED un total de 51.637,24 € por las tasas de matrícula.",
    ["2025/34001"], chunk="chunk 2 (75.000; 51.637,24)")
add("v2-018", "va", "annex", "annex_detail",
    "Quin import màxim total tenen les ajudes LEADER «Ticket Rural» per a la creació de microempreses no agràries (segon període de la convocatòria 2025)?",
    "Import màxim 22.255.556,00 € (Ordre 5/2025); cofinançament del 60% pel FEADER.",
    ["2026/11981"], chunk="chunk 1 (22.255.556,00; 60% FEADER)")
add("v2-019", "es", "annex", "annex_detail",
    "En la convocatoria de subvenciones para la promoción del valenciano en el ámbito festivo de 2026, ¿qué dotación total tienen los libros de Moros y Cristianos y cuántas subvenciones se convocan?",
    "25 subvenciones para libros de Moros y Cristianos, con una dotación total de 46.000 €.",
    ["2026/4408"], chunk="chunk 6 (Moros y Cristianos 25 / 46.000 €)")
add("v2-020", "va", "annex", "annex_detail",
    "Segons la resolució de concessió d'ajudes en matèria de comerç, artesania i consum (CMIART) de febrer de 2026, fins a quina data han de justificar les persones beneficiàries i quin percentatge de suport s'aplica?",
    "Han de justificar fins al 10 de març de 2026; el percentatge de suport és del 50%.",
    ["2026/6301"], chunk="chunk 2 (hasta 10 marzo 2026); chunk 4 (50%)")

# ============================ VAGUE / UNDER-SPECIFIED ============================
add("v2-021", "es", "vague", "underspecified",
    "¿Ha salido algo sobre las oposiciones de la Generalitat de este año?",
    "Sí: la OPE 2026 (Decreto 16/2026). Hay convocatorias de los subgrupos A1/A2 por concurso de méritos (Orden 23/2026) y A1 por oposición (Orden 18/2026).",
    ["2026/14121", "2026/12483"], chunk="convocatorias OPE 2026")
add("v2-022", "va", "vague", "underspecified",
    "Hi ha alguna ajuda nova per als pobles?",
    "Sí, l'assignació del Fons de Cooperació Municipal 2026 als municipis i entitats locals (Llei 5/2021, Decret 110/2025).",
    ["2026/12806"], chunk="Fondo Cooperación Municipal 2026")
add("v2-023", "es", "vague", "underspecified",
    "Quería información sobre las becas universitarias.",
    "Hay varias: becas para alumnado que finaliza estudios (180.000 €, curso 2025-2026) y becas GV-Talent de excelencia (1.600.000 €).",
    ["2026/13949", "2026/12145"], chunk="becas universitarias 2026")
add("v2-024", "va", "vague", "underspecified",
    "Volia saber sobre els premis de les falles.",
    "Els premis President i Generalitat a les falles grans i infantils de 2026; el premi President està dotat amb 5.000 €.",
    ["2026/17382"], chunk="premios fallas 2026")
add("v2-025", "es", "vague", "underspecified",
    "¿Hay ayudas para el campo o para emprender en pueblos pequeños?",
    "Sí, las ayudas LEADER 'Ticket Rural' para la creación de microempresas no agrarias (Orden 5/2025), con un máximo de 22.255.556 €.",
    ["2026/11981"], chunk="LEADER Ticket Rural")
add("v2-026", "va", "vague", "underspecified",
    "Què hi ha publicat sobre subvencions de cultura o música?",
    "Subvencions de l'IVC per al foment d'activitats musicals 2026 i per a la producció d'obres audiovisuals 2026-2028.",
    ["2026/9064", "2026/5962"], chunk="IVC subvenciones cultura")
add("v2-027", "es", "vague", "underspecified",
    "¿Qué hay de los certificados de valenciano?",
    "La JQCV ha convocado las pruebas para los certificados A2, B1, B2... La matrícula de A2 y B1 fue del 20 al 28 de abril.",
    ["2026/9267"], chunk="JQCV pruebas valenciano")
add("v2-028", "va", "vague", "underspecified",
    "M'han dit que han tret diners per a l'esport, és cert?",
    "Sí, subvencions als clubs dels XLIV Jocs Esportius (iniciació al rendiment), per un total de 500.000 €.",
    ["2026/17649"], chunk="Jocs Esportius 500.000")
add("v2-029", "es", "vague", "underspecified",
    "¿Han nombrado a alguien nuevo en la Generalitat últimamente?",
    "Sí, por ejemplo el Decreto 73/2026 nombra a María Isabel Peyró Fernández directora general de Relaciones Institucionales, Seguimiento y Comunicación.",
    ["2026/12103"], chunk="Decreto 73/2026 nombramiento")
add("v2-030", "va", "vague", "underspecified",
    "Busque informació sobre ajudes per a l'habitatge per la DANA.",
    "Les ajudes DANA al lloguer (Decret 167/2024), amb concessió directa; es va comprometre un import de 3.651.200 € (línia S1892).",
    ["2025/3669"], chunk="ayudas DANA alquiler")

# ============================ COLLOQUIAL / TYPOS / NO ACCENTS ============================
add("v2-031", "es", "colloquial", "typo",
    "oye cuanto pagan de beca a los deportistas de elite valencianos? lo del 2025",
    "Las becas academicodeportivas 2025 van por deportista (p. ej. taekwondo 2.250 €, balonmano 2.000 €), según el anexo.",
    ["2025/40814"], chunk="becas deportistas élite 2025")
add("v2-032", "va", "colloquial", "no_accents",
    "quant es el sou base mensual dun A1 de la generalitat este any?",
    "El sou base mensual del subgrup A1 és 1.366,74 € (taules retributives 2026, increment del 2,5%).",
    ["2026/3145"], chunk="A1 1.366,74")
add("v2-033", "es", "colloquial", "typo",
    "porfa los premios de las fayas de este año cuanto son y quien gano la mejor falla grande??",
    "El premio President a la Mejor Falla Grande de la Sección Especial (2026) está dotado con 5.000 €.",
    ["2026/17382"], chunk="premio President fallas 2026 5.000")
add("v2-034", "va", "colloquial", "no_accents",
    "ke ajudes hi ha per a comprar coses de robotica als col-legis concertats?",
    "Les ajudes del programa Codi Escola 4.0 per a l'adquisició d'equipament de robòtica, programació i llenguatge computacional als centres concertats.",
    ["2026/15721"], chunk="Código Escuela 4.0")
add("v2-035", "es", "colloquial", "typo",
    "me podeis decir cuanto cuesta apuntarse a las oposiciones A2 de la gva? la tasa esa",
    "Los derechos de examen del subgrupo A2 son 25,48 € (A1: 30,57 €), con bonificación del 10% por vía telemática.",
    ["2026/14121"], chunk="A2 25,48")
add("v2-036", "va", "colloquial", "no_accents",
    "hola volia saber qui han nomenat gerent de la universitat de valencia",
    "Daniel González Serisola ha sigut nomenat gerent de la Universitat de València (abril 2026).",
    ["2026/14662"], chunk="gerente UV")
add("v2-037", "es", "colloquial", "typo",
    "kuantos puestos tiene la plantilla de la epsar (aguas residuales) para el 2026?",
    "64 puestos de naturaleza laboral.",
    ["2026/12435"], chunk="64 puestos")
add("v2-038", "va", "colloquial", "no_accents",
    "fins quan em puc matricular del valencia nivell B1 a la jqcv?",
    "La matrícula dels nivells A2 i B1 va ser del 20 al 28 d'abril (telemàtica).",
    ["2026/9267"], chunk="matrícula 20-28 abril")
add("v2-039", "es", "colloquial", "typo",
    "y las becas pa los que acaban la carrera este año cuanto dinero hay en total?",
    "180.000 € en total para el ejercicio 2026 (becas para alumnado que finaliza estudios universitarios, curso 2025-2026).",
    ["2026/13949"], chunk="180.000")
add("v2-040", "va", "colloquial", "no_accents",
    "qui es el president del consell dadministracio de la corporacio valenciana de mitjans? lacord de maig",
    "Per l'Acord de 19 de maig de 2026 del Consell es disposa el nomenament del president del Consell d'Administració de la CACVSA (Corporació Audiovisual de la Comunitat Valenciana, SA).",
    ["2026/16662"], chunk="nombramiento presidente CACVSA")

# ============================ WRONG / APPROXIMATE REFERENCE ============================
add("v2-041", "es", "wrong_ref", "approx_ref",
    "¿Qué dispone el Decreto 74/2026 del Consell sobre el cese y nombramiento del director general de Relaciones Institucionales?",
    "La referencia correcta es el Decreto 73/2026: cesa a Javier Zaragosí Castelló y nombra a María Isabel Peyró Fernández directora general de Relaciones Institucionales, Seguimiento y Comunicación.",
    ["2026/12103"], chunk="Decreto 73/2026 (no 74)")
add("v2-042", "va", "wrong_ref", "approx_ref",
    "Què diu el Decret 167/2025 sobre les ajudes urgents de lloguer per la DANA?",
    "La norma reguladora és el Decret 167/2024 (no 167/2025). Regula ajudes urgents de lloguer per al temporal del 29 d'octubre de 2024; concessió directa.",
    ["2025/3669"], chunk="Decreto 167/2024 (no 167/2025)")
add("v2-043", "es", "wrong_ref", "approx_ref",
    "Sobre la Orden 22/2026 de oposiciones A1 y A2 por concurso de méritos, ¿cuáles son las tasas de examen?",
    "La convocatoria por concurso de méritos es la Orden 23/2026 (no 22/2026): A1 30,57 €, A2 25,48 €.",
    ["2026/14121"], chunk="Orden 23/2026 (no 22)")
add("v2-044", "va", "wrong_ref", "approx_ref",
    "En la resolució de les ajudes LEADER Ticket Rural emparada en l'Ordre 4/2025, quin és l'import màxim?",
    "Les bases són l'Ordre 5/2025 (no 4/2025). L'import màxim és 22.255.556,00 €.",
    ["2026/11981"], chunk="Orden 5/2025 (no 4)")
add("v2-045", "es", "wrong_ref", "approx_ref",
    "¿Cuánto fue el premio de excelencia académica de la Orden 11/2022 para los graduados universitarios?",
    "Las bases son la Orden 12/2022 (no 11/2022). El premio es de 3.000 € por persona.",
    ["2025/50491"], chunk="Orden 12/2022 (no 11)")
add("v2-046", "va", "wrong_ref", "approx_ref",
    "Quina dotació té la convocatòria de premis a les falles de 2026 segons el Decret 38/2023 del president?",
    "El decret de creació és el 38/2022 (no 38/2023). La convocatòria de 2026 té un import total màxim de 207.000 € (línia S0173).",
    ["2026/8487"], chunk="Decreto 38/2022; 207.000 €")

# ============================ MULTI-HOP / MULTI-INTENT ============================
add("v2-047", "es", "multihop", "multi_doc",
    "Compara las dos vías de acceso de la OPE 2026 de la Generalitat: ¿en qué se diferencian la Orden 23/2026 y la Orden 18/2026 y qué sistema selectivo usa cada una?",
    "La Orden 23/2026 cubre A1 y A2 por concurso de méritos; la Orden 18/2026 cubre A1 por oposición. Ambas para las plazas del Decreto 16/2026 (OPE 2026).",
    ["2026/14121", "2026/12483"], both=True, chunk="ambas órdenes OPE 2026")
add("v2-048", "va", "multihop", "multi_doc",
    "Quant va estar dotat el premi President a les falles en 2025 i en 2026? Ha canviat la quantia?",
    "En 2025 i en 2026 el premi President està dotat amb 5.000 € (no ha canviat).",
    ["2025/24021", "2026/17382"], both=True, chunk="premio President 5.000 ambos años")
add("v2-049", "es", "multihop", "multi_doc",
    "Para 2026, ¿cuál es el importe total de las becas para alumnado que finaliza estudios universitarios y cuál el de las becas GV-Talent? ¿Cuál tiene más presupuesto?",
    "Becas de finalización de estudios: 180.000 €. Becas GV-Talent: 1.600.000 €. Las GV-Talent tienen mucho más presupuesto.",
    ["2026/13949", "2026/12145"], both=True, chunk="180.000 vs 1.600.000")
add("v2-050", "va", "multihop", "multi_doc",
    "Vull comparar dues subvencions de l'IVC de 2026: la d'obres audiovisuals i la d'activitats musicals. Quina finalitat té cadascuna?",
    "Audiovisual (2026-2028): suport a la producció d'obres audiovisuals (ficció 4.860.000 €). Música: foment d'activitats musicals com festivals, concerts i composició d'obres noves.",
    ["2026/5962", "2026/9064"], both=True, chunk="IVC audiovisual vs música")
add("v2-051", "es", "multihop", "multi_doc",
    "¿Qué dos tipos de ayuda relacionadas con la DANA y los municipios aparecen: las de alquiler de vivienda y el Fondo de Cooperación Municipal? Indica el importe comprometido de cada una.",
    "Ayudas DANA al alquiler: 3.651.200 € comprometidos (línea S1892). Fondo de Cooperación Municipal 2026: asignación a municipios (p. ej. Alaquàs 261.560 €).",
    ["2025/3669", "2026/12806"], both=True, chunk="DANA alquiler + Fondo Cooperación")
add("v2-052", "va", "multihop", "multi_doc",
    "En l'àmbit dels nomenaments universitaris de 2026, qui és el nou gerent de la Universitat de València i qui s'ha nomenat vocal del Ple del Consell Valencià d'Universitats?",
    "Gerent de la UV: Daniel González Serisola. Vocals del Ple del Consell Valencià d'Universitats: Pilar Chiva Jordá i Enrique Vidal Pérez.",
    ["2026/14662", "2026/16864"], both=True, chunk="gerente UV + vocales CVU")

# ============================ MORE CLEAN / FACTUAL (balance) ============================
add("v2-053", "es", "clean", "none",
    "¿A quién cesa y a quién nombra el Decreto 73/2026 del Consell para la Dirección General de Relaciones Institucionales, Seguimiento y Comunicación?",
    "Cesa a Javier Zaragosí Castelló (a petición propia) y nombra a María Isabel Peyró Fernández.",
    ["2026/12103"], chunk="cese/nombramiento")
add("v2-054", "va", "clean", "none",
    "Qui ha sigut nomenat/nomenada presidenta del Consell Valencià de Transparència pel Decret 6/2026 del President?",
    "El Decret 6/2026 nomena la presidenta del Consell Valencià de Transparència.",
    ["2026/10726"], chunk="Decreto 6/2026")
add("v2-055", "es", "clean", "none",
    "¿Qué periodo cubre el Plan estratégico de subvenciones de la Conselleria de Medio Ambiente aprobado en abril de 2026 y quién lo firma?",
    "Cubre el periodo 2024-2026; lo firma el subsecretario Juan Martínez Caballero (por delegación).",
    ["2026/13421"], chunk="2024-2026; Juan Martínez Caballero")
add("v2-056", "va", "clean", "none",
    "Quin percentatge de cofinançament del FEDER tenen les subvencions del Sistema Valencià d'Innovació per als exercicis 2026-2028, i quina línia s'hi finança al 100%?",
    "Cofinançament FEDER fins al 60%, excepte la línia pressupostària S2124, que es finança fins al 100%.",
    ["2026/9192"], chunk="chunk 3 (60%; S2124 100%)")
add("v2-057", "es", "clean", "none",
    "¿Cuántas becas de prácticas de turismo (clase G) concedió Turisme Comunitat Valenciana para 2025-2026 y bajo qué decreto de bases?",
    "Ocho becas de clase G, bajo el Decreto 20/2017 (bases reguladoras de becas profesionales en materia de turismo).",
    ["2025/45563"], chunk="ocho becas; Decreto 20/2017")
add("v2-058", "va", "clean", "none",
    "El Decret 3/2026 del President quants vocals nomena i de quin òrgan?",
    "Nomena tres vocals del Consell Rector de l'Agència Valenciana de Seguretat Ferroviària.",
    ["2026/9140"], chunk="tres vocales AVSF")
add("v2-059", "es", "clean", "none",
    "En la convocatoria de oposiciones A1 por el sistema de oposición (Orden 18/2026), ¿cuál es el plazo de presentación de solicitudes?",
    "10 días hábiles desde el día siguiente a la publicación en el DOGV.",
    ["2026/12483"], chunk="chunk 5 (10 días hábiles)")
add("v2-060", "va", "clean", "none",
    "En la selecció de professorat expert per als CVSkills 2027, quin termini hi ha per a enviar la documentació i a quina adreça?",
    "15 dies naturals des de la publicació en el DOGV; la documentació s'envia a cv_skills@gva.es.",
    ["2026/12314"], chunk="chunk 4 (15 días; cv_skills@gva.es)")

# ============================ MORE ANNEX (balance, reach ~14) ============================
add("v2-061", "es", "annex", "annex_detail",
    "Según el anexo de las ayudas DANA al alquiler de febrero de 2025, ¿qué ayuda se concedió a la primera persona beneficiaria del listado, en Alberic?",
    "A Ineta Bardzilauskaite (Alberic), expediente DANALQ/2024/00141: 11.200,00 €.",
    ["2025/3669"], chunk="anexo, chunk 5 (INETA BARDZILAUSKAITE 11.200)")
add("v2-062", "va", "annex", "annex_detail",
    "En la concessió d'ajudes Codi Escola 4.0 (robòtica) de maig de 2026, quin import es va concedir al centre La Malvesia de Llombai?",
    "3.796,40 € concedits (de 4.800,00 € sol·licitats, 4 unitats).",
    ["2026/15721"], chunk="anexo I, chunk 4 (La Malvesia Llombai 3.796,40)")
add("v2-063", "es", "annex", "annex_detail",
    "En el Fondo de Cooperación Municipal 2026, ¿qué cantidad se asigna a Alboraia/Alboraya?",
    "235.896 €.",
    ["2026/12806"], chunk="anexo, chunk 5 (Alboraia 235.896)")
add("v2-064", "va", "annex", "annex_detail",
    "En els premis de classificació olímpica i paralímpica de 2025, quin premi va rebre Claudia Adán Lledó (vela)?",
    "2.500 € (annex I).",
    ["2025/23721"], chunk="anexo I, chunk 2 (ADAN LLEDO CLAUDIA VELA 2.500)")

# ============================ OUT OF SCOPE / ABSTAIN ============================
add("v2-065", "es", "out_of_scope", "abstain",
    "¿Cuál es el plazo para presentar la declaración de la renta (IRPF) de 2025 en la Agencia Tributaria estatal?",
    None, [], abstain=True, chunk="state tax, no en DOGV")
add("v2-066", "va", "out_of_scope", "abstain",
    "Quan ix la convocatòria de bombers forestals de la Diputació de Terol?",
    None, [], abstain=True, chunk="Teruel/Aragón, fuera de la C.Valenciana")
add("v2-067", "es", "out_of_scope", "abstain",
    "¿Qué subvenciones publicará el DOGV en agosto de 2026 para autónomos?",
    None, [], abstain=True, chunk="future/unknowable")
add("v2-068", "va", "out_of_scope", "abstain",
    "Què diu el Decret 999/2026 del Consell sobre les ajudes per a drons agrícoles?",
    None, [], abstain=True, chunk="decreto inexistente")
add("v2-069", "es", "out_of_scope", "abstain",
    "¿Cuál es el horario de atención al público de la oficina de la Seguridad Social en Alicante?",
    None, [], abstain=True, chunk="fuera de ámbito DOGV")
add("v2-070", "va", "out_of_scope", "abstain",
    "Quina nota de tall demanen per a estudiar Medicina a la Universitat de València el curs 2026-2027?",
    None, [], abstain=True, chunk="no es contingut del DOGV")
add("v2-071", "es", "out_of_scope", "abstain",
    "¿Qué tiempo hará mañana en València según la AEMET?",
    None, [], abstain=True, chunk="meteorología, fuera de ámbito")
add("v2-072", "va", "out_of_scope", "abstain",
    "On puc consultar el conveni col·lectiu dels treballadors de Mercadona publicat al DOGV?",
    None, [], abstain=True, chunk="convenio empresa estatal, no DOGV autonómico")
add("v2-073", "es", "out_of_scope", "abstain",
    "¿Cuánto cuesta renovar el DNI electrónico en 2026?",
    None, [], abstain=True, chunk="competencia estatal")
add("v2-074", "va", "out_of_scope", "abstain",
    "Quina resolució del DOGV regula les tarifes del metro de Madrid?",
    None, [], abstain=True, chunk="sense sentit / fora d'àmbit")

# ============================ MORE VAGUE/COLLOQUIAL/CLEAN to reach 100 & balance ============================
add("v2-075", "es", "vague", "underspecified",
    "¿Hay algo sobre premios para deportistas olímpicos valencianos?",
    "Sí, los premios para la clasificación y preparación en juegos olímpicos y paralímpicos de 2025 (importes por deportista en anexo).",
    ["2025/23721"], chunk="premios olímpicos 2025")
add("v2-076", "va", "vague", "underspecified",
    "Què s'ha tret sobre habitatge o lloguer últimament?",
    "Les ajudes DANA al lloguer d'habitatge (Decret 167/2024), de concessió directa.",
    ["2025/3669"], chunk="ayudas alquiler DANA")
add("v2-077", "es", "vague", "underspecified",
    "Información sobre el plan de subvenciones de medio ambiente.",
    "El Plan estratégico de subvenciones de la Conselleria de Medio Ambiente para 2024-2026 (firmado por Juan Martínez Caballero).",
    ["2026/13421"], chunk="plan estratégico 2024-2026")
add("v2-078", "va", "vague", "underspecified",
    "Hi ha ajudes per a fer pel·lícules o videojocs ací?",
    "Sí, les subvencions de l'IVC per a la producció d'obres audiovisuals 2026-2028 (inclou videojocs: 30.000 €).",
    ["2026/5962"], chunk="audiovisual; videojocs 30.000")
add("v2-079", "es", "colloquial", "typo",
    "cuanto dan de premio en los premios de excelencia academica de la universidad??",
    "3.000 € por persona premiada (premios de excelencia académica, diciembre 2025).",
    ["2025/50491"], chunk="3.000 €")
add("v2-080", "va", "colloquial", "no_accents",
    "quina ajuda hi ha per a microempreses no agraries als pobles rurals (leader)?",
    "Les ajudes LEADER 'Ticket Rural' per a la creació de microempreses no agràries (Ordre 5/2025), import màxim 22.255.556 €.",
    ["2026/11981"], chunk="Ticket Rural")
add("v2-081", "es", "colloquial", "typo",
    "los llibrets de falla cuanto pagan de subvencion el primer premio?",
    "El primer premio de los libros de Fallas es de 5.000 € (dotación total de esa modalidad: 88.500 €).",
    ["2026/4408"], chunk="1r premi 5.000; total 88.500")
add("v2-082", "va", "colloquial", "no_accents",
    "qui son els nous vocals del consell valencia duniversitats nomenats al maig?",
    "Pilar Chiva Jordá i Enrique Vidal Pérez (Resolució de 20 de maig de 2026).",
    ["2026/16864"], chunk="Pilar Chiva / Enrique Vidal")
add("v2-083", "es", "clean", "none",
    "¿Qué importe total máximo tiene la convocatoria de los premios President y Generalitat a las fallas de 2026?",
    "207.000 € (línea S0173 del presupuesto de la Generalitat).",
    ["2026/8487"], chunk="207.000 €")
add("v2-084", "va", "clean", "none",
    "Quina puja salarial recullen les taules retributives de la Generalitat de 2026?",
    "Una puja consolidable del 2,5% respecte als imports vigents a 31/12/2024.",
    ["2026/3145"], chunk="2,5%")
add("v2-085", "es", "clean", "none",
    "¿Qué requisito de antigüedad de la obra se exige para las ayudas a la composición de obras musicales nuevas del IVC en 2026?",
    "La obra no debe haber sido grabada, editada ni presentada ante público antes del 1 de diciembre de 2025, y debe estrenarse en el periodo de la convocatoria.",
    ["2026/9064"], chunk="chunk 6 (1 diciembre 2025)")
add("v2-086", "va", "clean", "none",
    "En el concurs ordinari de provisió de llocs reservats a funcionaris d'habilitació nacional (2025 i 2026), qui presidix el tribunal de valoració?",
    "Alberto Colom Alcácer (secretari-interventor de l'Ajuntament de Llosa de Ranes).",
    ["2026/10892"], chunk="chunk 7 (Alberto Colom Alcácer)")
add("v2-087", "es", "clean", "none",
    "¿Qué falla ganó el premio President a la Mejor Falla Infantil de la Sección Especial en 2026?",
    "La falla Espartero-Gran Vía Ramón y Cajal.",
    ["2026/17382"], chunk="chunk 1 (Espartero-Gran Vía)")
add("v2-088", "va", "clean", "none",
    "Quin import global màxim tenien les beques per a estudiants de la UNED a la Comunitat Valenciana del curs 2024-2025?",
    "75.000 € (import global màxim).",
    ["2025/34001"], chunk="75.000")

# wrong_ref extra (balance to ~12)
add("v2-089", "es", "wrong_ref", "approx_ref",
    "¿De cuánto es el sueldo base del A1 según el Acuerdo del Consell 30/2026 de tablas retributivas?",
    "Las tablas retributivas se publican por el Acuerdo de 30 de enero de 2026 (referencia 2026/3145), no '30/2026'. El sueldo base del A1 es 1.366,74 €.",
    ["2026/3145"], chunk="acuerdo 30 enero 2026")
add("v2-090", "va", "wrong_ref", "approx_ref",
    "Segons la Resolució de les beques GV-Talent, l'import global màxim és d'1.600.000 €? Crec que és la convocatòria del curs 2025-2026.",
    "L'import global màxim sí és 1.600.000 €, però la convocatòria és per al curs acadèmic 2024-2025 (no 2025-2026).",
    ["2026/12145"], chunk="curso 2024-2025 (no 2025-2026)")
add("v2-091", "es", "wrong_ref", "approx_ref",
    "En las becas para deportistas de élite del Decreto 59/2016, ¿qué importes se conceden?",
    "Las bases son la Orden 59/2016 (no 'Decreto'). Los importes van por deportista según el anexo (p. ej. 2.250 €, 2.000 €).",
    ["2025/40814"], chunk="Orden 59/2016 (no Decreto)")
add("v2-092", "va", "wrong_ref", "approx_ref",
    "Quins llocs té la RLT de l'EPSAR del 2025? (la resolució d'abril)",
    "La resolució d'abril de 2026 publica la RLT de l'exercici 2026 (ratifica la de 2025): 64 llocs laborals.",
    ["2026/12435"], chunk="ejercicio 2026, 64 puestos")

# multihop extra (balance ~12) + remaining to 100
add("v2-093", "es", "multihop", "multi_doc",
    "¿Qué dos resoluciones de premios de las fallas de 2026 hay, la convocatoria y la concesión, y qué importe tiene el premio President en cada una?",
    "La convocatoria (marzo 2026, total 207.000 €) y la concesión (mayo 2026). El premio President está dotado con 5.000 € en ambas.",
    ["2026/8487", "2026/17382"], both=True, chunk="convocatoria + concesión fallas 2026")
add("v2-094", "va", "multihop", "multi_doc",
    "Per a l'OPE 2026, quina taxa d'examen es paga (Ordre 23/2026) i quin és el sou base del subgrup A1 (taules 2026) si s'aprova la plaça?",
    "Taxa d'examen A1: 30,57 € (A2: 25,48 €). Sou base mensual de l'A1: 1.366,74 €.",
    ["2026/14121", "2026/3145"], both=True, chunk="tasa examen + sueldo A1")
add("v2-095", "es", "multihop", "multi_doc",
    "En materia de becas de 2025 ya resueltas: ¿qué importe global máximo tuvieron las de la UNED y cuánto se concedió por deportista de élite? Cita ambas convocatorias.",
    "Becas UNED 2024-2025: importe global máximo 75.000 €. Becas deportistas de élite 2025: importes por persona en anexo (p. ej. 2.250 €).",
    ["2025/34001", "2025/40814"], both=True, chunk="UNED + deportistas élite")

# remaining clean/colloquial to reach 100 with language balance
add("v2-096", "va", "colloquial", "no_accents",
    "kuant de temps tinc per a justificar la subvencio de comerc i artesania cmiart?",
    "Fins al 10 de març de 2026 (resolució de concessió de febrer de 2026).",
    ["2026/6301"], chunk="hasta 10 marzo 2026")
add("v2-097", "es", "colloquial", "typo",
    "y para innovacion y doctorandos en empresas hay subvenciones? cuanto presupuesto",
    "Sí, subvenciones del Sistema Valenciano de Innovación 2026-2028; p. ej. doctorandos empresariales 775.000 €, agentes de innovación de proximidad 1.250.000 €.",
    ["2026/9192"], chunk="chunk 4 (775.000; 1.250.000)")
add("v2-098", "va", "vague", "underspecified",
    "Hi ha res sobre oposicions o places de personal investigador a la universitat?",
    "Sí, la Universitat de València convoca l'oferta pública de places amb contracte laboral per a línies d'investigació (concurs de mèrits).",
    ["2026/16922"], chunk="oferta pública plazas investigación UV")
add("v2-099", "es", "vague", "underspecified",
    "¿Qué se ha publicado sobre plazas de profesorado ayudante doctor?",
    "La Universidad de Alicante convoca a concurso plazas de profesorado ayudante doctor en régimen laboral (marzo 2026).",
    ["2026/7581"], chunk="profesorado ayudante doctor UA")
add("v2-100", "va", "clean", "none",
    "Qui s'ha cessat com a director general de Relacions Institucionals, Seguiment i Comunicació en el Decret 73/2026, i a petició de qui?",
    "Javier Zaragosí Castelló, a petició pròpia (i es nomena María Isabel Peyró Fernández).",
    ["2026/12103"], chunk="cese Zaragosí a petición propia")


def main() -> int:
    os.makedirs("data/eval_v2", exist_ok=True)
    # sanity / balance asserts
    assert len(ITEMS) == 100, f"expected 100, got {len(ITEMS)}"
    langs = Counter(i["language"] for i in ITEMS)
    assert langs["va"] == 50 and langs["es"] == 50, langs
    ids = [i["id"] for i in ITEMS]
    assert len(set(ids)) == 100, "duplicate ids"

    with open("data/eval_v2/eval_set_v2.jsonl", "w", encoding="utf-8") as fh:
        for it in ITEMS:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    retrieval = [
        {"id": i["id"], "question": i["question"], "gold_sets": i["gold_sets"],
         "doc_ids": i["doc_ids"], "language": i["language"], "category": i["category"],
         "doc_kind": i["doc_kind"], "text_source": i["text_source"]}
        for i in ITEMS if not i["should_abstain"]
    ]
    with open("data/eval_v2/retrieval_input.json", "w", encoding="utf-8") as fh:
        json.dump(retrieval, fh, ensure_ascii=False, indent=1)

    with open("data/eval_v2/answer_input.json", "w", encoding="utf-8") as fh:
        json.dump(ITEMS, fh, ensure_ascii=False, indent=1)

    cat = Counter(i["category"] for i in ITEMS)
    src = Counter(i["text_source"] for i in ITEMS if i["text_source"])
    print(f"items=100 va/es={langs['va']}/{langs['es']} retrieval_items={len(retrieval)}")
    print("by category:", dict(cat))
    print("by text_source (answerable):", dict(src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
