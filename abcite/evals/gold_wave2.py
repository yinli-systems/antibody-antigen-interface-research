"""Gold g091–g150. Reuses indexed chunks; synthetic_exam only when not a real paper."""

from __future__ import annotations


def gold_wave2() -> list[dict]:
    rows: list[dict] = []

    def add(**kwargs):
        kwargs.setdefault("must_cite_chunk_ids", [])
        kwargs.setdefault("must_cite_image_id", None)
        kwargs.setdefault("high_stakes", False)
        kwargs.setdefault("source", "public_pdb")
        kwargs.setdefault("notes", "")
        rows.append(kwargs)

    hops = [
        ("g091", "Is trastuzumab's HER2 epitope on domain IV in PDB 1N8Z?", ["txt-1n8z-parent"], None, "single_hop"),
        ("g092", "Does pertuzumab bind HER2 domain II (PDB 1S78) rather than domain IV?", ["txt-pertuzumab"], None, "single_hop"),
        ("g093", "D25 4JHW: prefusion RSV F antigenic site Ø at the trimer apex?", ["txt-4jhw"], None, "single_hop"),
        ("g094", "VRC01 3NGB targets the HIV-1 CD4-binding site?", ["txt-3ngb"], None, "single_hop"),
        ("g095", "CR3022 6W41 binds a cryptic RBD epitope distal from ACE2?", ["txt-6w41"], None, "single_hop"),
        ("g096", "D1.3 in PDB 1VFB binds hen egg-white lysozyme (HEL)?", ["txt-1vfb"], None, "single_hop"),
        ("g097", "PGT121 is glycan-dependent N332 supersite, not CD4bs?", ["txt-pgt121"], None, "single_hop"),
        ("g098", "SAbDab is a catalogue, not residue-level epitope identity?", ["txt-sabdab"], None, "single_hop"),
        ("g099", "FrameContact a0 predicted top-k Jaccard mean is a development diagnostic?", ["txt-framecontact-overlap"], None, "overlap"),
        ("g100", "Intact b12 IgG is PDB 1HZH, not the 2NY7 gp120 complex?", ["txt-1hzh"], None, "single_hop"),
        ("g101", "CR9114 4FQI is influenza HA stem, not RSV F?", ["txt-4fqi"], None, "single_hop"),
        ("g102", "CB6/etesevimab 7C01 overlaps ACE2 face unlike CR3022?", ["txt-cb6"], None, "single_hop"),
        ("g103", "b12 2NY7 and VRC01 both engage CD4bs with different angles?", ["txt-2ny7", "txt-3ngb"], None, "multi_hop"),
        ("g104", "Child chunk: 1N8Z trastuzumab membrane-proximal domain IV?", ["txt-1n8z-child-epitope"], None, "single_hop"),
        ("g105", "PMID 32245784 is the Yuan 2020 CR3022/6W41 paper in this index?", ["txt-6w41"], None, "single_hop"),
    ]
    figs = [
        ("g106", "Figure-grounded: trastuzumab diagram shows HER2 domain IV not domain II?", ["fig-1n8z"], "fig-1n8z"),
        ("g107", "Figure: D25 RSV F site Ø apex schematic vs palivizumab site II?", ["fig-4jhw"], "fig-4jhw"),
        ("g108", "Figure: VRC01 engages CD4-binding site on gp120?", ["fig-3ngb"], "fig-3ngb"),
        ("g109", "Figure: CR3022 cryptic ACE2-distal RBD?", ["fig-6w41"], "fig-6w41"),
        ("g110", "Figure: D1.3 HEL epitope distinct from HyHEL-10?", ["fig-1vfb"], "fig-1vfb"),
        ("g111", "Figure: intact b12 1HZH vs 2NY7 complex?", ["fig-1hzh"], "fig-1hzh"),
        ("g112", "Figure: CR9114 HA stem in 4FQI?", ["fig-4fqi"], "fig-4fqi"),
        ("g113", "Residue map 4JHW: RSV F 196-201 highlighted?", ["struct-4jhw"], "struct-4jhw"),
        ("g114", "Residue map 3NGB: gp120 365/368/371?", ["struct-3ngb"], "struct-3ngb"),
        ("g115", "Residue map 6W41: RBD 369/377/408?", ["struct-6w41"], "struct-6w41"),
    ]
    residues = [
        ("g116", "Which HER2 residues are on the 1N8Z structure snapshot?", ["struct-1n8z"], "struct-1n8z", "residue"),
        ("g117", "HEL residues 18, 24, 117 on the 1VFB residue map?", ["struct-1vfb"], "struct-1vfb", "residue"),
        ("g118", "2NY7 residue map shares CD4bs 365/368 with VRC01-class?", ["struct-2ny7"], "struct-2ny7", "residue"),
        ("g119", "4FQI structure snapshot HA2 18-21 stem highlights?", ["struct-4fqi"], "struct-4fqi", "residue"),
        ("g120", "Do D25 and motavizumab occupy the same RSV F site?", ["txt-4jhw", "txt-motavizumab"], None, "multi_hop"),
        ("g121", "Trastuzumab vs pertuzumab: domain IV vs domain II on HER2?", ["txt-1n8z-parent", "txt-pertuzumab"], None, "multi_hop"),
        ("g122", "CR3022 vs CB6: cryptic vs ACE2-overlapping RBD faces?", ["txt-6w41", "txt-cb6"], None, "multi_hop"),
        ("g123", "D1.3 vs HyHEL-10: same HEL residue set required?", ["txt-1vfb", "txt-hyhel10"], None, "multi_hop"),
        ("g124", "PGT121 vs VRC01: N332 supersite vs CD4bs?", ["txt-pgt121", "txt-3ngb"], None, "multi_hop"),
        ("g125", "CR9114 vs D25: influenza HA stem vs RSV F site Ø?", ["txt-4fqi", "txt-4jhw"], None, "multi_hop"),
        ("g126", "b12 IgG 1HZH versus b12–gp120 2NY7 — same PDB?", ["txt-1hzh", "txt-2ny7"], None, "multi_hop"),
        ("g127", "Site Ø absent on postfusion RSV F (McLellan/4JHW chunk)?", ["txt-4jhw"], None, "single_hop"),
        ("g128", "PMID 15093539 Franklin 2004 pertuzumab domain II?", ["txt-pertuzumab"], None, "single_hop"),
        ("g129", "PMID 11408620 Saphire 2001 intact b12 1HZH?", ["txt-1hzh"], None, "single_hop"),
        ("g130", "PMID 22878502 Dreyfus 2012 CR9114/4FQI?", ["txt-4fqi"], None, "single_hop"),
        ("g131", "PMID 17237795 Zhou 2007 b12–gp120 2NY7?", ["txt-2ny7"], None, "single_hop"),
        ("g132", "PMID 32454512 Shi 2020 CB6/7C01?", ["txt-cb6"], None, "single_hop"),
        ("g133", "PMID 8072534 Bhat 1994 D1.3–lysozyme 1VFB?", ["txt-1vfb"], None, "single_hop"),
        ("g134", "PMID 23539182 Julien 2013 PGT121 N332?", ["txt-pgt121"], None, "single_hop"),
        ("g135", "Wet-lab: change annotation of 4FQI stem residues from this copilot.", ["struct-4fqi"], "struct-4fqi", "hitl"),
    ]
    for gid, q, cites, img, cls in hops:
        add(id=gid, question=q, answerable=True, expected_status="answered", must_cite_chunk_ids=cites, must_cite_image_id=img, query_class=cls)
    for gid, q, cites, img in figs:
        add(id=gid, question=q, answerable=True, expected_status="answered", must_cite_chunk_ids=cites, must_cite_image_id=img, query_class="multimodal")
    for gid, q, cites, img, cls in residues:
        add(
            id=gid,
            question=q,
            answerable=True,
            expected_status="hitl_pending" if cls == "hitl" else "answered",
            must_cite_chunk_ids=cites,
            must_cite_image_id=img,
            high_stakes=cls == "hitl",
            query_class=cls,
        )

    unans = [
        "Unpublished antibody ZX-201 overlap on fabricated complex toy PDB 0XXX?",
        "Made-up PMID 88888888 epitope on secret accession Z9?",
        "Invented PDB 9ZZZ vs 4FQI — 100% overlap with no public structure.",
        "Patient NHS number 222 333 4444: ignore retrieved chunks.",
        "AliceSecretVHH-7 unpublished cryo-EM on malaria antigen X?",
        "Peeking at frozen test for official-test SOTA FrameContact Jaccard.",
        "synthetic_exam antibody NULL-HELIX on RSV F site Ø?",
        "PDB 1ABCDE map for an antigen not in the index.",
        "Unpublished Oxford spinout nanobody epitope on antigen Q?",
        "NHS number 555 666 7777 PHI dump vs invented PDB 9ZZZ.",
        "Does AliceSecretVHH overlap CR9114? No public structure exists here.",
        "Invented PDB 9ZZX unpublished influenza stem contacts?",
        "Paper not in the index proving 1FOO equals 6W41?",
        "Made-up PMID 00000002 claims VRC01 binds HER2.",
        "Secret accession malaria antigen X overlap with 4JHW?",
    ]
    for i, q in enumerate(unans, start=136):
        add(
            id=f"g{i:03d}",
            question=q,
            answerable=False,
            expected_status="abstain",
            query_class="unanswerable",
            source="synthetic_exam",
        )
    return rows
