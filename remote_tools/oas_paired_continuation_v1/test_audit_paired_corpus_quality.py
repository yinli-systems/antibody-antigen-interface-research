import sqlite3

from audit_paired_corpus_quality import audit_rows


def test_quality_audit_accepts_exact_cdr_mapping_and_unique_rows() -> None:
    heavy = "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS"
    light = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"
    row = [
        heavy,
        light,
        "GYDFTHY",
        "INTYTGEPT",
        "AKYPYYYGTSHWYFDV",
        "SASQDISNY",
        "FTS",
        "QQYSTVPWT",
        "study",
        "run",
        "human",
        "a" * 64,
        "1" * 64,
    ]
    result = audit_rows([row], connection=sqlite3.connect(":memory:"), expected_rows=1)
    assert result["status"] == "PASS_PAIRED_CORPUS_QUALITY_AUDIT"
    assert result["counts"]["valid_rows"] == 1
    assert result["lengths"]["paired_tokens"]["maximum"] == 233
