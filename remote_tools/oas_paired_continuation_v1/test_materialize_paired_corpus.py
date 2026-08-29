import sqlite3

from materialize_paired_corpus import materialize_rows


def _row(row_hash: str, donor_hash: str) -> list[str]:
    return [
        "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS",
        "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK",
        "GYDFTHY",
        "INTYTGEPT",
        "AKYPYYYGTSHWYFDV",
        "SASQDISNY",
        "FTS",
        "QQYSTVPWT",
        "study",
        "run",
        "human",
        donor_hash,
        row_hash,
    ]


def test_materialization_excludes_rows_and_freezes_donor_split(tmp_path) -> None:
    excluded = bytes.fromhex("1" * 64)
    rows = [
        _row("1" * 64, "2" * 64),
        _row("3" * 64, "0" * 64),
        _row("4" * 64, "f" * 64),
    ]
    result = materialize_rows(
        rows,
        exclusions={excluded},
        output_root=tmp_path / "shard",
        expected_source_rows=3,
    )
    assert result["counts"] == {
        "excluded_rows": 1,
        "kept_rows": 2,
        "source_rows": 3,
        "train_rows": 1,
        "validation_rows": 1,
    }
    assert result["artifacts"]["split"]["bytes"] == 2
