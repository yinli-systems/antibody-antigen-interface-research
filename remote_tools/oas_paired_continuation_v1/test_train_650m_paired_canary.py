from pathlib import Path

import numpy as np

from materialize_paired_corpus import materialize_rows
from train_650m_paired_canary import PairedBinaryShard, select_indices


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


def test_binary_shard_loader_and_selection(tmp_path: Path) -> None:
    root = tmp_path / "shard"
    result = materialize_rows(
        [
            _row("1" * 64, "0" * 64),
            _row("2" * 64, "f" * 64),
            _row("3" * 64, "e" * 64),
        ],
        exclusions=set(),
        output_root=root,
        expected_source_rows=3,
    )
    manifest = {
        "artifacts": result["artifacts"],
        "counts": result["counts"],
        "format_version": "oas_paired_development_clean_binary_v1",
        "status": "PASS_PAIRED_DEVELOPMENT_CLEAN_MATERIALIZATION",
        "test_access": False,
    }
    (root / "MANIFEST.json").write_text(__import__("json").dumps(manifest), encoding="ascii")
    shard = PairedBinaryShard(root, verify_hashes=True)
    validation = select_indices(shard, 1, 1, 17)
    training = select_indices(shard, 0, 2, 17)
    assert len(validation) == 1
    assert len(training) == 2
    tokens, types, regions, keys = shard.batch(np.asarray([validation[0], training[0]]))
    assert tokens.shape == types.shape == regions.shape
    assert tokens.shape[1] % 8 == 0
    assert keys.shape == (2,)
