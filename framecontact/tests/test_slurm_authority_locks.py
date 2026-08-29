"""Static guards against mistyped data-authority digests in Slurm launchers."""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARED_SHA256 = "a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335"
HEX_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{60,68}(?![0-9a-f])")


def test_sabdab2_slurm_launchers_pin_the_exact_prepared_authority() -> None:
    paths = (
        "scripts/slurm_build_sabdab2_split_matrix.sbatch",
        "scripts/slurm_build_sabdab2_split_matrix_v2.sbatch",
        "scripts/slurm_build_sabdab2_split_matrix_v3.sbatch",
        "scripts/slurm_verify_sabdab2_esm.sbatch",
        "scripts/slurm_verify_sabdab2_esm_v2.sbatch",
    )
    for relative_path in paths:
        text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert PREPARED_SHA256 in text, relative_path
        suspicious = [token for token in HEX_TOKEN.findall(text) if len(token) != 64]
        assert suspicious == [], (relative_path, suspicious)
