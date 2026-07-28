from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "slurm" / "oas_authority_discovery.sbatch"


def test_oas_authority_scheduler_is_fail_closed() -> None:
    source = SCHEDULER.read_text(encoding="ascii")

    assert "set -euo pipefail" in source
    assert "umask 077" in source
    assert "OAS_AUTHORITY_SOURCE_SHA256" in source
    assert "OAS_CONTROLLER_SOURCE_SHA256" in source
    assert "OAS_SCHEDULER_SOURCE_SHA256" in source
    assert "The authority output root is create-only" in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "--fetch" in source
    assert "chmod -R a-w" in source
