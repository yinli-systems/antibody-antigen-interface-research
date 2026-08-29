"""Test the fail-closed AsEP development IMGT query launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/oas_asep_imgt_query_export.sbatch"


def test_launcher_pins_sources_cache_runtime_and_development_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert 'test ! -e "${OAS_QUERY_OUTPUT_ROOT}"' in source
    assert 'DEPLOYED_COMMIT' in source
    assert 'OAS_QUERY_SCRIPT_SHA256' in source
    assert 'OAS_QUERY_MODULE_SHA256' in source
    assert 'OAS_ASEP_CACHE_MANIFEST_SHA256' in source
    assert 'OAS_ANARCII_FREEZE_SHA256' in source
    assert 'ASEP_DEVELOPMENT_CACHE_MANIFEST.json' in source
    assert '--expected-query-count 3106' in source
    assert '--batch-size 256' in source
    assert '--ncpu 8' in source
    assert "frozen" not in source.lower()
    assert "official_test" not in source.lower()
    assert "external_test" not in source.lower()
