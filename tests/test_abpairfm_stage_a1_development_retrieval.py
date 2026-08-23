from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scripts import run_abpairfm_stage_a1_frozen_development_retrieval as module


ROOT = Path(__file__).resolve().parents[1]


def test_unmasked_batch_preserves_chain_boundaries() -> None:
    batch = [(('ordered', 'a', 'b'), 'ACD', 'EFGH')]
    tokens, chain_ids, residues = module.build_unmasked_batch(
        batch, device=torch.device('cpu')
    )
    assert residues == 7
    assert tokens.shape == chain_ids.shape == (1, 11)
    assert chain_ids[0, :5].tolist() == [0] * 5
    assert chain_ids[0, 5:].tolist() == [1] * 6
    assert 32 not in tokens[0].tolist()


def test_protocol_freezes_development_only_retrieval() -> None:
    path = ROOT / 'protocols' / 'abpairfm_stage_a1_development_retrieval_v1.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    assert protocol['status'] == 'FROZEN_BEFORE_GPU_EVALUATION'
    assert protocol['authorization']['development_pair_retrieval'] is True
    assert protocol['authorization']['development_contact'] is False
    assert protocol['authorization']['test_access'] is False
    assert protocol['test_access'] is False
    assert protocol['compute']['partition'] == 'gpu_4090'
    assert hashlib.sha256(path.read_bytes()).hexdigest()


def test_sbatch_uses_only_gpu_4090_and_fixed_python() -> None:
    text = (
        ROOT / 'scripts' / 'submit_abpairfm_stage_a1_frozen_development_retrieval.sbatch'
    ).read_text(encoding='utf-8')
    assert '#SBATCH -p gpu_4090' in text
    assert '#SBATCH --gres=gpu:8' in text
    assert 'hp_4090' not in text
    assert '#SBATCH --mem' not in text
    assert '/ssd/scxi253/model_deployment/envs/contact/bin/python3.10' in text
