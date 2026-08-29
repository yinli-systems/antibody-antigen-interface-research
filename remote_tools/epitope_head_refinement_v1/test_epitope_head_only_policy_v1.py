#!/usr/bin/env python3
"""Focused CPU test for the isolated epitope-head training policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    module_path = source_root / "scripts" / "train_framecontact.py"
    spec = importlib.util.spec_from_file_location("head_only_train", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load patched training module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.FrameContactModel(
        antibody_input_dim=8,
        antigen_input_dim=8,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        geometry_feature_mode="full_pose",
        geometry_bias_mode="pair_mlp",
        num_rbf=4,
        neighbor_k=4,
        neighbor_radius=16.0,
        sequence_window=4,
        dropout=0.1,
        fusion_mode="interleaved",
        cross_attention_contact_gate_mode="learned",
        adapter_mode="epiformer_style_same_protocol",
        distance_bin_count=0,
        contrastive_dim=8,
    )
    frozen_before = module._state_sha256_excluding(model, ("epitope_head",))
    head_before = {
        name: value.detach().clone()
        for name, value in model.epitope_head.state_dict().items()
    }
    names = module._configure_training_policy(model, "epitope_head_only")
    if names != ("epitope_head.weight", "epitope_head.bias"):
        raise AssertionError(names)
    module._set_training_mode(model, "epitope_head_only")
    if model.training or not model.epitope_head.training:
        raise AssertionError("backbone must remain in eval mode")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    hidden = torch.randn(2, 5, 16)
    target = torch.randn(2, 5)
    loss = torch.nn.functional.mse_loss(
        model.epitope_head(hidden).squeeze(-1), target
    )
    loss.backward()
    optimizer.step()
    frozen_after = module._state_sha256_excluding(model, ("epitope_head",))
    if frozen_before != frozen_after:
        raise AssertionError("frozen state changed")
    if all(
        torch.equal(head_before[name], value)
        for name, value in model.epitope_head.state_dict().items()
    ):
        raise AssertionError("epitope head did not update")
    print(
        {
            "status": "PASS_EPITOPE_HEAD_ONLY_POLICY",
            "trainable_parameter_names": names,
            "frozen_state_unchanged": True,
        }
    )


if __name__ == "__main__":
    main()
