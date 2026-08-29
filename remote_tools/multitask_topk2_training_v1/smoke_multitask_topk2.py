#!/usr/bin/env python3
import torch

from interaction_first.framecontact import FrameContactModel, build_backbone_frames


coordinates = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
batch = {
    "antibody_features": torch.randn(1, 3, 8),
    "antigen_features": torch.randn(1, 4, 8),
    "antibody_mask": torch.ones(1, 3, dtype=torch.bool),
    "antigen_mask": torch.ones(1, 4, dtype=torch.bool),
    "antibody_backbone": coordinates.repeat(1, 3, 1, 1),
    "antigen_backbone": coordinates.repeat(1, 4, 1, 1),
    "antibody_chain_ids": torch.zeros(1, 3, dtype=torch.long),
    "antigen_chain_ids": torch.zeros(1, 4, dtype=torch.long),
}
for partner in ("antibody", "antigen"):
    valid = build_backbone_frames(
        batch[f"{partner}_backbone"], batch[f"{partner}_mask"]
    ).valid_mask
    batch[f"{partner}_frame_valid_mask"] = valid
    batch[f"{partner}_missing_frame_mask"] = batch[f"{partner}_mask"] & ~valid

outputs = {}
for paratope_pooling, contact_readout in (
    ("logmeanexp", "pair"),
    ("topk_mean", "pair"),
    ("logmeanexp", "topk_context_equal_logit"),
):
    torch.manual_seed(17)
    model = FrameContactModel(
        antibody_input_dim=8,
        antigen_input_dim=8,
        d_model=16,
        num_heads=4,
        num_self_layers=1,
        adapter_mode="epiformer_style_same_protocol",
        fusion_mode="interleaved",
        cross_attention_contact_gate_mode="content_only",
        paratope_pooling=paratope_pooling,
        contact_readout=contact_readout,
        distance_bin_count=5,
        dropout=0.0,
    )
    output = model(batch)
    assert output["contact_logits"].shape == (1, 3, 4)
    assert output["paratope_logits"].shape == (1, 3)
    assert output["epitope_logits"].shape == (1, 4)
    assert torch.isfinite(output["contact_logits"]).all()
    outputs[(paratope_pooling, contact_readout)] = output

baseline = outputs[("logmeanexp", "pair")]
paratope = outputs[("topk_mean", "pair")]
contact = outputs[("logmeanexp", "topk_context_equal_logit")]
assert torch.equal(baseline["contact_logits"], paratope["contact_logits"])
assert not torch.equal(baseline["paratope_logits"], paratope["paratope_logits"])
assert not torch.equal(baseline["contact_logits"], contact["contact_logits"])
assert torch.equal(baseline["paratope_logits"], contact["paratope_logits"])
print("MULTITASK_TOPK2_SMOKE_PASS")
