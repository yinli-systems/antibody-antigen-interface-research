"""Published FrameContact overlap numbers — development-validation only.

Sourced from framecontact receipts (read-only). Not an official test / SOTA claim.
"""

from __future__ import annotations

OVERLAP_PUBLIC = {
    "source": "framecontact/receipts/asep-joint-same-antigen-predicted-overlap-20260909/"
    "FRAMECONTACT_ASEP_JOINT_SAME_ANTIGEN_PREDICTED_OVERLAP_AGGREGATE.json",
    "claim_boundary": (
        "Five-seed development-validation predicted epitope-overlap diagnostic on "
        "frozen a0-a5 checkpoints. It did not select a model and is not an official-test "
        "or SOTA result."
    ),
    "true_epitope_jaccard_mean": 0.0565925081616571,
    "a0_predicted_topk_jaccard_mean": 0.4843973841345885,
    "a0_hotspot_excess_mean": 0.4278048759729314,
}


def lookup_predicted_overlap(condition: str = "framecontact_joint_a0_pair_only") -> dict:
    payload = dict(OVERLAP_PUBLIC)
    payload["condition"] = condition
    return payload
