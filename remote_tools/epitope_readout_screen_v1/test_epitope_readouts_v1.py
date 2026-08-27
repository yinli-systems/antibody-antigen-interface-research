#!/usr/bin/env python3
"""CPU contract test for frozen epitope readout formulas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    module_path = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location("epitope_readouts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    contact_logits = np.asarray(
        [[0.0, 2.0, -2.0], [1.0, -1.0, 3.0]], dtype=np.float64
    )
    interaction_logits = contact_logits - 0.25
    mask = np.asarray([[True, True, True], [True, False, True]])
    epitope_contact_logits = np.asarray([0.2, -0.3, 0.7])
    head_logits = np.asarray([-0.4, 0.5, -0.2])
    antigen_unary_logits = np.asarray([0.1, -0.2, 0.3])
    readouts = module.candidate_readouts(
        total_contact_logits=contact_logits,
        interaction_contact_logits=interaction_logits,
        pair_mask=mask,
        epitope_total_logits=head_logits + epitope_contact_logits,
        epitope_contact_logits=epitope_contact_logits,
        antigen_unary_logits=antigen_unary_logits,
        direct_head_logits=head_logits,
    )
    if tuple(readouts) != module.READOUT_ORDER:
        raise AssertionError("readout order differs")
    expected_mean = np.asarray(
        [
            np.mean(module.stable_sigmoid(contact_logits[:, 0])),
            module.stable_sigmoid(contact_logits[0, 1]),
            np.mean(module.stable_sigmoid(contact_logits[:, 2])),
        ]
    )
    np.testing.assert_allclose(
        readouts["paper_mean_total_contact"], expected_mean, rtol=1e-12, atol=1e-12
    )
    expected_hybrid = module.stable_sigmoid(
        head_logits + module.stable_logit(expected_mean)
    )
    np.testing.assert_allclose(
        readouts["head_plus_paper_mean_a100"],
        expected_hybrid,
        rtol=1e-12,
        atol=1e-12,
    )
    if not all(
        values.shape == (3,) and np.isfinite(values).all()
        for values in readouts.values()
    ):
        raise AssertionError("invalid readout values")
    print(
        {
            "status": "PASS_EPITOPE_READOUT_CONTRACT",
            "readout_count": len(readouts),
        }
    )


if __name__ == "__main__":
    main()
