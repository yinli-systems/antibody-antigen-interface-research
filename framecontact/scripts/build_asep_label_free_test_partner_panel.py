#!/usr/bin/env python3
"""Freeze full or SAbDab2-decontaminated AsEP test partner panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.asep_test_partner_panel import (  # noqa: E402
    build_asep_label_free_test_partner_panel,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label-free-test-root", type=Path, required=True)
parser.add_argument("--label-free-test-receipt-sha256", required=True)
parser.add_argument("--decontamination-root", type=Path)
parser.add_argument("--decontamination-receipt-sha256")
parser.add_argument("--panel-size-per-direction", type=int, default=31)
parser.add_argument("--protocol-seed", type=int, default=20_260_730)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
print(json.dumps(build_asep_label_free_test_partner_panel(
    label_free_test_root=arguments.label_free_test_root,
    label_free_test_receipt_sha256=arguments.label_free_test_receipt_sha256,
    output_path=arguments.output,
    panel_size_per_direction=arguments.panel_size_per_direction,
    protocol_seed=arguments.protocol_seed,
    decontamination_root=arguments.decontamination_root,
    decontamination_receipt_sha256=arguments.decontamination_receipt_sha256,
), sort_keys=True))
