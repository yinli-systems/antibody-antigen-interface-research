# Partner-negative ablation — raw-Brier v2 running

Status at `2026-08-21T20:46:10+08:00`: the paired v2 study is running on
twenty `gpu_4090` cards. Job `1457687` tasks 0–9 cover both AsEP arms;
job `1457709` tasks 10–19 cover both SAbDab2 arms. Each dataset and arm uses
seeds 17, 42, 73, 101, and 137. No final comparison is reported yet.

All ten AsEP tasks have written epoch 3 with the required raw-Brier field and
zero stderr. The authenticated launch snapshot is
[`V2_LAUNCH_RECEIPT_1457687_1457709.json`](V2_LAUNCH_RECEIPT_1457687_1457709.json)
(SHA-256 `1992dc6ca0242b560789e969903d4220cb58c9a82f6d2e68ea3993ffcba0aeff`).
SAbDab2 is still inside its longer first epoch and is not counted as
instrumentation-verified until it writes the same field.

`audit_negative_sample_ablation_progress.py` now checks every task directly
from its JSONL log, rejects a supposedly raw metric unless temperature is 1,
threshold is 0.5, calibration source is `none`, and Brier is finite in [0, 1],
and reports stderr/final-result presence. Its live status is
`RUNNING_AWAITING_FIRST_EPOCH`: AsEP is 10/10 instrumented, SAbDab2 is 0/10,
and test access is false. The script deliberately does not compare arm means
from unmatched latest epochs.

## Why v1 was stopped

The frozen analyzer requires the calibration-independent field
`validation.contact.fixed_raw_logit_threshold_zero.brier`. A fail-closed
readiness run showed that all 13 readable v1 results—ten historical
with-negative baselines and three completed no-negative runs—contained zero
instances of that field. Their top-level `contact.brier` was computed after a
temperature fit on the same validation epoch and cannot be relabeled as raw
Brier. The remaining v1 array tasks were therefore cancelled after preserving
[`V1_RAW_BRIER_FAILURE_RECEIPT_1457186.json`](V1_RAW_BRIER_FAILURE_RECEIPT_1457186.json)
(SHA-256 `9499b2b43cc6c7b95dac96fc864b54eaeae28d3189d571edae59a9b6d4fbd126`).

## v2 comparison

- with-negative arm: one deterministic authenticated wrong partner with
  specificity loss weight 0.25;
- no-negative arm: identical model, split, seed, optimizer, learning rate,
  epoch limit, early stopping, partner panels, and execution path, with
  specificity loss weight exactly 0.0;
- shared train curve: contact BCE and
  `contact + 0.25 * epitope + 0.25 * paratope`;
- validation curve: Brier from `sigmoid(raw_contact_logits)`, fixed
  temperature 1.0 and probability threshold 0.5;
- final development metrics: pair contact AUPRC/AUROC/ranking/calibration,
  epitope/paratope metrics, and paired five-seed bootstrap intervals.

The v2 trainer change is measurement-only: it computes raw probabilities from
contact logits already held during validation. It does not alter gradients,
sampling, calibration, early stopping, or checkpoint selection. Each copied
source tree is frozen by a full `scripts/`, `src/`, and `protocols/` inventory.

| Dataset | Arm | Protocol SHA-256 | Source inventory SHA-256 |
|---|---|---|---|
| AsEP | with negative | `60b13fabd162978556adcf884a8193ebcbf21eb25d8b2b2b317f86f9a6e2cf47` | `7a5d83342dd61856947861d30b82732a9f3be4dfa46cd586e3d3ec31018fd815` |
| AsEP | no negative | `8fbf71dd547d4b4c316055130c7a98904e958a2c3aaa9ba8fd167972c9e9ef88` | `7a5d83342dd61856947861d30b82732a9f3be4dfa46cd586e3d3ec31018fd815` |
| SAbDab2 | with negative | `d20994b6924bafff47ad9741d5a6ff0f8988c3c3b5ec9fde91af3e2e24839d59` | `1df974bb5c3cbb2d2234d9024489ec22b014f5233c0d1e8278c8da95ff3bf3b2` |
| SAbDab2 | no negative | `42feba723c1a134cec5e420cef4d1eda0e7a267f5dd157f36cc4f7049cc8cd5a` | `1df974bb5c3cbb2d2234d9024489ec22b014f5233c0d1e8278c8da95ff3bf3b2` |

The first SAbDab2 v2 launch bound the copied source to the old absolute-root
manifest and failed before training. The ten AsEP tasks were left running; a
new manifest was generated for the copied immutable closure and only SAbDab2
was resubmitted. Evidence is retained in
[`V2_SABDAB_SOURCE_MANIFEST_FAILURE_RECEIPT_1457687.json`](V2_SABDAB_SOURCE_MANIFEST_FAILURE_RECEIPT_1457687.json).

Test and external-test roles remain sealed. They may be opened only after all
twenty development runs finish, the analyzer emits a verified receipt, and the
development comparison is reported and accepted for progression.
