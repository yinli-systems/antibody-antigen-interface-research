# Aligned FrameContact historical test handoff

This is the **549-instance antigen-fold test**, not the original 455-instance
split. Its exact instance IDs are disjoint from the selected checkpoints' 2,565
training and 549 validation instances. This does **not** establish sequence
double-cold generalization or turn a previously exposed benchmark into a blind test.

The old 455 set intersects those training/validation/test roles by **272/73/110**.
It is not a truncated version of this test. The old embedding-only 455 package
contains no geometry; the separately named original455 full cache contains
experimental intra-chain geometry. Neither fixes the split mismatch.

## Contents and fixed interpretation

- `inputs/embeddings`: byte-identical ESM-2 650M tensors for this cohort, with a
  portable subset manifest and the original complete manifest in `authority`.
- `inputs/structure`: existing independent AF3-predicted N/CA/C, chain IDs and
  frame masks. Antibody and antigen were predicted separately; some predictions
  used homolog MSAs. Do not describe this as query-only structure prediction.
- `labels/contact`: existing experimental contact labels, used for measurement
  only. No experimental backbone or distance-label cache is a model input.
- `checkpoints`: baseline and TopK2 expert for each seed 17, 29 and 43, selected
  on development only. `CONFIG.json` stores frozen validation temperatures and
  thresholds. There is no selection or calibration on this test.
- `source`: exact source snapshots, separately imported in each process because
  the baseline and expert implement different residue readouts.
- `authority/SPLIT_MISMATCH_AUDIT.json`: legacy455 instance-by-instance role audit.

The fixed route is **Contact + Paratope from baseline; Epitope from the
seed-matched TopK2 expert**. Seed43 epoch6 alone is the contact case-study model,
not the full three-seed task-routed system. No probability ensemble is fitted.

## Verification and inference

Use the externally supplied SHA-256 of `BUNDLE.json`; do not trust a substituted
manifest simply because it is in the same directory. Every bundled artifact is
hashed before imports or inference. Download/copy the entire directory; provenance
documents retain original paths, but runtime inputs and source imports are relative.
Do not edit a packaged source or replace the cache/checkpoint under an existing seal.

Python 3.10 with PyTorch, NumPy, SciPy, scikit-learn and pandas is required; other
optional source imports may have existing environment dependencies. No package
installation is performed by these scripts. The verified ParaCloud interpreter is
`/ssd/scxi253/model_deployment/envs/contact/bin/python3.10`.

```bash
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python run_bundle.py --bundle /path/to/bundle --bundle-sha256 RECEIVED_SHA256 \
  --component 0 --mode full --output /path/to/new-results/0
```

Run components 0–5 in **separate processes**. Order: baseline17/29/43, expert17/29/43.
On ParaCloud run only on allocated `gpu_4090` compute nodes; no login-node inference.
Then run:

```bash
python finalize_bundle.py --bundle /path/to/bundle --bundle-sha256 RECEIVED_SHA256 \
  --runs /path/to/new-results --mode full --output /path/to/new-summary
```

`READY/RESULT.json` is the full-loader + six-model forward verification. A packaged
directory without this PASS receipt is **not yet verified model-ready**. This smoke
result is not a completed performance evaluation; the full-run summary is separate.

## Metrics and reproducibility

The output includes AUROC, **pooled AP** (`auprc`, sklearn average precision),
**MPC-AP** (mean per eligible complex; zero-positive cases excluded with counts),
micro Precision/Recall/F1/MCC and TP/FP/TN/FN. MPC-AP follows the original
uncalibrated sigmoid per-complex computation; pooled metrics use the frozen
validation temperature. Saturated sigmoid scores can create ties.

Primary threshold metrics use **frozen validation-selected thresholds**. Fixed
raw-sigmoid 0.5 and fixed calibrated-probability 0.5 are separately named diagnostics;
do not silently merge them. Ranking and threshold metrics are not interchangeable.
Per-instance logits, labels, masks and hashes are retained for exact reconciliation.
Labels are removed before `model.forward` and model state must remain unchanged.

Do not claim to have reproduced a colleague's 455/Chai-1 result from this bundle:
their evaluated ID manifest, inference script, checkpoint SHA, geometry manifest,
threshold/temperature settings and raw output are needed to reconcile that run.
