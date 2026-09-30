# Historical identity parity audit

Recorded UTC: 2026-09-29T22:42:39.985668+00:00

Before opening the extension's formal metric results, independently check its
identity arm against each existing checkpoint's original validation receipt.
The nine reference checkpoint/split hashes have already been matched. This is
an engineering consistency audit, not model selection or a new test reading.

Require all nine tasks and all 217 identical development-validation IDs. Compare
per-complex means of pair AP, epitope AP and paratope AP; omit undefined AP only
with explicit counts. Maximum tolerated absolute difference is 1e-6 for each
metric in each task. Any larger difference is HOLD_FOR_DIAGNOSIS, not a reason
to rewrite old results, relax the threshold or promote a new measurement.
The audit records both means, eligible counts, and the largest deviation.
