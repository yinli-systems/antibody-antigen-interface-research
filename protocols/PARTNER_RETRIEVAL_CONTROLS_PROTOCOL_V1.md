# SAbDab2 partner-retrieval controls protocol v1

Status: implementation locked; no holdout evaluation has been run.

Parent execution protocol: `PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md`, final SHA-256 `b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078`. Fit, evaluation, and two-boundary finalization re-read this physical file and reject both substituted content and retired v1.6 drafts. The reference is embedded directly in the common bundle, each evaluation JSON, every score context, both boundary ledgers, and the final suite-completion receipt.

This controls protocol document is itself an immutable physical input. Its final SHA-256 is hardcoded only after these instructions are frozen; fit, evaluation, finalization, statistics, and learned-versus-control comparison rehash the physical file and bind its exact path and digest through every provenance layer. A version label without matching document bytes is insufficient.

## Purpose

These controls test whether development-set retrieval performance requires a learned, partner-dependent antibody–antigen representation. They are diagnostics, not replacement models. Every fitted quantity is estimated from the authorized training partition only. The development holdout is read once by a separate command after an independently reviewed split is supplied. The authoritative frozen test is never selected, embedded, scored, or used for model choice.

## Locked controls

1. `raw_shared_esm_mean_cosine`

   Mean-pooled frozen ESM-2 residue embeddings are compared directly by cosine similarity on both sides. This is the defensible raw-embedding control because antibody and antigen vectors use the same encoder, weights, dimension, and coordinate system. Direct AntiBERTy-to-ESM cosine similarity is not used: their 512- and 1,280-dimensional coordinates are unrelated, and truncation or padding would not create a scientifically meaningful shared space.

2. `random_untrained_calm_head`

   AntiBERTy antibody residues and ESM-2 antigen residues pass through independent, frozen, randomly initialized LayerNorm–Linear–GELU–Linear heads followed by residue mean pooling and cosine similarity. The seed is fixed at `20260723`; dimensions are fixed at 256 hidden and 128 output; dropout and parameter updates are both zero. This checks whether architecture and embedding geometry alone create apparent retrieval signal.

3. `antibody_only_unary`

   A low-capacity L2 logistic scorer uses only antibody length, amino-acid composition, entropy, and coarse chemistry. Before retrieval, logits undergo a strict-order-preserving integer-rank transform. The structurally tied antigen direction is resolved by domain-separated SHA-256 ranks of antigen keys with jitter in `(0, 0.25)`. Consequently, antibody-to-antigen retrieval contains no biological antigen information, while antigen-to-antibody retrieval can expose antibody-side marginal propensity.

4. `antigen_only_unary`

   The symmetric marginal control uses only antigen descriptors and the same rank transform. The structurally tied antibody direction is resolved by domain-separated SHA-256 ranks of antibody keys with jitter in `(0, 0.25)`. Consequently, antigen-to-antibody retrieval contains no biological antibody information, while antibody-to-antigen retrieval can expose antigen-side marginal propensity.

The score artifact and downstream statistics API require one deterministic point order, so an analytic expectation over all uniform tie permutations cannot be represented as the emitted score matrix. The preregistered hash jitter is label-blind, never reads the positive graph, is independent of input or lexicographic library order, and has a maximum range strictly below the transformed base-rank gap of `1.0`. Thus it cannot reverse any strict unary-logit ordering. Unary metrics must be reproduced after arbitrary input-axis permutations before evaluation is authorized.

5. `length_composition_logistic`

   A 100-feature L2 logistic model combines antibody and antigen marginal descriptors with composition absolute differences, composition products, log-length difference, and log-length product. It has no pretrained embeddings and no residue-order information. Performance here measures how much can be explained by coarse sequence composition and length.

## Training-only pseudo-negative policy

The three logistic controls require negative observations, but unobserved SAbDab pairs are not experimentally verified non-binders. Five fixed train-only antigen-endpoint permutations are constructed with seed `42017`; known positives are rejected. This approximately preserves observed endpoint composition while preventing label overlap. The bundle records accepted observation counts and rejected collisions. Claims must call these examples *pseudo-negatives*, never confirmed non-binders.

Logistic fitting uses deterministic class-balanced Newton/IRLS, standardized training features, L2 penalty `0.01`, at most 50 iterations, and tolerance `1e-9`. No hyperparameter is selected on the holdout.

## Leakage and provenance firewalls

- The `fit` command calls the shared authority-checked loader with `materialized_splits=("train",)`.
- Before row materialization, the loader requires prepared `instances.parquet` SHA-256 `a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009` and canonical sequence-manifest SHA-256 `548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63`.
- Authority split `e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b` and independent two-boundary verification `3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8` are exact constants, not caller-selectable scientific inputs.
- Before any optimization, the common authority-train embedding inventories must exactly match AntiBERTy `9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e`, shared antibody ESM-2 `03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36`, and antigen ESM-2 `d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e`.
- Fit holds a path-independent, nonblocking operating-system lock for the full process lifetime. A concurrent duplicate fails before dataset or embedding access; the persistent lock file is provenance, not a completed-run marker.
- Split, authority, prepared-Parquet, model-weight, embedding-corpus, source-file, array, and bundle hashes are recorded and revalidated.
- The independent v2 verification manifest predeclares both allowed boundaries. Because both have exactly the same 3,229-member authority-train inventory, one common train-only bundle is fitted. Boundary-specific validation and excluded-validation memberships are deliberately absent from the fitted parameter contract; evaluation is allowed only when its exact boundary SHA and mode appear in the pinned verification manifest.
- Every training embedding artifact is loaded once during fitting to authenticate its model metadata, tensor shape, and sequence length.
- Antibody ESM-2 artifacts are mandatory for the raw same-encoder control. Any missing or mismatched antibody artifact aborts fitting; there is no truncation, padding, alternate encoder, or silent fallback.
- The fitted NPZ contains parameters and hashes but no raw sequence. Evaluation reads its bytes once, hashes that exact buffer, and loads arrays from the same in-memory buffer.
- Evaluation creates an exclusive one-shot ledger before validation rows are materialized. The registry root is fixed at `/cluster/project/model_deployment/results/sabdab2/asymmetric_partner_retrieval/control_evaluation_registry`; callers cannot choose another ledger path.
- The ledger filename is the canonical hash of bundle SHA, boundary SHA, authority SHA, literal `validation`, and suite version. Output paths are not part of this identity, so changing an output directory cannot bypass the one-shot reservation.
- Evaluation refuses an existing output directory or ledger and evaluates all five controls in one holdout read.
- Evaluation physically reauthenticates the prepared table, canonical manifest, final v1.6 protocol, independent boundary verification, authority split, selected boundary, common train embeddings, and fitted bundle before loading the holdout embeddings.
- After both mandatory ledgers complete, `finalize-suite` physically rehashes both ledgers and all ten evaluation/score artifacts, validates each internal score context, and exclusively creates `<registry>/suite-completions/<bundle_sha256>.json`. Statistics and learned-versus-control comparison derive this canonical receipt and fail if either boundary or any control is missing or changed.
- The evaluation template is disarmed unless `RETRIEVAL_CONTROL_EVALUATION_ARMED=YES_REPAIRED_BOUNDARY_REVIEWED`.
- The templates contain no split path or split hash. They require an independently reviewed repaired boundary at submission time.

## Statistical compatibility

Each control emits:

- a float32 full-library similarity matrix;
- a boolean exact-positive mask;
- sorted SHA-256 antibody and antigen keys;
- a standard `sabdab2_asymmetric_retrieval_scores_v1` contract;
- a `sabdab2_asymmetric_partner_retrieval_v1` development evaluation JSON with complete per-anchor ranks.

The artifacts are directly consumable by `sabdab2_asymmetric_retrieval_statistics.py` for exact finite-library chance, fixed-degree label permutations, bootstrap intervals, and candidate-library curves. Control statistics must be interpreted as baselines and must not be passed through the main model's preregistered success gate as if they were candidate models.

## Required execution order

1. Independently approve a repaired, authority-validation-only NW-clean development boundary.
2. Pin the independent v2 verification manifest, immutable v1.6 inputs, and final physical protocol, then run `asymmetric_partner_retrieval_controls_fit.sbatch` once on either approved boundary; the common authority-train membership is the only fitted data inventory.
3. Review the fit summary, convergence fields, source hashes, split hash, and bundle hash.
4. Arm and run `asymmetric_partner_retrieval_controls_evaluate.sbatch` exactly once for each of the two approved boundaries. Both evaluations are mandatory and use the same common train-only bundle; neither is used to choose the other.
5. Authenticate both central ledgers and run `asymmetric_partner_retrieval_controls_finalize_suite.sbatch` once. Review the canonical receipt and all ten physical artifact hashes.
6. Run `asymmetric_partner_retrieval_controls_statistics.sbatch` once per control per boundary. Statistics derives and verifies the canonical full-suite receipt; callers cannot select a substitute receipt path.
7. Compare the selected learned retriever with all controls using paired per-anchor differences on the identical ordered score matrix and positive graph. Comparison likewise requires the canonical full-suite receipt.

## Interpretation limits

- Strong unary performance indicates marginal partner propensity, not pair specificity.
- The raw ESM control is same-encoder geometry, not an antibody-specialized representation.
- Random-head performance should be judged across prespecified seeds only if a multi-seed extension is registered before any holdout read; this v1 protocol fixes one seed.
- Length/composition performance depends on the pseudo-negative construction and cannot establish physical non-binding.
- None of these controls addresses unavoidable protein-language-model pretraining exposure.
