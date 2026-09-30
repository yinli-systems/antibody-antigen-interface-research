# Execution recovery: AF3 inputs and structural-PLM result collection

This extension repairs verified execution defects without modifying any live
training source, old experiment, checkpoint, label or frozen evidence file.
See [RESEARCH.md](RESEARCH.md) for the primary-source investigation,
[PROTOCOL.md](PROTOCOL.md) for AF3 execution gates, and
[ANALYSIS_PROTOCOL.md](ANALYSIS_PROTOCOL.md) for the schema-only analysis repair.

## Actual state

Observed **2026-09-30T08:09:57.023969+00:00**; this is a snapshot, not a live monitor.

| Work | Job / artifact | Verified state |
|---|---|---|
| Original frozen structural-PLM sweep | `1640503` | 12/36 complete; tasks 12–17 running; 18–35 pending; no formal failures |
| Original frozen analyzer | `1640514` | Pending on training dependency; unchanged |
| AF3 task-0 recovery | `1641607` | Pending Priority; no AF3 inference/output success claimed |
| Independent schema-corrected analyzer | `1641627` | Pending: `afterok:1640503,afterany:1640514` |
| Old AF3 task 0 | `1640914_0` | Failed 2:0 before inference; original log retained |
| New regression suite | four test files | 49 individual tests passed |
| Full aggregation integration test | `test_pipeline_fixture.py` | Passed on synthetic 36-task / 217-case / 68-component fixtures; not scientific data |

Only AF3 task 0 has been submitted. Tasks 1–39 require its complete output audit
and terminal scheduler state. No assistant background monitor is configured.

## Verified defects and engineering response

**Input layout.** The old launcher expected `batches/batch-NNN/inputs`, whereas
the frozen builder wrote `inputs/batch-NNN`. The new controller copies exact
input bytes into fresh, task-owned directories and calls the original wrapper
with relative `inputs results` arguments. No original directories are moved.

**Runtime.** The default shared sandbox is missing. The original SIF exists and
is explicitly pinned, along with `af3.bin`. Installed container source reports
AF3 **3.0.1**; the parameter directory is labelled **3.0.0**. These are separately
recorded, not conflated. FUSE is loaded before SIF execution, with shared-cache
population disabled. This is not a claim that all historical structures used
identical software. See the [actual source probe](evidence/cluster/RUNTIME_SOURCE_PROBE.json).

**Output contract.** Exactly 25 unique seed/sample outputs, each with exact
sequence and finite N/CA/C/O, are required. Top-level CIF bytes must match a
maximum-full-precision-score sample. Summary scores are rounded to two decimals
by AF3; they are checked against rounded CSV values, never used to select a
winner. Unexpected timestamp directories, missing samples, modified inputs or
conflicting output receipts fail closed.

**Analysis schema.** Actual completed-task evaluation receipts expose
`partner_retrieval.mean_reciprocal_rank`; the frozen analyzer reads nonexistent
`mrr`. The KeyError was reproduced on an actual completed receipt. The adapter
imports the hashed original analyzer and replaces only its metric-reader
function. AP, bootstrap, primary checkpoint and layer selection are unchanged.
The original analyzer file is not patched because running tasks pin its hash.
See [the reproduced defect](evidence/cluster/REAL_SCHEMA_DEFECT.json).

**Hash provenance.** A previously stated conversational SHA did not match the
legacy completion plan. The already committed main copy and live plan were
compared byte for byte and agree at
`f2199a1f1940ff64fb75d7810b974963ed984fc293c6d3f2f0f150663b1bd615`.
That independently verified anchor is used; the original plan was not rewritten.
See [the comparison](evidence/LEGACY_PLAN_HASH_AUDIT.json).

## Reproduction and controls

```bash
# Python 3.10+; first three suites need only the standard library.
python3 extensions/execution_recovery_v1/test_af3.py
python3 extensions/execution_recovery_v1/test_analysis.py
python3 extensions/execution_recovery_v1/test_assembly.py
# Needs numpy and biopython, using real parsers/writers on synthetic structures.
python3 extensions/execution_recovery_v1/test_structure_io.py
```

`test_pipeline_fixture.py` needs numpy, pandas, pyarrow and scikit-learn, plus
access to the exact frozen original aggregator source named in `--real-plan`.
It creates synthetic inputs in a temporary directory and saves only a labelled
test receipt. It is not a rerun of the real layer sweep.

The actual cluster execution root is
`/ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1`.

- AF3 source: `source_0bd7366`, commit `0bd7366b68b692fbaec2aa0bf7c18b28b47e726a`.
- AF3 execution plan: `af3_attempt2/PLAN.json`, SHA
  `6db2ca822003647080561af7124795d855117a920fd634ce0392d8e995539df3`.
- Scheduled schema adapter: `source_b72e6bf`, commit
  `b72e6bf1736e763135f2104303383b3e4ef0336a`.
- Its authority: `layer_analysis_v1/AUTHORITY.json`, SHA
  `146222211e5a88d789ba61174103db3a88e34e6ef944978295d94a149d1ed505`.

Submission intents are created exclusively before `sbatch`. Ambiguous network
acknowledgements stay locked for reconciliation, never trigger an automatic
retry. All jobs use new output locations and `--no-requeue`. Existing jobs are
not cancelled or resubmitted. The schema repair waits for the unchanged original
analyzer's terminal state, and requires all 36 formal receipts itself.

## Structure assembly: implemented and tested, not yet complete

`assemble.py` admits only all 40 audited tasks, then preserves every base PDB and
writes a separate combined authority. It checks exact missing-chain coverage,
all output hashes and a <=0.0006 Å componentwise CIF-to-PDB rounding bound.
The result must cover all 217 validation complexes. The current training feature
banks are never extended in place. See [ASSEMBLY_PROTOCOL.md](ASSEMBLY_PROTOCOL.md).

The 40-chain structures have **not** finished and the combined authority does
**not** yet exist. Full RePaRank structure evaluation, learned rescue, newly
admitted OOD, co-folder and affinity benchmarks remain unfinished. No new model
accuracy improvement, layer winner, blind result or SOTA is claimed here.

## Evidence

[Final job snapshot](evidence/cluster/FINAL_SNAPSHOT.json) ·
[Raw scheduler states](evidence/cluster/FINAL_SACCT.txt) ·
[AF3 submission](evidence/cluster/af3_attempt2/task0_SUBMITTED.json) ·
[Analysis submission](evidence/cluster/layer_analysis_v1/SUBMITTED.json) ·
[Synthetic integration test](evidence/cluster/SYNTHETIC_PIPELINE_TEST.json) ·
[Export hashes and original paths](evidence/cluster/EXPORT_MANIFEST.json)

The exported evidence contains 35 selected small logs/receipts plus the export
manifest, not weights, SIFs, embeddings or a backup of all cluster data. Original
research branches and previous evidence are retained. Main remains a curated
publication hub; exact source revisions must be used for execution.
