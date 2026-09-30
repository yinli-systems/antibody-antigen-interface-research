# AF3 top-model semantic-audit correction — 2026-09-30

## Preserved failure

Slurm task `1641607_0` completed the full AlphaFold 3 inference grid for task 0:
five registered seeds × five samples. Its post-inference audit failed with:

```text
ValueError: top-level model is not a maximum-score sample byte for byte
```

The original source snapshot, plan, scheduler logs, `FAILURE.json`, input and all
prediction bytes remain unchanged under `af3_attempt2/runs/task_000/`.

## Read-only finding

`ranking_scores.csv` contains exactly 25 unique registered `(seed, sample)` rows.
Its unique maximum is seed 42, sample 2, ranking score
`0.6402947117306571`. All 25 structures have one chain A, the exact 78-residue
input sequence, 597 finite atoms, complete N/CA/C/O backbones, finite confidence
payloads and expected 78×78 PAE/contact matrices.

The top-level confidence and summary-confidence JSON files are byte-identical to
seed 42/sample 2. The top-level mmCIF and that sample have identical chain,
sequence, every parsed atom and coordinate, and every mmCIF category/value except:

```text
_ma_model_list.model_group_name
... @ 2026-09-30 16:40:10)
... @ 2026-09-30 16:40:11)
```

AlphaFold 3's official output documentation defines the root model as the
highest-ranked prediction selected by `ranking_scores.csv`; it does not promise
byte-identical reserialization of the mmCIF.

## Corrected acceptance rule

A non-byte-identical root model is accepted only when all of the following hold:

1. the 5×5 seed/sample grid is exact and all ranking scores are finite;
2. the candidate is a maximum-score registered seed/sample;
3. root and candidate have exact chain A sequence and complete finite atoms;
4. full all-atom and N/CA/C/O coordinate fingerprints are identical;
5. the complete mmCIF dictionaries are identical after normalising only the
   timestamp inside `_ma_model_list.model_group_name`;
6. every other mmCIF field remains exact, including atom coordinates;
7. root confidence and summary-confidence payloads equal the selected sample;
8. all 25 sample confidence payloads pass field, shape, range and chain checks;
9. processed input remains single-chain, query-only MSA, no-template, registered
   seed input.

Changing a coordinate, model name, sequence, seed/sample row, confidence payload
or any non-allowed mmCIF field still fails closed. Raw byte hashes and the raw
difference-key list remain in the receipt.

## Execution boundary

The successful task-0 prediction is admitted by copying its preserved bytes into
a new attempt and rerunning audit/assembly only. No GPU inference is repeated.
Remaining tasks may be submitted only through the existing exclusive controller
after the new task-0 `COMPLETE.json` passes and scheduler state is reconciled.
