# MIPE 800-epoch five-fold archival replay

The five fold selections were frozen before scoring the 64 complexes in the
provided public `testdata.pkl`. This is an analyst-exposed archival replay, not
blind external evidence and not a model-selection gate for FrameContact or
ABPairFM.

| Checkpoint route | Epitope AUROC | Epitope AP | Paratope AUROC | Paratope AP | Native pair macro AP |
|---|---:|---:|---:|---:|---:|
| Fixed epoch 800 | 0.842742 | 0.373718 | 0.801819 | 0.526279 | 0.006231 |
| Validation-best | 0.847916 | 0.392746 | 0.803202 | 0.529539 | 0.006580 |

The residue-level rankings contain signal, but both routes remain below the
published joint MIPE AUPRC values. Native attention is not an adequate
residue-pair contact predictor; this negative result motivates the explicit
pair-contact objective in the distributed pretraining curriculum.

## ParaSurf context

The [ParaSurf paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11855283/) reports
paratope PR AUC/ROC AUC `0.781/0.967` for its Fv model on the full 626-complex
MIPE benchmark (and `0.782/0.972` for Fab). Those published numbers are not a
head-to-head ranking against this 64-complex supplied-test replay: the evaluated
cohort, model inputs, residue scope, and aggregation differ. They are retained
only as external context and do not select or promote an ABPairFM checkpoint.
