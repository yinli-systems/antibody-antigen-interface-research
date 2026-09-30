# Branch provenance and preservation map

Generated from pre-consolidation main `910b752ecf3e1493b80a7605151048bf2cd3e69b` on 2026-09-30.

No experimental branch was deleted. Every branch head below is pinned by an `archive/2026-09-30/...` tag. The machine-readable full path/blob inventory is [`archive/BRANCH_FILE_INVENTORY_20260930.json`](../archive/BRANCH_FILE_INVENTORY_20260930.json); the filtered scientific evidence inventory is [`results/BRANCH_EVIDENCE_INDEX.json`](../results/BRANCH_EVIDENCE_INDEX.json).

| Branch | Head | Ahead | Changed files | Evidence files | Role |
|---|---|---:|---:|---:|---|
| `abcite/multimodal-copilot` | `fb871f356` | 41 | 306 | 3 | adjacent multimodal AbCite project; preserved but not merged into the antibody-interface headline story |
| `feature/asep-pretraining-pipeline` | `b730070e4` | 5 | 31 | 2 | early AsEP/pretraining integration branch |
| `kevin/correspondence-upgrade-20260930` | `14a385fea` | 136 | 205 | 126 | post-freeze correspondence perturbation extension and strict GPU evidence |
| `kevin/evaluate-sensitivity-ci-fix-v1` | `ed8c33fb3` | 16 | 19 | 8 | evaluation sensitivity and CI fixes |
| `kevin/framecontact-multimodal-evidence-20260902` | `27605737a` | 2 | 536 | 160 | FrameContact multimodal evidence and paper track |
| `kevin/geometry-gate-power-reanalysis-v1` | `96e2a244c` | 1 | 6 | 3 | geometry gate power reanalysis |
| `kevin/geometry-powered-replication-v1` | `f04dbeb92` | 3 | 5 | 1 | geometry powered replication protocol |
| `kevin/graph-community-split-v1` | `822d91590` | 1 | 9 | 0 | graph/community split experiment |
| `kevin/hadamard-pair-decoder-v1` | `347acbdfb` | 8 | 9 | 3 | Hadamard pair decoder negative/ablation evidence |
| `kevin/multimodel-scale-program-20260930` | `28dffada7` | 159 | 384 | 264 | current structural-PLM, score-dissociation, RePaRank and scale programme |
| `kevin/partner-pose-contract-v1` | `c72fa83be` | 1 | 3 | 0 | partner-pose contract |
| `kevin/pretrained-representations-v1` | `400022a29` | 5 | 5 | 0 | initial pretrained representation wiring |
| `kevin/protocol-binding-ci-v1` | `191a8b241` | 1 | 8 | 3 | protocol/source binding CI |
| `kevin/representation-ablation-v1` | `55cfd137a` | 132 | 161 | 94 | controlled frozen-representation, correspondence, objective, gradient and benchmark evidence |
| `kevin/s2-split-readiness-v1` | `053a0665c` | 1 | 3 | 0 | S2 split readiness work |
| `kevin/teacher-paper-handoff-20260903` | `54afd7987` | 10 | 92 | 80 | latest FrameContact teacher/paper handoff and corrected historical test alignment |

## Preservation policy

- Branches remain untouched and continue to carry their complete code/history.
- Archive tags pin every 2026-09-30 branch head even if a branch is renamed or removed later.
- Main receives curated, canonical evidence and indexes rather than blindly merging every experimental implementation.
- Negative results, failed gates, corrections and spent-test warnings remain discoverable and are never rewritten into positive claims.
- Large/raw artifacts that do not belong in Git remain referenced by receipt/path/hash rather than copied into the repository.
