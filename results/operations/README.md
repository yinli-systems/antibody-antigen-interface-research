# Operational snapshot — not a live monitor

Observed: **2026-09-30T08:09:57 UTC**. Generated from [snapshot_20260930T080957.json](snapshot_20260930T080957.json); no GPU jobs were submitted, cancelled or restarted by publication tooling.

## Frozen structural-PLM sweep

**12/36** completed receipts; **0** recorded formal failure files at this snapshot. The final analysis remains dependent on the full array.

| Task | Arm | Seed | Validation-best pair AP | Last-epoch pair AP |
|---:|---|---:|---:|---:|
| 0 | saprot_layer_00 | 1 | 0.017988 | 0.015839 |
| 1 | saprot_layer_00 | 2 | 0.021676 | 0.018874 |
| 2 | saprot_layer_00 | 3 | 0.022082 | 0.018469 |
| 3 | saprot_layer_08 | 1 | 0.020780 | 0.018392 |
| 4 | saprot_layer_08 | 2 | 0.020481 | 0.018313 |
| 5 | saprot_layer_08 | 3 | 0.021858 | 0.018770 |
| 6 | saprot_layer_16 | 1 | 0.022855 | 0.020608 |
| 7 | saprot_layer_16 | 2 | 0.023087 | 0.019798 |
| 8 | saprot_layer_16 | 3 | 0.023350 | 0.019869 |
| 9 | saprot_layer_24 | 1 | 0.022856 | 0.019488 |
| 10 | saprot_layer_24 | 2 | 0.025876 | 0.021018 |
| 11 | saprot_layer_24 | 3 | 0.024624 | 0.021638 |

These intermediate values do not select a winner and are not independent test scores.

## Actual scheduler records

```text
1640503_0|struct-plm-train|COMPLETED|0:0|02:36:35
1640503_1|struct-plm-train|COMPLETED|0:0|02:36:57
1640503_2|struct-plm-train|COMPLETED|0:0|02:37:13
1640503_3|struct-plm-train|COMPLETED|0:0|02:39:56
1640503_4|struct-plm-train|COMPLETED|0:0|02:34:12
1640503_5|struct-plm-train|COMPLETED|0:0|02:33:50
1640503_6|struct-plm-train|COMPLETED|0:0|02:40:34
1640503_7|struct-plm-train|COMPLETED|0:0|02:40:01
1640503_8|struct-plm-train|COMPLETED|0:0|02:32:50
1640503_9|struct-plm-train|COMPLETED|0:0|02:35:56
1640503_10|struct-plm-train|COMPLETED|0:0|02:36:32
1640503_11|struct-plm-train|COMPLETED|0:0|02:38:28
1640503_12|struct-plm-train|RUNNING|0:0|00:16:05
1640503_13|struct-plm-train|RUNNING|0:0|00:12:34
1640503_14|struct-plm-train|RUNNING|0:0|00:11:34
1640503_15|struct-plm-train|RUNNING|0:0|00:11:04
1640503_16|struct-plm-train|RUNNING|0:0|00:11:04
1640503_17|struct-plm-train|RUNNING|0:0|00:07:04
1640503_[18-35%6]|struct-plm-train|PENDING|0:0|00:00:00
1640514|struct-plm-analyze|PENDING|0:0|00:00:00
1640914_0|sabdab2-sdh-af3|FAILED|2:0|00:00:01
1641607_[0%4]|ab-af3r-6db2ca82|PENDING|0:0|00:00:00
1641627|ab-layer-schema-fix|PENDING|0:0|00:00:00
```

AF3 job `1640914_0` failed before prediction. Earlier references to `1640888` were not the actual ID found in `TASK0_JOB_ID.txt`. No structure result is promoted from that failed launch.

## Still unfinished

Full layer/control comparison; learned realignment; a newly admitted untouched OOD cohort; same-split RePaRank retraining; complete structure-RePaRank evaluation; co-folder comparison; affinity endpoint. The prior 14 OOD candidates were not admitted.


## Additive execution recovery

AF3 task 0 `1641607`: **PENDING** (Priority); audited completed structures: **0**. The remaining 39 tasks are not submitted.

Separate schema-corrected analysis `1641627`: **PENDING** (Dependency); dependency `afterok:1640503,afterany:1640514`. Original training and original analyzer were not changed.

[Verified defects, 49 tests and synthetic integration check](../../extensions/execution_recovery_v1/README.md). This is not an accuracy improvement or completed structural benchmark.
