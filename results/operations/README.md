# Operational snapshot — not a live monitor

Observed: **2026-09-30T07:18:25 UTC**. Generated from [snapshot_20260930T071825.json](snapshot_20260930T071825.json); no GPU jobs were submitted, cancelled or restarted by publication tooling.

## Frozen structural-PLM sweep

**6/36** completed receipts; **0** recorded formal failure files at this snapshot. The final analysis remains dependent on the full array.

| Task | Arm | Seed | Validation-best pair AP | Last-epoch pair AP |
|---:|---|---:|---:|---:|
| 0 | saprot_layer_00 | 1 | 0.017988 | 0.015839 |
| 1 | saprot_layer_00 | 2 | 0.021676 | 0.018874 |
| 2 | saprot_layer_00 | 3 | 0.022082 | 0.018469 |
| 3 | saprot_layer_08 | 1 | 0.020780 | 0.018392 |
| 4 | saprot_layer_08 | 2 | 0.020481 | 0.018313 |
| 5 | saprot_layer_08 | 3 | 0.021858 | 0.018770 |

These intermediate values do not select a winner and are not independent test scores.

## Actual scheduler records

```text
1640503_0|COMPLETED|0:0|02:36:35
1640503_1|COMPLETED|0:0|02:36:57
1640503_2|COMPLETED|0:0|02:37:13
1640503_3|COMPLETED|0:0|02:39:56
1640503_4|COMPLETED|0:0|02:34:12
1640503_5|COMPLETED|0:0|02:33:50
1640503_6|RUNNING|0:0|02:00:11
1640503_7|RUNNING|0:0|01:59:41
1640503_8|RUNNING|0:0|01:57:41
1640503_9|RUNNING|0:0|01:57:11
1640503_10|RUNNING|0:0|01:56:41
1640503_11|RUNNING|0:0|01:54:11
1640503_[12-35%6]|PENDING|0:0|00:00:00
1640514|PENDING|0:0|00:00:00
1640813|COMPLETED|0:0|00:01:09
1640868|FAILED|1:0|00:00:10
1640914_0|FAILED|2:0|00:00:01
```

AF3 job `1640914_0` failed before prediction. Earlier references to `1640888` were not the actual ID found in `TASK0_JOB_ID.txt`. No structure result is promoted from that failed launch.

## Still unfinished

Full layer/control comparison; learned realignment; a newly admitted untouched OOD cohort; same-split RePaRank retraining; complete structure-RePaRank evaluation; co-folder comparison; affinity endpoint. The prior 14 OOD candidates were not admitted.
