# SAbDab2 ESM-2 test-455 handoff status

## Completed export

The authenticated canonical SAbDab2 `frozen_split=test` subset was exported at:

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1
```

The directory is directly compatible with the existing sequence-SHA-addressed
embedding loader. It contains 455 instance mappings and 514 unique ESM-2
artifacts (536,892,184 logical bytes).

Verified identities:

- requested instance ID-set SHA-256:
  `9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8`
- source embedding manifest SHA-256:
  `9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545`
- subset embedding manifest SHA-256:
  `67c2ee885771cba395e42809bad9c7b0e3d6d02a323b00059d295b365498758f`
- instance-to-embedding map SHA-256:
  `f95ab224044011cc103a876e088c942952de74f02a2541d92ea7ef131257dc6b`
- artifact tree SHA-256:
  `37d535648204e0d3b8cde291f6566af7cadf1ebaaa9acb9e983eaee3ce8e0e1b`

All artifact hashes, model identity fields, sequence lengths, tensor shapes,
residue-axis coverage, and finite-value checks passed. No structures, task
labels, model predictions, or metrics were read by the exporter.

## Remaining cross-host identity check

The requester path
`/data/ab02/yangwuyu/framecontact_case_study_20260904/test_split_request_v1/`
is not mounted on the ParaCloud host. Therefore the completed export is bound
to our authenticated canonical 455-sample split, but cannot yet be claimed to
match the requester's independently stored list byte-for-byte. Copying the
request README and ID manifest to an accessible location is sufficient to run
the exact-set check. If its sorted newline-delimited ID SHA-256 equals
`9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8`,
the current export already matches and no tensor work needs to be repeated.
