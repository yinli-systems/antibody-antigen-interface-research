# Prospective study collaborator intake

Complete this page before filling and freezing `PROTOCOL_DRAFT_V1.json`. This is
an experimental-design intake, not a wet-lab procedure.

## 1. Target and safety gate

- Target/common name:
- Exact antigen construct and artifact or vendor/lot identifier:
- Antigen artifact SHA-256 (when a digital sequence/structure is supplied):
- Molecule type and biological context:
- Intended epitope, defined before candidate ranking:
- Structure source and state (bound, unbound, predicted, or mixed):
- Independent safety reviewer and review date:
- Confirm that the target is neither a pathogen component nor a toxin: yes/no
- Confirm that gain-of-function work is outside scope: yes/no

Any `no` or unresolved answer in the last three items blocks this study.

## 2. Assay-readiness gate

- Primary assay name:
- Numeric primary measurement:
- Hit threshold, unit, and direction (`>=` or `<=`):
- Positive control with expected behavior:
- Negative control with expected behavior:
- Orthogonal confirmation assay:
- Can the assay operator remain blind to policy membership? yes/no
- Estimated capacity and turnaround time:

The positive control must satisfy, and the negative control must fail, the same
frozen primary hit rule. Otherwise the analysis fails closed.

## 3. Candidate and budget gate

- Fixed proposal generator and exact revision/weight SHA-256:
- Proposal-pool manifest SHA-256:
- Candidate format or construct constraints:
- Eligibility rule that can be evaluated without assay outcomes:
- Maximum unique candidates that can be produced and assayed:
- Candidate budget per policy arm, `K` (3–96):
- Confirm that one overlapping candidate can be assayed once and reused across
  policy analyses: yes/no

The same frozen proposal pool feeds all three policies. Candidate generation or
eligibility cannot differ by arm.

## 4. Blinding and custody

- Private candidate-to-arm mapping custodian:
- Blinded assay-results custodian:
- Confirm these are different people or independently controlled roles: yes/no
- Location for the private blinding key and mapping (must not be Git-tracked):
- Location for the public blinded manifest and its receipt:

## 5. Pre-freeze sign-off

- No candidate-level assay outcome has been observed: yes/no
- Generator, proposal pool, score implementations, and diversity clustering are
  bound by SHA-256: yes/no
- Primary endpoint, controls, budget, and analysis are fixed: yes/no
- Failed expression and missing confirmation will remain reported: yes/no
- Result will be interpreted only for this target, pool, assay, and slate:
  yes/no

Only an all-`yes` sign-off authorizes `freeze-protocol`. The completed intake
should be preserved with the frozen protocol receipt but should not contain
credentials, proprietary sequences, or the blinding key.
