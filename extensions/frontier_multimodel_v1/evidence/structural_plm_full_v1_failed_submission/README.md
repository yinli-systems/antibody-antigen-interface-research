# Failed full-programme submission attempt

The freeze step failed because the submit command pointed to a non-existent
source-snapshot copy of `MODEL_RUNTIME_AUTHORITY.json`; the actual authority is
under the programme receipts directory. The shell used `set -e` without
`pipefail`, so `freeze | tee` returned success and arrays 1640059, 1640060 and
dependent 1640061 were submitted with an empty plan digest. Every started task
failed at the wrapper's required-plan guard in 1–2 seconds. Remaining tasks were
cancelled. No model inference, quantisation, token or feature output was produced.

The replacement submitter is repository code with `set -Eeuo pipefail`, a direct
freeze command, JSON/commit/count assertions, nonempty SHA validation, scheduler
preflights, an error trap that cancels submitted jobs, and the correct authority
path. This directory is retained rather than rewritten.
