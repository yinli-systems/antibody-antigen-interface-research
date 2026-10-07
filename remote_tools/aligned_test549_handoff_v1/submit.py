"""Submit the fixed packaging -> six-component replay -> summary dependency chain."""
import argparse
import json
import subprocess
from pathlib import Path

from common import read_json, require, sha, write_json

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt-name', default='SUBMISSION_RECEIPT.json')
parser.add_argument('--transfer-manifest-sha256', required=True)
args = parser.parse_args()
require(Path(args.receipt_name).name == args.receipt_name, 'Receipt name must be a basename')
require(sha(root / 'TRANSFER.json') == args.transfer_manifest_sha256, 'Transfer manifest mismatch')
for name, digest in read_json(root / 'TRANSFER.json').items():
    require(Path(name).name == name and sha(root / name) == digest, f'Incomplete transfer: {name}')
receipt_path = root / args.receipt_name
require(not receipt_path.exists(), 'Already submitted; inspect existing jobs instead')
jobs = {}
try:
    for name, script in (('prepare', 'prepare.sbatch'), ('evaluate', 'evaluate.sbatch'), ('analyze', 'analyze.sbatch')):
        command = ['sbatch', '--parsable']
        if jobs:
            command.append('--dependency=afterok:' + list(jobs.values())[-1])
        command.append(str(root / script))
        job = subprocess.check_output(command, text=True, timeout=40).strip().split(';')[0]
        require(job.isdigit(), f'Unknown sbatch response: {job}')
        jobs[name] = job
        print(name, job, flush=True)
finally:
    write_json(receipt_path, {'jobs': jobs,
        'tool_sha256': {p.name: sha(p) for p in sorted(root.iterdir())
                        if p.is_file() and p.suffix in {'.py', '.sbatch', '.md'}},
        'status': 'SUBMITTED' if len(jobs) == 3 else 'PARTIAL_SUBMISSION_INSPECT_DO_NOT_RETRY',
        'model_update': False, 'test_selection': False, 'threshold_refit': False,
        'scope': 'Existing 549-instance exposed historical test; correct checkpoint/data handoff'})
print(json.dumps(jobs), flush=True)
