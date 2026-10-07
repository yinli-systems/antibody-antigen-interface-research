"""Verify all six receipts; route tasks without selecting or averaging checkpoints."""
import argparse
import csv
import statistics
from pathlib import Path

from common import read_json, require, sha, verify_bundle, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--bundle-sha256', required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--mode', choices=('smoke', 'full'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Final output already exists')
    bundle = verify_bundle(args.bundle.resolve(), args.bundle_sha256)
    results, hashes = {}, {}
    for i, component in enumerate(bundle['components']):
        folder = args.runs / str(i)
        receipt, result = read_json(folder / 'RECEIPT.json'), read_json(folder / 'RESULT.json')
        require(sha(folder / 'RESULT.json') == receipt['result_sha256'], 'Result SHA differs')
        require(result['component'] == component and receipt['checkpoint_sha256'] == component['checkpoint_sha256'],
                'Component identity differs')
        require(receipt['bundle_sha256'] == result['bundle_sha256'] == args.bundle_sha256, 'Wrong data authority')
        require(result['status'] == ('PASS_COMPONENT_SMOKE' if args.mode == 'smoke' else 'PASS_FULL_COMPONENT'),
                'Missing successful component')
        if args.mode == 'full':
            require(result['forward_instances'] == 549, 'Incomplete test evaluation')
            for name, digest in result['prediction_sha256'].items():
                require(sha(folder / name) == digest, 'Prediction SHA differs')
        else:
            require(result['forward_instances'] >= 1, 'Missing forward smoke')
        results[(component['arm'], component['seed'])] = result
        hashes[str(i)] = {'result_sha256': sha(folder / 'RESULT.json'), 'receipt_sha256': sha(folder / 'RECEIPT.json')}
    require(args.mode != 'smoke' or results[('baseline', 17)]['loaded_all_instances'] == 549,
            'Full production loader verification absent')
    final = {'status': 'PASS_MODEL_READY_ALIGNED_TEST549_BUNDLE' if args.mode == 'smoke' else 'PASS_ROUTED_HISTORICAL_TEST549',
             'bundle_sha256': args.bundle_sha256, 'component_receipts': hashes,
             'claim_boundary': bundle['claim_boundary'], 'routing': bundle['routing']}
    args.output.mkdir(parents=True)
    if args.mode == 'full':
        rows = []
        for seed in (17, 29, 43):
            for task, arm in bundle['routing'].items():
                result = results[(arm, seed)]
                m = result['metrics'][task]
                row = {'seed': seed, 'task': task, 'component': arm,
                       'selected_epoch': result['component']['selected_epoch']}
                for key in ('auroc', 'auprc', 'MPC_AP', 'precision', 'recall', 'f1', 'mcc',
                            'TP', 'FP', 'TN', 'FN', 'threshold', 'temperature'):
                    row[key] = m[key]
                rows.append(row)
        with (args.output / 'ROUTED_METRICS.csv').open('x', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        final['routed_per_seed'] = rows
        final['seed_medians'] = {task: {key: statistics.median(r[key] for r in rows if r['task'] == task)
            for key in ('auroc', 'auprc', 'MPC_AP', 'precision', 'recall', 'f1', 'mcc')}
            for task in bundle['routing']}
    write_json(args.output / 'RESULT.json', final)
    write_json(args.output / 'RECEIPT.json', {'status': final['status'], 'result_sha256': sha(args.output / 'RESULT.json'),
                                           'bundle_sha256': args.bundle_sha256})
    print(final['status'], flush=True)


if __name__ == '__main__':
    main()
