"""Read-only component inference using its exact frozen production source."""
import argparse
import json
import os
from pathlib import Path
import sys

from common import local_path, read_json, require, sha, verify_bundle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--component', type=int, choices=range(6), required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists; never silently retry/overwrite')
    root = args.bundle.resolve()
    bundle = verify_bundle(root, args.bundle_sha256)
    component = bundle['components'][args.component]
    source = root / 'source' / component['arm']
    sys.path.insert(0, str(source / 'src'))
    sys.path.insert(0, str(source / 'scripts'))
    import numpy as np
    import torch
    import train_framecontact as training
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset, Sabdab2ContactBatchCollator
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset
    from interaction_first.framecontact_metrics import binary_classification_metrics
    from sklearn.metrics import average_precision_score

    if os.environ.get('SLURM_JOB_ID'):
        require(os.environ.get('SLURM_JOB_PARTITION') == 'gpu_4090', 'ParaCloud requires gpu_4090')
    elif str(root).startswith('/ssd/scxi253/'):
        raise ValueError('No inference on the ParaCloud login node')
    require(torch.cuda.is_available(), 'CUDA is required')
    require(Path(training.__file__).resolve() == source / 'scripts/train_framecontact.py',
            'Wrong training module imported')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    training._set_seed(component['seed'])
    config = read_json(root / 'checkpoints' / component['key'] / 'CONFIG.json')
    conf = config['configuration']
    split = root / 'inputs/structure/TEST_SPLIT_VIEW.csv'
    base = Sabdab2StructureEmbeddingDataset(root / 'inputs/structure', root / 'inputs/embeddings',
        None, split_role='test', split_path=split, split_sha256=sha(split))
    dataset = Sabdab2ContactDataset(base, root / 'labels/contact')
    require(len(dataset) == 549, 'Loader did not return the full 549 cohort')
    collator = Sabdab2ContactBatchCollator()
    # The first component checks every production-loader item, including long/chunked ESM-2.
    loaded = 0
    if args.mode == 'smoke' and args.component == 0:
        for index in range(len(dataset)):
            item = dataset[index]
            for partner in ('antibody', 'antigen'):
                require(bool(torch.isfinite(item[f'{partner}_features']).all()), 'Nonfinite ESM-2')
                valid = item[f'{partner}_frame_valid_mask']
                require(bool(torch.isfinite(item[f'{partner}_backbone'][valid]).all()),
                        'Nonfinite valid backbone frame')
            collator([item])
            loaded += 1
    largest = max(range(len(base)), key=lambda i: base.records[i]['antibody_length'] * base.records[i]['antigen_length'])
    indices = list(range(len(dataset))) if args.mode == 'full' else sorted({0, largest})
    first = dataset[0]
    model = training._model_from_configuration(first, conf, config['experiment'])
    path = root / 'checkpoints' / component['key'] / 'FRAMECONTACT_SELECTED.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    before = training._state_sha256(model)
    args.output.mkdir(parents=True)
    all_logits = {t: [] for t in ('contact', 'paratope', 'epitope')}
    all_labels = {t: [] for t in all_logits}
    per_complex = []
    prediction_hashes = {}
    specs = [('contact', 'contact_logits', 'pair_candidate_mask', 'pair_labels'),
             ('paratope', 'paratope_logits', 'antibody_mask', 'antibody_labels'),
             ('epitope', 'epitope_logits', 'antigen_mask', 'antigen_labels')]
    with torch.inference_mode():
        for count, index in enumerate(indices, 1):
            raw = collator([dataset[index]])
            batch = training._move_batch(raw, device)
            # Label tensors are only consumed below by the metric code, never by model.forward.
            model_input = {k: v for k, v in batch.items() if 'label' not in k}
            with training._autocast(device, conf['precision'] == 'bf16'):
                output = model(model_input, return_attention=False)
            entry = {'instance_id': dataset.records[index]['instance_id']}
            saved = {'instance_id': np.asarray(entry['instance_id'])}
            for task, logit_key, mask_key, label_key in specs:
                mask = output[mask_key][0]
                logits = output[logit_key][0][mask].float().cpu().numpy()
                labels = batch[label_key][0][mask].cpu().numpy().astype(np.int8)
                require(np.isfinite(logits).all(), 'Nonfinite forward logits')
                require(logits.shape == labels.shape and logits.size > 0, 'Invalid score axes')
                entry[task] = {'positive_count': int(labels.sum()), 'candidate_count': int(labels.size),
                    'AP': float(average_precision_score(labels, training._sigmoid_numpy(logits)))
                    if labels.sum() else None}
                all_logits[task].append(logits)
                all_labels[task].append(labels)
                saved[f'{task}_logits'] = logits
                saved[f'{task}_labels'] = labels
                saved[f'{task}_mask'] = mask.cpu().numpy()
            per_complex.append(entry)
            if args.mode == 'full':
                relative = f'predictions/{index:04d}.npz'
                prediction = args.output / relative
                prediction.parent.mkdir(exist_ok=True)
                with prediction.open('xb') as handle:
                    np.savez_compressed(handle, **saved)
                prediction_hashes[relative] = sha(prediction)
            if count % 50 == 0:
                print(json.dumps({'component': component['key'], 'completed': count, 'total': len(indices)}), flush=True)
    require(training._state_sha256(model) == before, 'Inference changed model state')
    metrics = {}
    if args.mode == 'full':
        for task in all_logits:
            logits, labels = np.concatenate(all_logits[task]), np.concatenate(all_labels[task])
            calibration = config['calibration'][task]
            probabilities = training._sigmoid_numpy(logits / calibration['temperature'])
            scores = binary_classification_metrics(labels, probabilities, threshold=calibration['threshold'])
            eligible = [r[task]['AP'] for r in per_complex if r[task]['AP'] is not None]
            pred = probabilities >= calibration['threshold']
            truth = labels.astype(bool)
            scores.update(MPC_AP=float(np.mean(eligible)), MPC_AP_eligible_complexes=len(eligible),
                MPC_AP_zero_positive_excluded=549-len(eligible),
                threshold=calibration['threshold'], temperature=calibration['temperature'],
                TP=int((pred & truth).sum()), FP=int((pred & ~truth).sum()),
                TN=int((~pred & ~truth).sum()), FN=int((~pred & truth).sum()))
            # Fixed-0.5 is a separate readout, not a replacement or calibration refit.
            scores['fixed_raw_sigmoid_0_5'] = binary_classification_metrics(
                labels, training._sigmoid_numpy(logits), threshold=0.5)
            scores['fixed_calibrated_probability_0_5'] = binary_classification_metrics(
                labels, probabilities, threshold=0.5)
            metrics[task] = scores
    result = {'status': 'PASS_FULL_COMPONENT' if args.mode == 'full' else 'PASS_COMPONENT_SMOKE',
              'component': component, 'bundle_sha256': args.bundle_sha256,
              'loaded_all_instances': loaded, 'forward_instances': len(indices),
              'metrics': metrics, 'per_complex': per_complex, 'prediction_sha256': prediction_hashes,
              'audit': {'train_validation_test_exact_id_overlap': False,
                        'historical_exposed_test': True, 'fresh_blind_test': False,
                        'model_parameters_updated': False, 'labels_passed_to_model': False,
                        'calibration_or_threshold_refit': False,
                        'experimental_bound_coordinates_as_model_input': False},
              'runtime': {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
                          'job_id': os.environ.get('SLURM_JOB_ID'),
                          'source_module': str(Path(training.__file__).relative_to(root))}}
    write_json(args.output / 'RESULT.json', result)
    write_json(args.output / 'RECEIPT.json', {'status': result['status'], 'result_sha256': sha(args.output / 'RESULT.json'),
        'checkpoint_sha256': component['checkpoint_sha256'], 'bundle_sha256': args.bundle_sha256})
    print(json.dumps({'status': result['status'], 'component': component['key'], 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
