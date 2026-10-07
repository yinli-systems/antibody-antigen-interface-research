"""From-scratch training; this program has no test-dataset or checkpoint-input option."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time

from protocol import SEEDS, read, require, safe_path, selection_improves, selection_key, sha, validate_split, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--smoke-receipt-sha256')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    require(os.environ.get('SLURM_JOB_PARTITION') == 'gpu_4090', 'Requires GPU allocation')
    require(sha(bundle / 'BUNDLE.json') == args.bundle_sha256, 'Bundle authority differs')
    inventory = read(bundle / 'BUNDLE.json')['files']
    require(Path(__file__).resolve() == bundle / 'runner/train.py', 'Run the frozen bundled source')
    for relative, digest in inventory.items():
        require(sha(safe_path(bundle, relative)) == digest, f'Bundle file changed: {relative}')
    with (bundle / 'authority/FULL_SPLIT.csv').open(newline='') as handle:
        full_rows = list(csv.DictReader(handle))
    validate_split(full_rows)
    with (bundle / 'authority/DEVELOPMENT_SPLIT.csv').open(newline='') as handle:
        development = list(csv.DictReader(handle))
    require(development == [r for r in full_rows if r['role'] != 'test'], 'Development split differs')
    if not args.smoke:
        require(args.smoke_receipt is not None and args.smoke_receipt_sha256 is not None,
                'Successful frozen-bundle smoke receipt required before training')
        require(sha(args.smoke_receipt) == args.smoke_receipt_sha256, 'Smoke receipt hash differs')
        smoke = read(args.smoke_receipt)
        require(smoke['status'] == 'PASS_ALL_DEVELOPMENT_LOADERS_AND_TRAIN_SMOKE'
                and smoke['bundle_sha256'] == args.bundle_sha256, 'Smoke did not pass this exact bundle')
    protocol = read(bundle / 'PROTOCOL.json')
    require(protocol['initialization'] == 'random_from_scratch_no_checkpoint', 'Invalid initialization')
    config = protocol['configuration']
    require(not args.output.exists(), 'Never overwrite a run')
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(bundle / 'source/src'), str(bundle / 'source/scripts')]
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score
    import train_framecontact as training
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset, Sabdab2ContactBatchCollator
    require(Path(training.__file__).resolve() == bundle / 'source/scripts/train_framecontact.py', 'Wrong source loaded')
    require(torch.cuda.is_available(), 'CUDA required')
    torch.set_num_threads(1)
    training._set_seed(args.seed)
    device = torch.device('cuda')
    datasets = {}
    for role in ('train', 'validation'):
        base = Sabdab2StructureEmbeddingDataset(bundle / 'development/structure',
            Path(protocol['embedding_root']), None, split_role=role,
            split_path=bundle / 'authority/DEVELOPMENT_SPLIT.csv',
            split_sha256=inventory['authority/DEVELOPMENT_SPLIT.csv'])
        data = Sabdab2ContactDataset(base, bundle / 'development/contact')
        require({r['instance_id'] for r in data.records}
                == {r['instance_id'] for r in development if r['role'] == role}, 'Dataset roles differ')
        datasets[role] = data
    train_data = datasets['train']
    collator = Sabdab2ContactBatchCollator()
    model = training._model_from_configuration(train_data[0], config, protocol['experiment']).to(device)
    initial_hash = training._state_sha256(model)
    write(args.output / 'STARTED.json', {'bundle_sha256': args.bundle_sha256, 'seed': args.seed,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'initial_state_sha256': initial_hash,
        'old_checkpoint_loaded': False, 'test_label_artifacts_opened': False, 'smoke': args.smoke,
        'runner_sha256': sha(__file__), 'train_count': len(train_data), 'validation_count': len(datasets['validation'])})
    if args.smoke:
        for role, dataset in datasets.items():
            for index in range(len(dataset)):
                item = dataset[index]
                require(item['instance_id'] == dataset.records[index]['instance_id'], 'Loader changed ID')
                require(tuple(item['pair_labels'].shape)
                        == (item['antibody_features'].shape[0], item['antigen_features'].shape[0]), 'Bad pair axes')
                require(torch.isfinite(item['antibody_features']).all().item()
                        and torch.isfinite(item['antigen_features']).all().item(), 'Nonfinite embedding')
                if (index + 1) % 500 == 0 or index + 1 == len(dataset):
                    print({'stage': 'all_loader_preflight', 'role': role, 'completed': index + 1}, flush=True)
    weights = training.FrameContactLossWeights(contact=config['contact_loss_weight'],
        paratope=config['paratope_loss_weight'], epitope=config['epitope_loss_weight'],
        specificity=0.0, distance=0.0, contrastive=0.0)
    objective = training.FrameContactObjective(weights=weights,
        specificity_margin=config['specificity_margin'], score_fraction=config['score_fraction'],
        specificity_score_source='total_contact', contact_focal_gamma=config['contact_focal_gamma'],
        epitope_dice_weight=config['epitope_dice_weight'], epitope_count_weight=config['epitope_count_weight'],
        contrastive_temperature=config['contrastive_temperature'],
        contrastive_maximum_residues=config['contrastive_maximum_residues'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(args.seed)

    def forward(item, training_mode=False):
        batch = training._move_batch(collator([item]), device)
        inputs = {k: v for k, v in batch.items() if 'label' not in k}
        with training._autocast(device, config['precision'] == 'bf16'), \
                training._saved_tensor_context(config['offload_saved_tensors'] and training_mode):
            outputs = model(inputs, return_attention=False)
            loss = objective(training._cognate_training_view(outputs), batch, wrong_partner_outputs=None)['loss'] \
                if training_mode else None
        return batch, outputs, loss

    def evaluate(indices):
        model.eval()
        aps = {task: [] for task in ('contact', 'paratope', 'epitope')}
        counts = {task: {k: 0 for k in ('tp', 'fp', 'tn', 'fn')} for task in aps}
        with torch.inference_mode():
            for index in indices:
                batch, outputs, _ = forward(datasets['validation'][index])
                for task, label, mask in [('contact', 'pair_labels', 'pair_candidate_mask'),
                                         ('paratope', 'antibody_labels', 'antibody_mask'),
                                         ('epitope', 'antigen_labels', 'antigen_mask')]:
                    m = outputs[mask][0]
                    y = batch[label][0][m].detach().cpu().numpy().astype(bool)
                    s = outputs[task + '_logits'][0][m].detach().float().cpu().numpy()
                    require(np.isfinite(s).all() and y.size > 0, 'Nonfinite/empty evaluation')
                    if y.any():
                        aps[task].append(float(average_precision_score(y, s)))
                    predicted = s >= 0.0  # raw sigmoid >= 0.5, without calibration
                    for key, value in {'tp': (y & predicted).sum(), 'fp': (~y & predicted).sum(),
                                       'tn': (~y & ~predicted).sum(), 'fn': (y & ~predicted).sum()}.items():
                        counts[task][key] += int(value)
        return {task: {'mpc_ap': float(np.mean(aps[task])) if aps[task] else None,
                       'mpc_positive_complex_count': len(aps[task]), 'fixed_05_counts': counts[task]} for task in aps}

    best_key, best_epoch, best_file = None, None, None
    trace = []
    for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
        started = time.monotonic()
        model.train()
        schedule = rng.permutation(len(train_data)).tolist()
        if args.smoke:
            sizes = sorted(range(len(train_data)), key=lambda i: train_data.records[i]['candidate_pair_count'])
            schedule = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
        epoch_loss = 0.0
        for step, index in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            _, outputs, loss = forward(train_data[index], training_mode=True)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            if step % 200 == 0 or step == len(schedule) or args.smoke:
                print({'epoch': epoch, 'step': step, 'total': len(schedule), 'loss': float(loss.detach().cpu()),
                       'gradient_norm': float(norm.detach().cpu()), 'elapsed_seconds': time.monotonic() - started}, flush=True)
            del outputs, loss
        indices = [0, len(datasets['validation']) - 1] if args.smoke else list(range(len(datasets['validation'])))
        state_before_eval = training._state_sha256(model)
        validation = evaluate(indices)
        require(training._state_sha256(model) == state_before_eval, 'Validation changed model state')
        key = selection_key(validation)
        improved = selection_improves(key, best_key)
        if improved:
            best_key, best_epoch = key, epoch
            best_file = args.output / f'checkpoint-epoch-{epoch:03d}.pt'
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'epoch': epoch, 'seed': args.seed, 'bundle_sha256': args.bundle_sha256,
                        'initial_state_sha256': initial_hash, 'validation': validation}, best_file)
        row = {'epoch': epoch, 'mean_train_loss': epoch_loss / len(schedule),
               'validation': validation, 'selected': improved, 'seconds': time.monotonic() - started,
               'schedule_instance_ids': [train_data.records[i]['instance_id'] for i in schedule]}
        write(args.output / f'epoch-{epoch:03d}.json', row)
        trace.append({k: v for k, v in row.items() if k != 'schedule_instance_ids'})
        print({'epoch': epoch, 'validation': validation, 'best_epoch': best_epoch}, flush=True)
        if epoch - best_epoch >= config['early_stopping_patience']:
            break
    require(training._state_sha256(model) != initial_hash, 'No weight update occurred')
    write(args.output / 'RESULT.json', {'status': 'PASS_ALL_DEVELOPMENT_LOADERS_AND_TRAIN_SMOKE' if args.smoke
        else 'PASS_TRAINING_COMPLETE_CHECKPOINT_FROZEN', 'bundle_sha256': args.bundle_sha256,
        'seed': args.seed, 'initial_state_sha256': initial_hash, 'old_checkpoint_loaded': False,
        'test_label_artifacts_opened': False, 'test_selection_performed': False,
        'best_epoch': best_epoch, 'checkpoint': best_file.name, 'checkpoint_sha256': sha(best_file),
        'trace': trace, 'smoke_receipt_sha256': args.smoke_receipt_sha256})
    print('COMPLETE', str(args.output / 'RESULT.json'), sha(args.output / 'RESULT.json'), flush=True)


if __name__ == '__main__':
    main()
