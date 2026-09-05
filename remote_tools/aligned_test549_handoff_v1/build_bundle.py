"""Copy existing frozen artifacts only; never recompute structures or labels."""
import argparse
import shutil
from pathlib import Path

from common import audit_splits, local_path, read_json, require, sha, split_rows, write_json, verify_development_receipt

ROOT = Path('/ssd/scxi253/model_deployment')
DEV = ROOT / 'results/development/sabdab2_sequence_predicted_geometry_development_v1'
TEST = ROOT / 'results/exposed/abf30b_sabdab2_historical_test_v1/data'
CHECKPOINTS = {
    'baseline': ['8ac02a84e6d6ad13cb1e71a6a615b38431906e69e17773a8430218da5a3b65c2',
                 '7ddae81aa61f4a403077df7c2406bcdb93e606b49623c3e4fd93e6c91f2616e5',
                 '0ed02b23d0639c92fbaad64460ab38cfb10523fc2a9ab147cbf1dcdff6561fc3'],
    'edge_topk2_supervised': ['1935300f4271f175bf77b421c4a5645894f2c8be556ffee023c44c741f4517f3',
                            'eb3b4739cae008d6a26a891de6aada42ebb000e4020b601eeca781229e55fd18',
                            '8c423f52a8c0c55ff8d6dc275b7ea40953cf89d93da0c93a53a246e1f779673b']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), 'Never overwrite an existing bundle, including failed builds')
    out.mkdir(parents=True)
    inventory, sources = {}, {}

    def copy(src, relative, expected=None):
        src = Path(src)
        observed = sha(src)
        require(expected is None or observed == expected, f'Source SHA differs: {src}')
        dst = local_path(out, relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        require(not dst.exists(), f'Duplicate destination: {relative}')
        shutil.copyfile(src, dst)  # Independent bytes, no symlink or writable shared inode.
        require(sha(dst) == observed, f'Copy SHA differs: {relative}')
        inventory[relative] = observed
        return dst

    def emit(relative, value):
        write_json(out / relative, value)
        inventory[relative] = sha(out / relative)

    dev = copy(DEV / 'data/structure_cache_v1/DEVELOPMENT_SPLIT_VIEW.csv',
               'authority/DEVELOPMENT_SPLIT_VIEW.csv',
               'ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700')
    test = copy(TEST / 'test_structure_cache_v1/TEST_SPLIT_VIEW.csv',
                'inputs/structure/TEST_SPLIT_VIEW.csv',
                '84acf74ce3debc3e176c996fd04b75510daf84183060d5d8353fc17758705a46')
    old = copy(ROOT / 'handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1/REQUESTED_INSTANCE_IDS.txt',
               'authority/LEGACY455_IDS.txt',
               '9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8')
    audit = audit_splits(split_rows(dev), split_rows(test), old.read_text().splitlines())
    require(audit['old455_role_counts'] == {'train': 272, 'validation': 73, 'test': 110},
            'Legacy overlap changed')
    emit('authority/SPLIT_MISMATCH_AUDIT.json', audit)
    manifests = {}
    for kind, source_dir, target, expected in (
        ('STRUCTURE', 'test_structure_cache_v1', 'inputs/structure',
         'b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb'),
        ('CONTACT', 'test_contact_authority_adapter_v1', 'labels/contact',
         'ad7201ecdf339b16a666addbaa5f775a4a1da03fdf0ea5bcdd68e8aa0f6e8e41')):
        source = TEST / source_dir
        path = copy(source / f'{kind}_CACHE_MANIFEST.json', f'{target}/{kind}_CACHE_MANIFEST.json', expected)
        manifests[kind] = read_json(path)
        require(len(manifests[kind]['records']) == 549, 'Incomplete cohort')
        copy(source / f'{kind}_CACHE_RECEIPT.json', f'{target}/{kind}_CACHE_RECEIPT.json',
             'e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69' if kind == 'STRUCTURE'
             else 'e840d747bc36e00427b08d43c324c98c9ed8df0e79484c89a0dbe189990ff4ef')
        for row in manifests[kind]['records']:
            copy(local_path(source, row['artifact_path']), f"{target}/{row['artifact_path']}", row['artifact_sha256'])
    copy(TEST / 'test_structure_cache_v1/TEST_STRUCTURE_CACHE_BINDING_V1.json',
         'inputs/structure/TEST_STRUCTURE_CACHE_BINDING_V1.json',
         '165690c13a2d75f50dab8262cf48fbf3e1089e4f9cc5895ddfa4b85e3e09aa06')
    embroot = ROOT / 'data/sppider_seq_sabdab2_embeddings'
    parent = copy(embroot / 'SABDAB2_ESM_EMBEDDING_MANIFEST.json',
                  'authority/PARENT_ESM2_MANIFEST.json',
                  '9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545')
    emb = read_json(parent)
    keys = {r[f'{partner}_sequence_sha256'] for r in manifests['STRUCTURE']['records']
            for partner in ('antibody', 'antigen')}
    records = {r['sequence_sha256']: r for r in emb['records']}
    require(keys <= records.keys(), 'Missing ESM-2 sequences')
    subset = [records[k] for k in sorted(keys)]
    for row in subset:
        copy(local_path(embroot, row['artifact_path']), f"inputs/embeddings/{row['artifact_path']}", row['artifact_sha256'])
    emb.update(records=subset, selected_sequence_count=len(subset),
               complete_cohort_sequence_count=len(subset), complete=True)
    emb['handoff_parent_sha256'] = sha(parent)
    emb['handoff_scope'] = '549 antigen-fold historical test instances only'
    emit('inputs/embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json', emb)
    components = []
    for arm in CHECKPOINTS:
        for seed, expected in zip((17, 29, 43), CHECKPOINTS[arm]):
            run = DEV / 'runs_no_privileged_distance_v1' / arm / f'seed-{seed}'
            key = f'{arm}/seed-{seed}'
            receipt = read_json(run / 'RUN_RECEIPT.json')
            result_path = run / 'FRAMECONTACT_DEVELOPMENT_RESULT.json'
            verify_development_receipt(receipt, expected, sha(result_path), seed)
            result = read_json(result_path)
            conf = result['configuration']
            require(conf['distance_loss_weight'] == 0 and conf['distance_label_manifest_sha256'] is None,
                    'Privileged distance objective is not allowed')
            require(conf['split_manifest_sha256'] == sha(dev)
                    and conf['embedding_manifest_sha256'] == sha(parent), 'Training authority differs')
            source_manifest = Path(conf['source_manifest'])
            require(sha(source_manifest) == conf['source_manifest_sha256'], 'Model source authority differs')
            source = read_json(source_manifest)
            if arm not in sources:
                copy(source_manifest, f'source/{arm}/SOURCE_MANIFEST.json', conf['source_manifest_sha256'])
                for row in source['records']:
                    copy(local_path(Path(source['source_root']), row['path']),
                         f"source/{arm}/{row['path']}", row['sha256'])
                sources[arm] = conf['source_manifest_sha256']
            else:
                require(sources[arm] == conf['source_manifest_sha256'], 'Cross-seed source mismatch')
            copy(run / 'FRAMECONTACT_SELECTED.pt', f'checkpoints/{key}/FRAMECONTACT_SELECTED.pt', expected)
            copy(run / 'RUN_RECEIPT.json', f'checkpoints/{key}/RUN_RECEIPT.json')
            validation = result['selected_checkpoint_validation']
            calibration = {
                task: {'temperature': float(validation['validation_temperature'] if task == 'contact'
                                              else validation[task]['validation_temperature']),
                       'threshold': float(validation[task]['threshold_selected_on_validation'])}
                for task in ('contact', 'epitope', 'paratope')}
            emit(f'checkpoints/{key}/CONFIG.json',
                 {'configuration': conf, 'experiment': result['experiment'],
                  'calibration': calibration, 'selected_epoch': result['selection']['best_epoch'],
                  'development_result_sha256': receipt['result_sha256']})
            components.append({'arm': arm, 'seed': seed, 'key': key,
                               'checkpoint_sha256': expected, 'selected_epoch': result['selection']['best_epoch']})
    for filename in ('common.py', 'run_bundle.py', 'finalize_bundle.py', 'README.md'):
        copy(Path(__file__).parent / filename, filename)
    bundle = {'format_version': 'aligned_test549_handoff_v1',
              'status': 'PACKAGED_AWAITING_PRODUCTION_LOADER_AND_FORWARD_VERIFICATION',
              'instance_count': 549, 'unique_embeddings': len(keys),
              'components': components, 'files': inventory,
              'routing': {'contact': 'baseline', 'paratope': 'baseline', 'epitope': 'edge_topk2_supervised'},
              'claim_boundary': {'historical_exposed_test': True, 'fresh_blind_test': False,
                                 'sequence_double_cold_proven': False,
                                 'model_selection_or_retraining': False,
                                 'experimental_coordinates_as_input': False,
                                 'experimental_contact_labels_for_measurement': True,
                                 'MSA_homologs_may_be_used_by_AF3': True,
                                 'old455_is_not_checkpoint_test': True}}
    write_json(out / 'BUNDLE.json', bundle)
    print(f'PACKAGED {out} BUNDLE_SHA256={sha(out / "BUNDLE.json")} files={len(inventory)}', flush=True)


if __name__ == '__main__':
    main()
