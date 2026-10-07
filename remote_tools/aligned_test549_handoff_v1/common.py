"""Portable, fail-closed checks shared by the aligned historical-test handoff."""
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def verify_development_receipt(receipt, expected_checkpoint, expected_result, seed):
    require(receipt.get('format_version') == 'framecontact_same_protocol_adapter_development_v2',
            'Unexpected development receipt format')
    require(receipt.get('test_loaded') is False and receipt.get('external_test_loaded') is False,
            'Development receipt reports test access')
    require(receipt.get('selected_checkpoint_sha256') == expected_checkpoint,
            'Selected checkpoint digest differs')
    require(receipt.get('result_sha256') == expected_result and receipt.get('seed') == seed,
            'Development result or seed differs')


def local_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Non-portable artifact path')
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), 'Artifact escapes bundle')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent),
            'Symlink is not allowed in bundle path')
    return path


def split_rows(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len({r['instance_id'] for r in rows}) == len(rows), 'Duplicate split IDs')
    require(all(r['role'] in {'train', 'validation', 'test'} for r in rows),
            'Unknown split role')
    return {r['instance_id']: r for r in rows}


def audit_splits(development, test, old_ids):
    require(not set(development) & set(test), 'Test overlaps checkpoint development IDs')
    require(all(r['role'] == 'test' for r in test.values()), 'Non-test row in test split')
    require(all(r['role'] in {'train', 'validation'} for r in development.values()),
            'Unexpected development role')
    require(len(old_ids) == len(set(old_ids)), 'Duplicate legacy IDs')
    combined = {**development, **test}
    require(set(old_ids) <= set(combined), 'Unresolved legacy ID')
    counts = {role: sum(combined[i]['role'] == role for i in old_ids)
              for role in ('train', 'validation', 'test')}
    return {'old455_role_counts': counts,
            'old455_roles': {i: combined[i]['role'] for i in sorted(old_ids)},
            'test_not_in_old455_count': len(set(test) - set(old_ids)),
            'exact_instance_train_test_overlap': 0,
            'exact_instance_validation_test_overlap': 0,
            'sequence_double_cold_proven': False, 'fresh_blind_test': False}


def verify_bundle(root, expected_sha):
    require(sha(root / 'BUNDLE.json') == expected_sha, 'Bundle authority SHA differs')
    bundle = read_json(root / 'BUNDLE.json')
    require(bundle['format_version'] == 'aligned_test549_handoff_v1', 'Wrong bundle format')
    for relative, digest in bundle['files'].items():
        require(sha(local_path(root, relative)) == digest, f'Artifact SHA differs: {relative}')
    dev = split_rows(root / 'authority/DEVELOPMENT_SPLIT_VIEW.csv')
    test = split_rows(root / 'inputs/structure/TEST_SPLIT_VIEW.csv')
    require(len(test) == 549 and len(dev) == 3114, 'Wrong cohort size')
    require({role: sum(r['role'] == role for r in dev.values())
             for role in ('train', 'validation')} == {'train': 2565, 'validation': 549},
            'Checkpoint development role counts differ')
    require(not set(test) & set(dev), 'Test/development overlap')
    structure = read_json(root / 'inputs/structure/STRUCTURE_CACHE_MANIFEST.json')
    a = structure['authority']
    require(a['geometry_source'] == 'independent_sequence_predicted_af3'
            and a['experimental_bound_complex_coordinates_read'] is False
            and a['cross_partner_coordinates_or_distances_used'] is False,
            'Wrong geometry provenance')
    require({r['instance_id'] for r in structure['records']} == set(test),
            'Structure/test IDs differ')
    require(bundle['routing'] == {'contact': 'baseline', 'paratope': 'baseline',
                                  'epitope': 'edge_topk2_supervised'}, 'Task routing differs')
    require(len(bundle['components']) == 6, 'Missing checkpoint component')
    require({(c['arm'], c['seed']) for c in bundle['components']} ==
            {(a, s) for a in ('baseline', 'edge_topk2_supervised') for s in (17, 29, 43)},
            'Duplicate or unexpected component')
    return bundle
