"""Standard-library evidence extraction; no training or network access."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def descend(value, path):
    if not path:
        return value
    key, tail = path[0], path[1:]
    if key == '*':
        if not isinstance(value, list):
            raise ValueError('wildcard requires an array')
        return [descend(item, tail) for item in value]
    if isinstance(key, dict):
        fields = key['where']
        rows = [row for row in value if all(row.get(k) == v for k, v in fields.items())]
        if len(rows) != 1:
            raise ValueError('selector must identify exactly one row, found %d' % len(rows))
        return descend(rows[0], tail)
    return descend(value[key], tail)

def extract(record, root=ROOT):
    source = Path(root) / record['source']
    if sha256(source) != record['source_sha256']:
        raise ValueError('source changed: ' + record['source'])
    data = json.loads(source.read_text())
    if 'selectors' in record:
        result = [descend(data, path) for path in record['selectors']]
    else:
        result = descend(data, record['selector'])
    if record.get('reduce') == 'mean':
        if not result:
            raise ValueError('empty mean')
        result = statistics.fmean(result)
    elif record.get('reduce') not in (None, 'identity'):
        raise ValueError('unsupported reduction')
    return result

def equivalent(left, right):
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    if isinstance(left, (float,int)) and not isinstance(left,bool):
        return isinstance(right, (float,int)) and math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left == right

def load_registry(root=ROOT):
    data = json.loads((Path(root)/'results/HEADLINE_REGISTRY.json').read_text())
    values = {}
    for record in data['metrics']:
        if record['id'] in values:
            raise ValueError('duplicate metric ID: '+record['id'])
        if not all(record.get(k) for k in ['cohort','evidence_class','aggregation','interpretation_limit']):
            raise ValueError('incomplete evidence boundary: '+record['id'])
        value = extract(record,root)
        if not equivalent(value,record['value']):
            raise ValueError('metric drift: '+record['id'])
        values[record['id']] = value
    return data, values
