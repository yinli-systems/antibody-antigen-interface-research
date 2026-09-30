#!/usr/bin/env python3
"""Verify publication numbers, generated pages and preserved Git snapshots.

This does NOT rerun models, validate biological conclusions or replace a
historical experiment's test suite. Network access is optional and explicit.
"""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote
from core import ROOT, load_registry, sha256
from render import products


def local_links(path, root=ROOT):
    broken = []
    text = Path(path).read_text()
    for target in re.findall(r'\]\(([^)]+)\)',text):
        target = target.strip().split(' "')[0]
        if target.startswith(('https://','http://','mailto:','#')):
            continue
        target = unquote(target.split('#',1)[0])
        if not target:
            continue
        resolved = (Path(root)/target.lstrip('/')) if target.startswith('/') else Path(path).parent/target
        if not resolved.exists():
            broken.append(target)
    return broken


def verify_history(root=ROOT, remote=False):
    root = Path(root)
    m = json.loads((root/'archive/FULL_SNAPSHOT_MANIFEST_20260930.json').read_text())
    compressed = root/m['compressed_manifest']
    if sha256(compressed) != m['compressed_manifest_sha256']:
        raise ValueError('full snapshot inventory digest mismatch')
    entries = [json.loads(row) for row in gzip.decompress(compressed.read_bytes()).splitlines()]
    if len(entries) != m['file_entries_including_repetition']:
        raise ValueError('full tree entry count mismatch')
    groups = defaultdict(list)
    for row in entries:
        groups[row['snapshot']].append(row)
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args])
    remote_tags = {}
    if remote:
        for line in git('ls-remote','--tags','origin','archive/2026-09-30/*').decode().splitlines():
            oid, ref = line.split(); remote_tags[ref] = oid
    for snapshot in m['snapshots']:
        tag = snapshot['archive_tag']; expected = snapshot['head_sha']
        resolved = git('rev-parse','refs/tags/'+tag+'^{commit}').decode().strip()
        if resolved != expected:
            raise ValueError('archived tag changed: '+tag)
        raw = git('ls-tree','-r','-l','-z',expected)
        if hashlib.sha256(raw).hexdigest() != snapshot['ls_tree_sha256']:
            raise ValueError('tree mismatch: '+tag)
        actual = []
        for entry in raw.split(b'\0'):
            if not entry: continue
            meta, name = entry.split(b'\t',1)
            mode, kind, oid, size = meta.decode().split()
            actual.append({'snapshot':tag,'mode':mode,'type':kind,'git_oid':oid,'bytes':None if size=='-' else int(size),'path':name.decode()})
        if actual != groups[tag]:
            raise ValueError('archived per-file inventory mismatch: '+tag)
        if remote and remote_tags.get('refs/tags/'+tag) != expected:
            raise ValueError('remote archive tag missing or changed: '+tag)
    return {'snapshots':len(m['snapshots']),'tracked_entries':len(entries),'distinct_path_blob_versions':m['distinct_path_blob_versions'],'remote_tags_checked':remote}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',action='store_true')
    parser.add_argument('--remote',action='store_true',help='also query origin for archived tags (network)')
    parser.add_argument('--receipt',type=Path,help='write a new JSON verification receipt; never overwrite')
    args=parser.parse_args()
    registry,values=load_registry()
    generated=products()
    for relative,text in generated.items():
        if not (ROOT/relative).exists() or (ROOT/relative).read_text()!=text:
            raise ValueError('generated file drift: '+relative)
    pages=['README.md','docs/REPRODUCE_EXACT.md','docs/CLAIM_BOUNDARIES.md','docs/PRESENTATION_RESEARCH.md','archive/README.md','results/ALL_BRANCHES.md','results/operations/README.md']
    for relative in pages:
        errors=local_links(ROOT/relative)
        if errors: raise ValueError('broken links in '+relative+': '+str(errors))
    evidence_index=json.loads((ROOT/'results/BRANCH_EVIDENCE_INDEX.json').read_text())
    paths=sum(len(b['evidence_files']) for b in evidence_index['branches'])
    if paths!=747:
        raise ValueError('original evidence index count changed')
    # Guard the most damaging previous claim inflation / conflation.
    if not (values['rep_random_ci'][0] < 0 < values['rep_random_ci'][1]):
        raise ValueError('random-control undecided interpretation no longer supported')
    if not (values['holdout_ESM-IF1_ci'][0] < 0 < values['holdout_ESM-IF1_ci'][1]):
        raise ValueError('ESM-IF1 non-confirmation no longer matches artifact')
    if values['holdout_ProteinMPNN_ci'][0] <= 0:
        raise ValueError('ProteinMPNN direction no longer matches artifact')
    if values['reparank_macro_ap'] == values['reparank_pooled_ap']:
        raise ValueError('macro and pooled metrics collapsed')
    receipt={'status':'PASS_PUBLICATION_INTEGRITY','metric_checks':len(values),'unique_bound_sources':len({r['source'] for r in registry['metrics']}),'generated_files_checked':list(generated),'pages_link_checked':pages,'evidence_paths':paths,'scope':'reporting/artifact integrity only; NOT complete scientific validation or a rerun of GPU experiments'}
    if args.history or args.remote:
        receipt['history']=verify_history(remote=args.remote)
    if args.receipt:
        with args.receipt.open('x') as f:
            json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
