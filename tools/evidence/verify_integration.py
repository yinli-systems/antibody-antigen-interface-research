#!/usr/bin/env python3
"""Verify complete branch-history integration and exact preserved variants.

No model calls or scientific reruns. Remote checking is read-only and explicit.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'results/integration_20261007'

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])

def ancestor(sha,head='HEAD'):
    return subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',sha,head],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0

def verify(remote=False, receipt_target=None):
    refs=json.loads((EVIDENCE/'SOURCE_BRANCHES.json').read_text())
    variants=json.loads((EVIDENCE/'PRESERVED_VARIANTS.json').read_text())
    manifest=json.loads((EVIDENCE/'TREE_MANIFEST.json').read_text())
    packed=(EVIDENCE/'PREMERGE_TREES.jsonl.gz').read_bytes()
    if hashlib.sha256(packed).hexdigest()!=manifest['compressed_sha256']:
        raise ValueError('Premerge inventory digest drift')
    rows=[json.loads(x) for x in gzip.decompress(packed).splitlines()]
    if len(rows)!=manifest['tracked_entries']:
        raise ValueError('Premerge inventory count drift')
    grouped=defaultdict(list)
    for row in rows:grouped[row['commit']].append(row)
    for b in refs['branches']:
        if not ancestor(b['source_sha']):raise ValueError('Unmerged source HEAD: '+b['current_name'])
        tree=[]
        for raw in git('ls-tree','-r','-l','-z',b['source_sha']).split(b'\0'):
            if not raw:continue
            metadata,path=raw.split(b'\t',1)
            mode,kind,oid,size=metadata.decode().split()
            tree.append(dict(branch=b['current_name'],commit=b['source_sha'],path=path.decode(),mode=mode,kind=kind,blob=oid,bytes=None if size=='-' else int(size)))
        if tree!=grouped[b['source_sha']]:raise ValueError('Premerge tree changed: '+b['current_name'])
    for v in variants['files']:
        data=(ROOT/v['preserved_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=v['sha256']:
            raise ValueError('Preserved variant digest drift: '+v['preserved_path'])
        if git('rev-parse',v['commit']+':'+v['source_path']).decode().strip()!=v['git_blob']:
            raise ValueError('Variant source blob mismatch')
        if git('show',v['commit']+':'+v['source_path'])!=data:
            raise ValueError('Variant bytes differ from original source')
    # The independent contract implementation is not synthesized or rewritten.
    contract='extensions/strict_partner_retrieval_v1/contract_models.py'
    original='fdc2db39bd9f8817ae414a17dcf666a565433d83:extensions/strict_partner_retrieval_v1/models.py'
    if (ROOT/contract).read_bytes()!=git('show',original):raise ValueError('Contract namespace changed algorithm bytes')
    test='tests/test_strict_partner_retrieval_models.py'
    original_test=git('show','fdc2db39bd9f8817ae414a17dcf666a565433d83:'+test).decode()
    expected=original_test.replace('from strict_partner_retrieval_v1.models import (','from strict_partner_retrieval_v1.contract_models import (')
    if (ROOT/test).read_text()!=expected:raise ValueError('Contract tests changed beyond import namespace')
    namespace=json.loads((EVIDENCE/'NAMESPACE_INTEGRATION.json').read_text())
    for item in namespace:
        original_bytes=git('show',item['source_commit']+':'+item['source_path'])
        actual_bytes=(ROOT/item['integrated_path']).read_bytes()
        if item['integrated_path'].endswith('.py') and item['integrated_path'].startswith('src/'):
            if actual_bytes!=original_bytes:raise ValueError('Optional namespace algorithm bytes changed')
        else:
            expected_text=original_bytes.decode().replace('from antibody_pretraining.pair_aware_attention import','from antibody_pretraining.multimodal_v1.pair_aware_attention import').replace('from antibody_pretraining.pair_aware_esm2 import','from antibody_pretraining.multimodal_v1.pair_aware_esm2 import')
            if actual_bytes.decode()!=expected_text:raise ValueError('Optional namespace test changed beyond imports')
    for filename in ('test_pair_aware_attention.py','test_pair_aware_esm2.py'):
        if (ROOT/'tests'/filename).read_bytes()!=git('show','6e80c6d44be531edd6f40404521ba4f087a81fd4:tests/'+filename):raise ValueError('Frozen core test bytes changed')
    status=json.loads((EVIDENCE/'STATUS.json').read_text())
    for record in status['bindings']:
        data=(ROOT/record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Status source drift')
        v=json.loads(data)
        for k in record['selector']:v=v[k]
        if v!=record['value']:raise ValueError('Status field drift')
    checked_python=0
    for raw in git('ls-files','-z','*.py').split(b'\0'):
        if not raw:continue
        relative=raw.decode()
        if relative.startswith('archive/'):continue
        p=ROOT/relative
        if not p.exists():continue
        text=p.read_text()
        if re.search(r'^(<<<<<<< |>>>>>>> )',text,re.M):raise ValueError('Unresolved conflict marker: '+relative)
        ast.parse(text,filename=relative)
        checked_python+=1
    for relative in ['README.md','RESULTS.md','results/INDEX.md','results/operations/README.md','results/integration_20261007/BRANCH_PROGRESS.md','results/integration_20261007/VALIDATION.md']:
        p=ROOT/relative
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','mailto:','#')):continue
            target=unquote(link.split('#',1)[0])
            if target and not (p.parent/target).exists():
                pending_receipt = receipt_target is not None and (p.parent/target).resolve() == receipt_target.resolve()
                if not pending_receipt:raise ValueError('Broken navigation: '+relative+' -> '+target)
    result=dict(status='PASS_ALL_BRANCH_INTEGRATION',starting_branch_heads=len(refs['branches']),heads_verified_as_ancestors=len(refs['branches']),preserved_path_decisions=variants['path_decisions'],preserved_variant_entries=len(variants['files']),premerge_tree_entries=len(rows),canonical_python_files_parsed=checked_python,contract_model_bytes_unchanged=True,contract_tests_only_namespace_changed=True,optional_multimodal_source_modules_byte_identical=3,optional_multimodal_tests_only_namespace_changed=True,completion_metric_bindings_checked=len(status['bindings']),scope='Git history, archived source bytes, imports/syntax and reporting links. Not scientific replication or live job status.')
    if remote:
        remote_heads=[]
        for line in git('ls-remote','--heads','origin').decode().splitlines():
            sha,ref=line.split();name=ref.removeprefix('refs/heads/')
            if 'review' in name.lower():raise ValueError('Review-related active branch name remains: '+name)
            if not ancestor(sha):raise ValueError('Remote HEAD not included in local integrated history: '+name)
            remote_heads.append(dict(name=name,sha=sha))
        result['remote_heads_checked']=remote_heads
        result['active_review_named_branches']=0
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--remote',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
    result=verify(a.remote,a.receipt)
    text=json.dumps(result,indent=2)+'\n'
    if a.receipt:
        with a.receipt.open('x') as f:f.write(text)
    print(text,end='')
if __name__=='__main__':main()
