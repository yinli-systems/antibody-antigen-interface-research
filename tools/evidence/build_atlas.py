#!/usr/bin/env python3
"""Build a commit-pinned results atlas. Standard library; no network/model/data access.
The snapshot covers branch HEAD trees, not every historical commit.
"""
from __future__ import annotations
import argparse, gzip, hashlib, html, json, re, subprocess
from collections import Counter, defaultdict
from pathlib import Path
from core import load_registry
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2]
ATLAS=ROOT/'results'/'atlas'
URL='https://github.com/yinli-systems/antibody-antigen-interface-research'
EXTS={'.json','.jsonl','.csv','.tsv','.md','.txt','.out','.err','.log','.svg','.png','.pdf'}
PREFIXES=('results/','extensions/','paper/','framecontact/evaluation/','framecontact/receipts/','framecontact/reports/','abcite/evals/','docs/evaluations/')
LEVELS={'metric-bound':'Committed numeric fields checked; no model rerun','readout-bound':'Committed research readout inspected; not independently recomputed','report-only':'Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees','protocol-only':'Protocol or implementation, not completed performance','engineering':'Execution/provenance evidence, not model accuracy','historical':'Earlier snapshot or superseded interpretation, retained unchanged'}
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def dumps(x,pretty=True):return json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2 if pretty else None,separators=None if pretty else (',',':'),allow_nan=False)+'\n'
def link(sha,path):
    if not re.fullmatch(r'[0-9a-f]{40}',sha):raise ValueError('Full immutable commit required')
    return f'{URL}/blob/{sha}/{quote(path,safe="/")}'
def select(x,selector):
    for p in selector:
        if isinstance(p,dict):
            if set(p)!={'where'} or not isinstance(x,list):raise ValueError('Invalid named-row selector')
            match=[r for r in x if isinstance(r,dict) and all(r.get(k)==v for k,v in p['where'].items())]
            if len(match)!=1:raise ValueError(f'Expected exactly one row: {p}, found {len(match)}')
            x=match[0]
        else:x=x[p]
    return x
def require_value(actual,expected):
    if isinstance(actual,(int,float)) and not isinstance(actual,bool) and isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if abs(actual-expected)>1e-12*max(1,abs(expected)):raise ValueError(f'Metric drift: {actual} != {expected}')
    elif actual!=expected:raise ValueError(f'Metric drift: {actual!r} != {expected!r}')
def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
def family(p):
    parts=p.split('/')
    if p.startswith('results/pretraining/'):return '/'.join(parts[:3])
    if p.startswith(('framecontact/evaluation/','framecontact/receipts/','framecontact/reports/','abcite/evals/')):return '/'.join(parts[:2])
    return '/'.join(parts[:2]) if len(parts)>2 else parts[0]
def evidence(p):return p.startswith(PREFIXES) and Path(p).suffix.lower() in EXTS

def snapshot(refs):
    versions={};trees={}
    for i,b in enumerate(refs['branches']):
        if git('rev-parse',b['sha']+'^{commit}').decode().strip()!=b['sha']:raise ValueError('Missing pinned commit')
        tree={}
        for raw in git('ls-tree','-r','-l','-z',b['sha']).split(b'\0'):
            if not raw:continue
            meta,p=raw.split(b'\t',1);mode,kind,oid,size=meta.decode().split();p=p.decode();tree[p]=oid
            r=versions.setdefault((p,oid),dict(path=p,blob=oid,bytes=None if size=='-' else int(size),type=kind,branches=[],modes={}))
            r['branches'].append(i);r['modes'][str(i)]=mode
        if len(tree)!=b['files']:raise ValueError('Branch file count drift')
        trees[b['name']]=tree
    rows=[versions[k] for k in sorted(versions)];blobs={}
    process=subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for oid in sorted({r['blob'] for r in rows if r['type']=='blob'}):
            process.stdin.write((oid+'\n').encode());process.stdin.flush();head=process.stdout.readline().split()
            if len(head)!=3 or head[1]!=b'blob':raise ValueError('Unreadable blob '+oid)
            size=int(head[2]);data=process.stdout.read(size)
            if len(data)!=size or process.stdout.read(1)!=b'\n':raise ValueError('Truncated blob')
            if hashlib.sha1(f'blob {size}\0'.encode()+data).hexdigest()!=oid:raise ValueError('Git digest mismatch')
            blobs[oid]=data
    finally:process.stdin.close();process.wait()
    for r in rows:
        if r['type']=='blob':
            data=blobs[r['blob']]
            if len(data)!=r['bytes']:raise ValueError('Blob size mismatch')
            r['sha256']=hashlib.sha256(data).hexdigest()
        r['evidence_candidate']=evidence(r['path']);r['family']=family(r['path'])
    actual=dict(branch_count=len(trees),tracked_entries=sum(map(len,trees.values())),distinct_path_blob_versions=len(rows),distinct_blobs=len(blobs))
    if actual!=refs['counts']:raise ValueError(f'Count drift: {actual}')
    return rows,trees,blobs

def products():
    refs=json.loads((ATLAS/'SOURCE_REFS.json').read_text());cur=json.loads((ATLAS/'CURATED.json').read_text())
    rows,trees,blobs=snapshot(refs);branches=refs['branches'];byname={b['name']:b for b in branches};main=trees['main'];bykey={(r['path'],r['blob']):r for r in rows};out={}
    def put(p,s):out['results/atlas/'+p]=s.encode() if isinstance(s,str) else s
    def source(b,p):
        if b not in trees or p not in trees[b]:raise ValueError(f'Missing curated source {b}:{p}')
        oid=trees[b][p];r=bykey[p,oid]
        return dict(branch=b,commit=byname[b]['sha'],path=p,blob=oid,sha256=r['sha256'],url=link(byname[b]['sha'],p))
    def raw(s):return blobs[s['blob']]
    registry_source=source('main','results/HEADLINE_REGISTRY.json');registry=json.loads(raw(registry_source));metrics=[]
    for record in registry['metrics']:
        s=source('main',record['source'])
        if s['sha256']!=record['source_sha256']:raise ValueError('Registry source digest mismatch '+record['id'])
        require_value(load_registry(ROOT)[1][record['id']],record['value'])
        metrics.append({**record,'source_binding':s,'verification':'Exact JSON field and source digest; no experiment rerun'})
    ids={r['id'] for r in metrics};cards=[];extra=[]
    for original in cur['cards']:
        c=dict(original)
        if c['level'] not in LEVELS:raise ValueError('Unknown evidence level')
        c['sources']=[source(s['branch'],s['path']) for s in original['sources']];c['checks']=[]
        for check in original.get('checks',[]):
            s=c['sources'][check['source_index']];v=select(json.loads(raw(s)),check['selector']);require_value(v,check['expected'])
            bound={**check,'actual':v,'source_binding':s};extra.append({'card':c['id'],**bound});c['checks'].append(bound)
        if c['level']=='metric-bound' and not (c['checks'] or c.get('legacy_metric_ids')):raise ValueError('Unbound metric card')
        for ident in c.get('legacy_metric_ids',[]):
            if ident not in ids:raise ValueError('Unknown legacy metric '+ident)
        if c['level']=='report-only' and c.get('legacy_metric_ids'):raise ValueError('Narrative claim cannot inherit old measurements')
        cards.append(c)
    json_count=0;errors=[]
    for r in rows:
        if r['type']=='blob' and r['path'].endswith('.json'):
            json_count+=1
            try:json.loads(blobs[r['blob']].decode(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite '+x)))
            except (ValueError,UnicodeDecodeError) as ex:errors.append(dict(path=r['path'],blob=r['blob'],error=str(ex)))
    packed=gzip.compress(b''.join(dumps(r,False).encode() for r in rows),mtime=0)
    # CPython 3.11/3.12 may inherit zlib's OS byte; canonicalize to RFC gzip unknown.
    packed=packed[:9]+b'\xff'+packed[10:]
    put('FILE_VERSIONS.jsonl.gz',packed)
    put('METRIC_BINDINGS.json',dumps(dict(scope='Reporting-value integrity, not scientific replication',legacy_registry_source=registry_source,legacy_metrics=metrics,additional_checks=extra)))
    put('RESULT_CARDS.json',dumps(dict(snapshot_utc=refs['snapshot_utc'],levels=LEVELS,cards=cards)))
    index=['# All 30 branch HEADs — frozen 7 October 2026','','[Results atlas](README.md) · [Cohorts](COHORTS.md) · [Full file inventory](FILE_VERSIONS.jsonl.gz)','','One branch is not one independent experiment. Ahead/behind uses the original main SHA. Shared files are not replications, and a file shared with another side branch is not necessarily on main.','','| Branch | Role / evidence state | HEAD | Different file versions vs main | Detail |','|---|---|---|---:|---|']
    for i,b in enumerate(branches):
        ann=cur['branches'][b['name']];tree=trees[b['name']];different=sorted((p,o) for p,o in tree.items() if main.get(p)!=o);missing=set(main)-set(tree);pagepath=f'branches/{i:02d}.md'
        index.append(f"| `{b['name']}` | {cell(ann['role'])}; {cell(ann['state'])} | [`{b['sha'][:8]}`]({URL}/tree/{b['sha']}) | {len(different)} | [Inspect]({pagepath}) |")
        page=[f"# {b['name']}",'','[All branches](../BRANCHES.md) · [Atlas](../README.md)','',f"**Role:** {ann['role']}  ",f"**State:** {ann['state']}  ",f"**Pinned HEAD:** `{b['sha']}`  ",f"**Commit time:** {b['committed_at']}  ",f"**Ahead / behind original main:** {b['ahead']} / {b['behind']}  ",f"**Tracked files:** {b['files']}  ",f"**Path/blob versions unique to this one branch HEAD:** {b['distinct_versions_only_this_branch']}",'',ann['interpretation'],'','## Read first','']
        for p in ann.get('read_first',[]):page.append(f'- [{p}]({source(b["name"],p)["url"]})')
        linked=[c for c in cards if any(s['branch']==b['name'] for s in c['sources'])]
        if linked:page+=['','## Result cards','']+[f'- [{c["title"]}](../cards/{c["id"]}.md) — {c["level"]}' for c in linked]
        page+=['','## Exact differences from original main','',f'{len(different)} file versions below are absent from or differ from original main. {len(missing)} main paths are absent from this branch (often later curated material, not proof of intentional deletion). Unchanged inherited files remain in the full explorer.','','| Path | Relationship | Evidence candidate | Git blob |','|---|---|---|---|']
        for p,oid in different:page.append(f'| [{cell(p)}]({link(b["sha"],p)}) | {"different bytes" if p in main else "not on main"} | {"candidate, not validation" if evidence(p) else "source/support"} | `{oid}` |')
        if not different:page.append('| No differing versions | Same/ancestral snapshot | — | — |')
        put(pagepath,'\n'.join(page)+'\n')
    put('BRANCHES.md','\n'.join(index)+'\n')
    card_rows=[]
    for c in cards:
        card_rows.append(f'| [{cell(c["title"])}](cards/{c["id"]}.md) | {c["level"]} | {cell(c["cohort"])} | {cell(c["answer"])} |')
        page=[f'# {c["title"]}','','[Atlas](../README.md) · [Cohorts](../COHORTS.md)','',f'**Evidence level:** `{c["level"]}` — {LEVELS[c["level"]]}','',c['answer'],'','| Field | Value |','|---|---|']
        for k in ['workstream','cohort','inputs','seeds','estimand']:page.append(f'| {k} | {cell(c[k])} |')
        page+=['','## What this does not establish','',c['boundary'],'','## Pinned evidence','']
        for j,s in enumerate(c['sources']):page +=[f'{j+1}. [{s["path"]}]({s["url"]})',f'   - Branch `{s["branch"]}`; commit `{s["commit"]}`.',f'   - Git blob `{s["blob"]}`; SHA-256 `{s["sha256"]}`.']
        if c.get('legacy_metric_ids'):
            page+=['','## Exact checked registry values','','| ID | Value | Estimator |','|---|---|---|']
            for ident in c['legacy_metric_ids']:
                m=next(m for m in metrics if m['id']==ident);page.append(f'| `{ident}` | `{json.dumps(m["value"])}` | {cell(m["aggregation"])} |')
        if c['checks']:
            page+=['','## Exact checked JSON fields','','| Selector | Value |','|---|---|']
            for check in c['checks']:page.append(f'| `{cell(json.dumps(check["selector"]))}` | `{json.dumps(check["actual"])}` |')
        page+=['','**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.','']
        put(f'cards/{c["id"]}.md','\n'.join(page))
    erows=[r for r in rows if r['evidence_candidate']];groups=defaultdict(list)
    for r in erows:groups[r['family']].append(r)
    fi=['# Complete evidence-location catalogue','','[Atlas](README.md) · [All branches](BRANCHES.md)','','Exhaustive for the stated filter, not a list of scientifically validated experiments. Every other tracked path is still in the full file inventory. A newer branch date does not automatically supersede a frozen result.','',f'Roots: `{", ".join(PREFIXES)}`. Extensions: `{", ".join(sorted(EXTS))}`.','','| Location | Distinct path/blob versions | All source versions |','|---|---:|---|']
    for i,(fam,items) in enumerate(sorted(groups.items())):
        fi.append(f'| `{fam}` | {len(items)} | [Inspect](families/{i:03d}.md) |')
        page=[f'# {fam}','','[Catalogue](../FAMILIES.md) · [Atlas](../README.md)','','Automatically catalogued: not all manually adjudicated or scientifically validated. Protocols, configs, logs, receipts and measurements are not interchangeable.','','| File | Same bytes on main? | Branch memberships | Blob |','|---|---|---|---|']
        for r in items:
            bid=next((j for j in r['branches'] if branches[j]['name']=='main'),r['branches'][0]);members=', '.join(f'[{j:02d}](../branches/{j:02d}.md)' for j in r['branches'])
            page.append(f'| [{cell(r["path"])}]({link(branches[bid]["sha"],r["path"])}) | {"yes" if main.get(r["path"])==r["blob"] else "no / different"} | {members} | `{r["blob"]}` |')
        put(f'families/{i:03d}.md','\n'.join(page)+'\n')
    put('FAMILIES.md','\n'.join(fi)+'\n')
    for key,name in [('cohorts_md','COHORTS.md'),('findings_md','REVIEW_FINDINGS.md'),('presentation_md','PRESENTATION_GUIDE.md')]:put(name,cur[key])
    intro=f'''# Antibody–antigen results atlas

**A result is a claim plus its cohort, estimator and source — not just a score.**

[All 30 branches](BRANCHES.md) · [Every evidence location](FAMILIES.md) · [Cohorts & comparability](COHORTS.md) · [Audit findings](REVIEW_FINDINGS.md) · [Presentation guide](PRESENTATION_GUIDE.md)

Snapshot: **{refs['snapshot_utc']}**. Original main: `{refs['main_sha']}`. **30 branch HEADs · {len(rows):,} distinct path/blob versions · {refs['counts']['tracked_entries']:,} tracked entries including repetition.** This is not every historical commit or an external GPU filesystem.

## Start here

The strongest artifact-backed story is controlled correspondence / representation evidence, a bounded three-output interface predictor, and explicit negative controls. It is not a single cross-protocol model leaderboard. AbCite is a separate multimodal evidence-retrieval application.

**Completion corrections:** final structural-PLM receipts record36/36 and no promotion; centered RePaRank structure covers217/217. Earlier12/36 / pending records are historical snapshots, not today's live status. See [PLMs](cards/structural-plm.md), [RePaRank](cards/reparank-centered.md), and [findings](REVIEW_FINDINGS.md).

**Newer does not mean better verified.** The October4 reviewer manuscript reports additional campaigns and an improved ensemble. These claims are shown as `report-only` where matching terminal artifacts were not identified in the inspected branch HEADs. This does not mean the runs did not happen or their files do not exist off-Git.

## Evidence levels

| Level | Meaning |
|---|---|
'''+ '\n'.join(f'| `{k}` | {v} |' for k,v in LEVELS.items())+'\n\n## Result-by-result navigation\n\n| Result | Level | Cohort | Recorded answer |\n|---|---|---|---|\n'+'\n'.join(card_rows)+f'''

## Beyond the curated cards

[{len(groups)} evidence-location pages](FAMILIES.md) cover **{len(erows):,} candidate evidence-file versions**, including smaller studies, failure logs and protocols. The [{len(rows):,}-version inventory](FILE_VERSIONS.jsonl.gz) contains every tracked path from all30 frozen HEADs, including code/assets. Neither number counts independent experiments.

The self-contained [HTML explorer](index.html) provides text, branch, file-kind, workstream and evidence-level filters. GitHub displays HTML source; open a saved copy in a browser for the interactive view. No external scripts, uploads, telemetry, model execution or local server. Markdown works directly on GitHub.

## Verify the presentation

```bash
python3 tools/evidence/build_atlas.py --check
python3 tools/evidence/verify.py --history
```

The clone must contain the pinned commits; no automatic fetch. [Verification receipt](VERIFICATION.json) · [Numeric bindings](METRIC_BINDINGS.json) · [Frozen refs](SOURCE_REFS.json) · [Manual definitions](CURATED.json).

**This verifies reporting integrity.** Source bytes, values, trees and reproducible pages are checked. Training, inference, unpublished cluster artifacts, label reconstruction and biological conclusions were not independently rerun. The old17-snapshot /747-link archive remains intact and is not relabelled as the new30-branch snapshot.
'''
    put('README.md',intro)
    payload=dict(branches=[dict(name=b['name'],sha=b['sha']) for b in branches],rows=[[r['path'],r['blob'],r['bytes'],r['branches'],r['evidence_candidate'],r['family'],main.get(r['path'])==r['blob']] for r in rows],cards=cards,levels=LEVELS,counts=refs['counts'],snapshot=refs['snapshot_utc'])
    safe=dumps(payload,False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026');put('index.html',HTML.replace('__DATA__',safe))
    receipt=dict(status='PASS_SNAPSHOT_AND_REPORTING_INTEGRITY',snapshot_utc=refs['snapshot_utc'],**refs['counts'],git_blob_digests_checked=len(blobs),json_file_versions_syntax_checked=json_count,json_syntax_errors=errors,candidate_evidence_file_versions=len(erows),evidence_location_pages=len(groups),curated_cards=len(cards),card_levels=dict(Counter(c['level'] for c in cards)),exact_metric_field_checks=len(metrics)+len(extra),legacy_metric_checks=len(metrics),additional_metric_checks=len(extra),file_inventory_sha256=hashlib.sha256(packed).hexdigest(),source_refs_sha256=hashlib.sha256((ATLAS/'SOURCE_REFS.json').read_bytes()).hexdigest(),curation_sha256=hashlib.sha256((ATLAS/'CURATED.json').read_bytes()).hexdigest(),scope='All30 frozen branch HEAD trees mechanically inventoried; selected readouts manually assessed. Numeric source-field checks, not independent scientific replication. No inference, training, dataset access or scheduler queries.',not_checked=['Every historical commit','Live GPU/scheduler state','Off-Git artifacts','Every code line or biological claim','External publication validity','Independent recalculation from original predictions'],generated_files_sha256={p:hashlib.sha256(b).hexdigest() for p,b in sorted(out.items())})
    put('VERIFICATION.json',dumps(receipt));return out

HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Antibody–antigen · Results atlas</title><style>
:root{font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#181818;background:#fafafa}*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1280px;margin:auto;padding:28px}header{padding-top:46px;border-bottom:1px solid #ccc}h1{font-size:38px;line-height:1.12;letter-spacing:-1.2px;margin:10px 0 20px}h2{font-size:25px}h3{font-size:19px;margin:0 0 12px}a{color:inherit;text-underline-offset:3px}p{max-width:950px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase}.muted{color:#555}.stats{display:flex;gap:28px;flex-wrap:wrap;margin:24px 0}.stats strong{font-size:29px;display:block}.stats span{font-size:13px}.notice{border-left:4px solid;padding:12px 20px;background:#eee;margin:24px 0}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}input,select,button{font:inherit;padding:10px 12px;border:1px solid #999;border-radius:5px;background:white;color:inherit}input{min-width:270px;flex:1}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}label{display:flex;align-items:center;gap:8px;font-size:14px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}.card{border:1px solid #c7c7c7;background:white;border-radius:8px;padding:22px}.tag{display:inline-block;font-size:11px;letter-spacing:.7px;text-transform:uppercase;border:1px solid;padding:3px 7px;margin-bottom:12px}.meta{font-size:13px;margin:12px 0}.boundary{font-size:13px;border-top:1px solid #ddd;padding-top:13px}details{margin-top:14px;font-size:13px}summary{cursor:pointer;font-weight:600}code{font-size:.88em;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;border-bottom:2px solid #333;padding:12px 8px}td{padding:11px 8px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}td:first-child{max-width:650px}.pager{display:flex;align-items:center;gap:16px;margin-top:18px}.tablewrap{overflow:auto}footer{font-size:13px;border-top:1px solid #ccc;margin-top:25px}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:600px){header,main,footer{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:30px}}@media print{.filters,.pager{display:none}.cards{display:block}.card{break-inside:avoid;margin-bottom:15px}body{background:white}}
</style></head><body><header><div class="eyebrow">Research evidence / 7 October 2026</div><h1>Antibody–antigen<br>results atlas</h1><p>Read the result. Check the cohort. Follow the evidence. Every branch HEAD, with completed measurements separated from narrative claims and unexecuted protocols.</p><nav><a href="#results">Result cards</a><a href="#files">Every file version</a><a href="README.md">GitHub-readable atlas</a><a href="COHORTS.md">Comparability</a><a href="REVIEW_FINDINGS.md">Audit findings</a></nav><div class="stats" id="stats"></div><div class="notice"><strong>Important boundary.</strong> Verified here means committed bytes and reporting fields were checked—not that experiments or biological conclusions were independently reproduced. Newer claims without matching terminal evidence remain report-only.</div></header><main><section id="results"><h2>Result by result</h2><div class="filters"><input id="cq" aria-label="Search result cards" placeholder="Search results, cohorts, caveats…"><select id="level" aria-label="Evidence level"><option value="">Every evidence level</option></select><select id="work" aria-label="Workstream"><option value="">Every workstream</option></select></div><p class="muted" id="cn"></p><div class="cards" id="cards"></div></section><section id="files"><h2 style="margin-top:55px">Every tracked file version</h2><p class="muted">Identical path/blob versions are deduplicated. Branch count is not replicate count. Evidence candidate is a location filter, not a scientific verdict.</p><div class="filters"><input id="fq" aria-label="Search paths" placeholder="Search paths, family or blob ID…"><select id="branch" aria-label="Branch"><option value="">All30 frozen branches</option></select><select id="kind" aria-label="File extension"><option value="">Every file kind</option><option>.md</option><option>.json</option><option>.csv</option><option>.py</option><option>.svg</option><option>.pdf</option></select><label><input id="ev" type="checkbox" style="min-width:0;flex:none">Evidence only</label><label><input id="diff" type="checkbox" style="min-width:0;flex:none">Different from main</label></div><p id="fn" class="muted"></p><div class="tablewrap"><table><thead><tr><th>Path / source</th><th>Kind</th><th>Size</th><th>On main?</th><th>Branches</th></tr></thead><tbody id="rows"></tbody></table></div><div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div></section></main><footer>Frozen snapshot, not a live monitor. No external scripts, uploads, telemetry or model calls. Source links require access to the private repository.</footer><script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('stats').innerHTML=Object.entries({'branch HEADs':D.counts.branch_count,'file versions':D.counts.distinct_path_blob_versions,'curated result cards':D.cards.length}).map(([s,n])=>`<div><strong>${n.toLocaleString()}</strong><span>${E(s)}</span></div>`).join('');for(const k of Object.keys(D.levels))$('level').add(new Option(k,k));for(const w of [...new Set(D.cards.map(x=>x.workstream))].sort())$('work').add(new Option(w,w));D.branches.forEach((b,i)=>$('branch').add(new Option(b.name,String(i))));
function cards(){const q=$('cq').value.toLowerCase(),l=$('level').value,w=$('work').value;const cc=D.cards.filter(c=>(!l||c.level===l)&&(!w||c.workstream===w)&&JSON.stringify(c).toLowerCase().includes(q));$('cn').textContent=`${cc.length} of ${D.cards.length} cards · each retains its own protocol`;$('cards').innerHTML=cc.map(c=>`<article class="card"><div class="tag">${E(c.level)}</div><h3>${E(c.title)}</h3><p>${E(c.answer)}</p><div class="meta"><strong>Cohort:</strong> ${E(c.cohort)}<br><strong>Seeds:</strong> ${E(c.seeds)}<br><strong>Estimator:</strong> ${E(c.estimand)}</div><p class="boundary"><strong>Limit:</strong> ${E(c.boundary)}</p><details><summary>Exact source evidence (${c.sources.length})</summary>${c.sources.map(s=>`<p><a href="${E(s.url)}" target="_blank" rel="noopener">${E(s.path)}</a><br><code>${E(s.commit)}</code></p>`).join('')}</details></article>`).join('');}
let page=0,filtered=[];const pageSize=50;function files(reset=true){if(reset)page=0;const q=$('fq').value.toLowerCase(),b=$('branch').value,k=$('kind').value,ev=$('ev').checked,diff=$('diff').checked;filtered=D.rows.filter(r=>(r[0]+' '+r[1]+' '+r[5]).toLowerCase().includes(q)&&(!k||r[0].endsWith(k))&&(b===''||r[3].includes(Number(b)))&&(!ev||r[4])&&(!diff||!r[6]));const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);$('fn').textContent=`${filtered.length.toLocaleString()} matching versions out of ${D.rows.length.toLocaleString()}`;$('page').textContent=`Page ${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1;$('rows').innerHTML=filtered.slice(page*pageSize,(page+1)*pageSize).map(r=>{const bid=b===''?r[3][0]:Number(b),ref=D.branches[bid],url='https://github.com/yinli-systems/antibody-antigen-interface-research/blob/'+ref.sha+'/'+r[0].split('/').map(encodeURIComponent).join('/');return `<tr><td><a href="${E(url)}" target="_blank" rel="noopener">${E(r[0])}</a><br><code>${E(r[1].slice(0,12))}</code></td><td>${E(r[0].includes('.')?r[0].split('.').pop():'—')}</td><td>${r[2]===null?'—':r[2].toLocaleString()}</td><td>${r[6]?'same bytes':'no / different'}</td><td><details><summary>${r[3].length} HEADs</summary>${r[3].map(i=>E(D.branches[i].name)).join('<br>')}</details></td></tr>`}).join('');}['cq','level','work'].forEach(id=>$(id).addEventListener('input',cards));['fq','branch','kind','ev','diff'].forEach(id=>$(id).addEventListener('input',()=>files()));$('prev').onclick=()=>{page--;files(false)};$('next').onclick=()=>{page++;files(false)};cards();files();
</script></body></html>'''
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args();generated=products();bad=[]
    for relative,data in generated.items():
        path=ROOT/relative
        if args.check:
            if not path.exists() or path.read_bytes()!=data:bad.append(relative)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    if bad:raise SystemExit('Generated atlas drift: '+', '.join(bad))
    print(json.dumps(dict(status='PASS' if args.check else 'BUILT',generated_files=len(generated),scope='Pinned reporting/inventory, not a model rerun'),indent=2))
if __name__=='__main__':main()
