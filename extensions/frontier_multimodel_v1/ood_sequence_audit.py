#!/usr/bin/env python3
"""Metadata-frozen OOD candidate sequence decontamination with MMseqs2.

No coordinates or contact labels are read. Full Fab chains are compared against
project variable domains using target coverage, while antigen chains use either
query or target coverage. The output is an admission audit, not a performance
result.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n';fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp-');os.close(fd)
 try:Path(tmp).write_text(data);os.replace(tmp,path)
 except BaseException:Path(tmp).unlink(missing_ok=True);raise
def clean(seq):return ''.join(str(seq or '').split()).upper()
def fasta(path,rows):
 with Path(path).open('x') as f:
  for name,seq in rows:
   if not seq or set(seq)-set('ACDEFGHIKLMNPQRSTVWYBXZJUO'):raise ValueError(f'invalid sequence {name}')
   f.write(f'>{name}\n{seq}\n')
def run(mmseqs,query,target,out,tmp,threads):
 subprocess.run([str(mmseqs),'easy-search',str(query),str(target),str(out),str(tmp),'--threads',str(threads),'-s','7.5','-e','1e-3','--max-seqs','10000','--format-output','query,target,fident,alnlen,qcov,tcov,evalue,bits'],check=True)
def read_hits(path):
 hits=[]
 if not Path(path).exists():return hits
 with Path(path).open() as f:
  for line in f:
   q,t,fi,aln,qc,tc,e,b=line.rstrip().split('\t')
   hits.append({'query':q,'target':t,'fident':float(fi),'alnlen':int(aln),'qcov':float(qc),'tcov':float(tc),'evalue':float(e),'bits':float(b)})
 return hits
def main():
 a=argparse.ArgumentParser();a.add_argument('--freeze',type=Path,required=True);a.add_argument('--canonical-manifest',type=Path,required=True);a.add_argument('--mmseqs',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--threads',type=int,default=8);x=a.parse_args()
 out=x.output.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);freeze=json.loads(x.freeze.read_text())
 if freeze.get('format_version')!='untouched_ood_metadata_freeze_v1' or freeze.get('coordinates_read_for_selected_entries') or freeze.get('contact_labels_read'):raise ValueError('freeze firewall')
 query={'antibody':[],'antigen':[]};query_meta={}
 for rec in freeze['records']:
  for role in query:
   for entity in rec[f'{role}_entities']:
    name=f"{rec['pdb_id']}|{role}|entity{entity['entity_id']}";sequence=clean(entity['sequence']);query[role].append((name,sequence));query_meta[name]={'pdb_id':rec['pdb_id'],'role':role,'entity_id':str(entity['entity_id']),'description':entity['description'],'length':len(sequence)}
 targets={'antibody':{},'antigen':{}};pdb_ids=set()
 with x.canonical_manifest.open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  for row in reader:
   pdb_ids.add((row.get('PDB_ID') or '').upper())
   for role,cols in [('antibody',['vh_seq','vl_seq']),('antigen',['ag_seq'])]:
    for col in cols:
     seq=clean(row.get(col));
     if seq:targets[role].setdefault(hashlib.sha256(seq.encode()).hexdigest(),seq)
 for role in query:
  fasta(out/f'{role}_query.fasta',query[role]);fasta(out/f'{role}_project.fasta',[(k,v) for k,v in sorted(targets[role].items())])
  run(x.mmseqs,out/f'{role}_query.fasta',out/f'{role}_project.fasta',out/f'{role}_hits.tsv',out/f'tmp_{role}',x.threads)
 entity_results=[]
 for role in query:
  by={}
  for h in read_hits(out/f'{role}_hits.tsv'):by.setdefault(h['query'],[]).append(h)
  for name,sequence in query[role]:
   hits=by.get(name,[])
   if role=='antibody':eligible=[h for h in hits if h['alnlen']>=70 and h['tcov']>=0.80]
   else:eligible=[h for h in hits if h['alnlen']>=50 and max(h['qcov'],h['tcov'])>=0.80]
   best=max(eligible,key=lambda h:(h['fident'],h['bits']),default=None);meta=query_meta[name]
   entity_results.append({**meta,'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),'best_eligible_hit':best,'eligible_hit_count':len(eligible),'passes_70pct_identity_gate':best is None or best['fident']<0.70})
 pdb_results=[]
 for rec in freeze['records']:
  rows=[e for e in entity_results if e['pdb_id']==rec['pdb_id']];id_absent=rec['pdb_id'].upper() not in pdb_ids
  pdb_results.append({'pdb_id':rec['pdb_id'],'project_pdb_id_absent':id_absent,'entity_results':rows,
   'passes_sequence_gate':id_absent and all(e['passes_70pct_identity_gate'] for e in rows),
   'gate':'PDB ID absent; every frozen antibody entity <70% identity at >=80% project-variable target coverage and every antigen entity <70% at >=80% query-or-target coverage'})
 payload={'format_version':'untouched_ood_sequence_audit_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'metadata_freeze_sha256':sha(x.freeze),'canonical_manifest_sha256':sha(x.canonical_manifest),'mmseqs_binary_sha256':sha(x.mmseqs),'mmseqs_version':subprocess.check_output([str(x.mmseqs),'version'],text=True).strip(),
  'pdb_candidates':len(pdb_results),'pdb_passing_sequence_gate':sum(r['passes_sequence_gate'] for r in pdb_results),'results':pdb_results,
  'coordinates_read':False,'contact_labels_read':False,'performance_evaluated':False,
  'limitations':['Full Fab sequences are compared with project variable domains using target coverage; this is conservative but not ANARCI numbering.','Sequence passage does not imply interface-structure novelty; Foldseek/interface audit remains required.','A failed entity blocks its PDB; no post-hoc threshold changes are authorized.']}
 atomic(out/'AUDIT.json',payload);print(json.dumps({'status':'PASS_AUDIT_COMPLETE','pdb_candidates':len(pdb_results),'pdb_passing_sequence_gate':payload['pdb_passing_sequence_gate'],'output':str(out)}))
if __name__=='__main__':main()
