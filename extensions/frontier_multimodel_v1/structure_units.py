#!/usr/bin/env python3
"""Freeze sequence-indexed chain structures from the development-only AF3 cache.

The cache contains antibody and antigen chains predicted independently. This
materializer never reads contact labels. Repeated sequences may have more than
one predicted geometry; the canonical one is selected by the highest AF3
ranking score, then by source artifact SHA-256. Every alternative remains in the
receipt. Antibody heavy and light chains are emitted as separate PDB units, so
no artificial VH/VL peptide bond is introduced for SaProt or ProSST.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, tempfile, time
from collections import defaultdict
from pathlib import Path
from typing import Any

AA3={
'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE',
'L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL',
'U':'SEC','O':'PYL','B':'ASX','Z':'GLX','J':'LEU','X':'UNK'
}

def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()

def sequence_key(sequence:str)->str:
 return hashlib.sha256(sequence.encode('ascii')).hexdigest()

def atomic_json(path:Path,payload:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 encoded=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n'
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp-',suffix='.json'); os.close(fd)
 try:
  Path(tmp).write_text(encoded); os.replace(tmp,path)
 except BaseException:
  Path(tmp).unlink(missing_ok=True); raise

def read_manifest(path:Path)->dict[str,dict[str,str]]:
 rows={}
 with path.open(newline='',encoding='utf-8') as handle:
  reader=csv.DictReader(handle)
  required={'INSTANCE','vh_seq','vl_seq','ag_seq'}
  if not required<=set(reader.fieldnames or []): raise ValueError('canonical manifest lacks sequence columns')
  for row in reader:
   iid=row['INSTANCE'];
   if iid in rows: raise ValueError(f'duplicate instance {iid}')
   rows[iid]=row
 return rows

def contiguous_segments(ids):
 values=ids.tolist(); out=[]; start=0
 for i in range(1,len(values)+1):
  if i==len(values) or values[i]!=values[start]:
   out.append((int(values[start]),start,i)); start=i
 return out

def exact_backbone_from_cif(model_path:Path,sequence:str,reference,expected_sha256:str):
 import numpy as np, torch
 from Bio.PDB import MMCIFParser
 from Bio.SeqUtils import seq1
 if not model_path.exists() or sha256_file(model_path)!=expected_sha256:
  raise ValueError(f'predicted CIF missing or digest mismatch: {model_path}')
 structure=MMCIFParser(QUIET=True).get_structure('predicted',str(model_path))
 candidates=[]
 for chain in next(structure.get_models()):
  aas=[]; coords=[]
  for residue in chain:
   if residue.id[0] not in {' ','H_MSE'}: continue
   if any(name not in residue for name in ('N','CA','C','O')): continue
   aa=seq1(residue.resname,custom_map={'MSE':'M'},undef_code='X')
   aas.append(aa); coords.append([residue[name].coord.tolist() for name in ('N','CA','C','O')])
  if ''.join(aas)!=sequence: continue
  tensor=torch.tensor(coords,dtype=torch.float32)
  ncac=tensor[:,:3]
  ref=torch.as_tensor(reference,dtype=torch.float32)
  if tuple(ncac.shape)!=tuple(ref.shape): continue
  max_abs=float((ncac-ref).abs().max()); rmsd=float(torch.sqrt(((ncac-ref)**2).mean()))
  candidates.append((max_abs,rmsd,str(chain.id),tensor))
 if not candidates: raise ValueError(f'no exact sequence/O-complete chain in {model_path}')
 candidates.sort(key=lambda x:(x[0],x[1],x[2])); max_abs,rmsd,chain_id,tensor=candidates[0]
 # Cache coordinates were parsed from these exact model files; a larger difference
 # means chain identity or residue ordering is wrong rather than harmless precision.
 if max_abs>0.02: raise ValueError(f'predicted CIF/cache NCAC mismatch {max_abs:.6f} in {model_path}')
 return tensor,{'model_path':str(model_path),'model_sha256':expected_sha256,'chain_id':chain_id,
                'ncac_max_abs':max_abs,'ncac_rmsd':rmsd,'oxygen_source':'exact_predicted_cif'}

def pdb_text(sequence:str,atoms)->str:
 import torch
 tensor=torch.as_tensor(atoms,dtype=torch.float32).contiguous()
 if tuple(tensor.shape)!=(len(sequence),4,3) or not bool(torch.isfinite(tensor).all()):
  raise ValueError('backbone must be finite [L,4,3] N/CA/C/O aligned to sequence')
 lines=[]; serial=1
 for resid,(aa,resatoms) in enumerate(zip(sequence,tensor),1):
  res=AA3.get(aa,'UNK')
  for atom,xyz,element in zip(('N','CA','C','O'),resatoms,('N','C','C','O')):
   x,y,z=map(float,xyz)
   lines.append(f'ATOM  {serial:5d} {atom:^4s} {res:>3s} A{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{100.0:6.2f}          {element:>2s}')
   serial+=1
 lines.extend([f'TER   {serial:5d}      {AA3.get(sequence[-1],"UNK"):>3s} A{len(sequence):4d}','END'])
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--structure-cache',type=Path,required=True)
 p.add_argument('--canonical-manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(); out=a.output.resolve()
 if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
 out.mkdir(parents=True); (out/'pdb/antibody').mkdir(parents=True); (out/'pdb/antigen').mkdir(parents=True)
 import torch
 source=a.structure_cache.resolve(); sm=source/'STRUCTURE_CACHE_MANIFEST.json'
 payload=json.loads(sm.read_text()); rows=read_manifest(a.canonical_manifest)
 if payload.get('counts',{}).get('roles',{}).get('train',0)<=0 or payload.get('counts',{}).get('roles',{}).get('validation',0)<=0:
  raise ValueError('expected development train and validation roles')
 if any(r.get('role') not in {'train','validation'} for r in payload['records']):
  raise ValueError('test firewall: structure cache contains an undeclared role')
 candidates=defaultdict(list); excluded=[]
 for index,record in enumerate(payload['records']):
  iid=record['instance_id']; row=rows.get(iid)
  if row is None: excluded.append({'instance_id':iid,'reason':'absent canonical row'}); continue
  artifact=source/record['artifact_path']
  if sha256_file(artifact)!=record['artifact_sha256']: raise ValueError(f'artifact digest mismatch {iid}')
  data=torch.load(artifact,map_location='cpu',weights_only=False)
  if data['geometry_source']!='independent_sequence_predicted_af3': raise ValueError('unexpected geometry source')
  if bool(data['antibody_missing_frame_mask'].any()) or bool(data['antigen_missing_frame_mask'].any()):
   excluded.append({'instance_id':iid,'reason':'missing predicted frame'}); continue
  vh=(row.get('vh_seq') or '').strip(); vl=(row.get('vl_seq') or '').strip(); ag=(row.get('ag_seq') or '').strip()
  if not vh or not ag: excluded.append({'instance_id':iid,'reason':'empty primary sequence'}); continue
  if sequence_key(vh+vl)!=record['antibody_sequence_sha256'] or sequence_key(ag)!=record['antigen_sequence_sha256']:
   raise ValueError(f'sequence authority mismatch {iid}')
  segments=contiguous_segments(data['antibody_chain_ids'])
  expected=[vh]+([vl] if vl else [])
  lengths=[end-start for _,start,end in segments]
  if lengths!=[len(x) for x in expected]:
   raise ValueError(f'antibody chain layout mismatch {iid}: {lengths} vs {[len(x) for x in expected]}')
  abscore=float(record['antibody_prediction']['selected_ranking_score'])
  agscore=float(record['antigen_prediction']['selected_ranking_score'])
  for chain_index,((chain_id,start,end),seq) in enumerate(zip(segments,expected)):
   key=sequence_key(seq); coords=data['antibody_backbone'][start:end].contiguous()
   candidates[('antibody',key)].append({'sequence':seq,'score':abscore,'instance_id':iid,'role':record['role'],
    'source_artifact':str(artifact),'source_artifact_sha256':record['artifact_sha256'],'start':start,'end':end,
    'chain_index':chain_index,'geometry_sha256':hashlib.sha256(coords.numpy().tobytes()).hexdigest(),
    'model_path':record['antibody_prediction']['model_path'],'model_sha256':record['antibody_prediction']['model_sha256']})
  coords=data['antigen_backbone'].contiguous(); key=sequence_key(ag)
  candidates[('antigen',key)].append({'sequence':ag,'score':agscore,'instance_id':iid,'role':record['role'],
   'source_artifact':str(artifact),'source_artifact_sha256':record['artifact_sha256'],'start':0,'end':len(ag),
   'chain_index':0,'geometry_sha256':hashlib.sha256(coords.numpy().tobytes()).hexdigest(),
   'model_path':record['antigen_prediction']['model_path'],'model_sha256':record['antigen_prediction']['model_sha256']})
 records=[]; conflict_groups=[]
 for (side,key),options in sorted(candidates.items()):
  # No task label enters this choice. Confidence is maximised; bytes break exact ties.
  ordered=sorted(options,key=lambda x:(-x['score'],x['source_artifact_sha256'],x['instance_id'],x['chain_index']))
  chosen=ordered[0]; source_data=torch.load(chosen['source_artifact'],map_location='cpu',weights_only=False)
  backbone=source_data[f'{side}_backbone'][chosen['start']:chosen['end']].contiguous()
  if hashlib.sha256(backbone.numpy().tobytes()).hexdigest()!=chosen['geometry_sha256']: raise AssertionError('geometry changed')
  exact_atoms,exact_receipt=exact_backbone_from_cif(Path(chosen['model_path']),chosen['sequence'],backbone,chosen['model_sha256'])
  path=out/'pdb'/side/f'{key}.pdb'; path.write_text(pdb_text(chosen['sequence'],exact_atoms))
  variants=sorted({x['geometry_sha256'] for x in options})
  rec={'side':side,'sequence_sha256':key,'sequence':chosen['sequence'],'length':len(chosen['sequence']),
       'pdb_path':str(path.relative_to(out)),'pdb_sha256':sha256_file(path),'selection':'highest_af3_ranking_score_then_source_sha256',
       'chosen':{k:chosen[k] for k in chosen if k!='sequence'},'exact_coordinate_receipt':exact_receipt,
       'candidate_count':len(options),'geometry_variant_count':len(variants),
       'candidate_roles':sorted({x['role'] for x in options}),'candidate_geometry_sha256':variants}
  records.append(rec)
  if len(variants)>1: conflict_groups.append({k:rec[k] for k in ['side','sequence_sha256','candidate_count','geometry_variant_count','chosen']})
 manifest={'format_version':'frontier_structure_units_v2','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  'evidence_role':'development train/validation feature materialization; no task labels read','source_structure_cache':str(source),
  'source_manifest_sha256':sha256_file(sm),'canonical_manifest_sha256':sha256_file(a.canonical_manifest),
  'selection_policy':'maximum AF3 ranking score, then source artifact SHA-256; fixed without contact labels',
  'coordinate_policy':'exact N/CA/C/O atoms from the selected independently predicted CIF, verified against the cached NCAC tensor',
  'record_count':len(records),'counts_by_side':{s:sum(r['side']==s for r in records) for s in ['antibody','antigen']},
  'conflicting_geometry_sequence_count':len(conflict_groups),'conflicts':conflict_groups,'exclusions':excluded,'records':records,
  'test_loaded':False,'contact_labels_loaded':False}
 atomic_json(out/'STRUCTURE_UNITS_MANIFEST.json',manifest)
 atomic_json(out/'STRUCTURE_UNITS_RECEIPT.json',{'status':'PASS','manifest_sha256':sha256_file(out/'STRUCTURE_UNITS_MANIFEST.json'),
  'record_count':len(records),'conflicting_geometry_sequence_count':len(conflict_groups),'test_loaded':False,'contact_labels_loaded':False})
 print(json.dumps({'status':'PASS','record_count':len(records),'conflicts':len(conflict_groups),'output':str(out)}))
if __name__=='__main__': main()
