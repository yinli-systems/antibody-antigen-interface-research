#!/usr/bin/env python3
"""Prepare SaProt/ProSST structural tokens and extract frozen layer features.

Every command is development-only and refuses to overwrite. SaProt receives
real Foldseek 3Di tokens; ProSST receives tokens from the official 2048-state
GVP quantizer. Long proteins are evaluated in overlapping windows and merged in
residue space. Artifacts use the repository's existing content-addressed format.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile, time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(REPOSITORY_ROOT/'src'))
from interaction_first.pretrained_representations import (
 RepresentationUnit, build_artifact, representation_path, write_artifact,
 validate_representation_artifact,
)

def sha256_file(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()

def atomic_json(path:Path,payload:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n'
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.tmp-',suffix='.json'); os.close(fd)
 try: Path(tmp).write_text(data); os.replace(tmp,path)
 except BaseException: Path(tmp).unlink(missing_ok=True); raise

def load_units(path:Path)->tuple[dict,list[dict]]:
 payload=json.loads(path.read_text())
 if payload.get('format_version') not in {'frontier_structure_units_v1','frontier_structure_units_v2'}: raise ValueError('wrong structure unit authority')
 if payload.get('test_loaded') or payload.get('contact_labels_loaded'): raise ValueError('data firewall failed')
 rows=payload.get('records',[])
 if len(rows)!=payload.get('record_count'): raise ValueError('structure record count mismatch')
 keys={(r['side'],r['sequence_sha256']) for r in rows}
 if len(keys)!=len(rows): raise ValueError('duplicate side/sequence structure units')
 return payload,rows

def selected(rows:list[dict],side:str,shard_index:int,shard_count:int)->list[dict]:
 if side not in {'antibody','antigen'} or not 0<=shard_index<shard_count: raise ValueError('bad side or shard')
 return [r for r in sorted(rows,key=lambda x:x['sequence_sha256']) if r['side']==side and int(r['sequence_sha256'][:16],16)%shard_count==shard_index]

def windows(length:int,max_residues:int,overlap:int):
 if max_residues<=0 or overlap<0 or overlap>=max_residues: raise ValueError('bad window contract')
 if length<=max_residues: return [(0,length)]
 stride=max_residues-overlap; starts=list(range(0,max(1,length-max_residues+1),stride))
 last=length-max_residues
 if starts[-1]!=last: starts.append(last)
 return [(s,min(length,s+max_residues)) for s in sorted(set(starts))]

def prepare_saprot(args):
 authority,rows=load_units(args.structure_units)
 out=args.output.resolve()
 if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
 out.mkdir(parents=True)
 binary=args.foldseek.resolve()
 binary_sha=sha256_file(binary)
 emitted=[]
 for side in ['antibody','antigen']:
  pdb_dir=args.structure_units.parent/'pdb'/side
  tsv=out/f'{side}.tsv'
  subprocess.run([str(binary),'structureto3didescriptor','-v','0','--threads',str(args.threads),'--chain-name-mode','1',str(pdb_dir),str(tsv)],check=True)
  expected={r['sequence_sha256']:r for r in rows if r['side']==side}; seen=set()
  with tsv.open() as handle:
   for line in handle:
    fields=line.rstrip('\n').split('\t')
    if len(fields)<3: raise ValueError('truncated Foldseek descriptor')
    desc,aa,three_di=fields[:3]; stem=Path(desc.split()[0]).name
    key=stem.split('.pdb_')[0] if '.pdb_' in stem else stem.split('_')[0]
    record=expected.get(key)
    if record is None or key in seen: raise ValueError(f'unexpected or duplicate Foldseek key {key}')
    if aa!=record['sequence'] or len(three_di)!=len(aa): raise ValueError(f'Foldseek sequence mismatch {key}')
    invalid={symbol for symbol in three_di if not (symbol.isascii() and (symbol.isalpha() or symbol=='#'))}
    if invalid: raise ValueError(f'unknown 3Di symbols {sorted(invalid)}')
    token_path=out/side/key[:2]/f'{key}.json'
    atomic_json(token_path,{'format_version':'saprot_structural_tokens_v1','side':side,'sequence_sha256':key,
      'sequence':aa,'three_di':three_di.lower(),'combined_sequence':''.join(a+b.lower() for a,b in zip(aa,three_di)),
      'pdb_sha256':record['pdb_sha256'],'foldseek_binary_sha256':binary_sha,
      'foldseek_version':args.foldseek_version,'structure_authority_sha256':sha256_file(args.structure_units)})
    seen.add(key); emitted.append(str(token_path.relative_to(out)))
  if seen!=set(expected): raise ValueError(f'Foldseek coverage mismatch {side}: missing {len(set(expected)-seen)}')
 receipt={'status':'PASS','format_version':'saprot_token_receipt_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  'structure_units_sha256':sha256_file(args.structure_units),'foldseek_binary_sha256':binary_sha,
  'foldseek_version':args.foldseek_version,'token_count':len(emitted),'paths_sha256':hashlib.sha256(('\n'.join(sorted(emitted))+'\n').encode()).hexdigest(),
  'test_loaded':False,'contact_labels_loaded':False}
 atomic_json(out/'RECEIPT.json',receipt); print(json.dumps(receipt))

def load_prosst_quantizer(args):
 import joblib, torch
 sys.path.insert(0,str(args.prosst_source.resolve()))
 from prosst.structure.encoder.gvp import AutoGraphEncoder
 model=AutoGraphEncoder(node_in_dim=(20,3),node_h_dim=(256,32),edge_in_dim=(32,1),edge_h_dim=(64,2),num_layers=6)
 state=torch.load(args.prosst_source/'prosst/structure/static/AE.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(state); model.to(args.device).eval()
 cluster=joblib.load(args.prosst_source/'prosst/structure/static/2048.joblib')
 return model,cluster

def quantize_one(pdb:Path,model,cluster,source:Path,device:str,batch_nodes:int=8000):
 import numpy as np, torch
 from torch_geometric.data import Batch
 from torch_scatter import scatter_mean
 sys.path.insert(0,str(source.resolve()))
 from prosst.structure.build_graph import generate_graph
 from prosst.structure.build_subgraph import generate_pos_subgraph
 from prosst.structure.utils.data_utils import convert_graph
 graph=generate_graph(str(pdb),10)
 sub=[]; labels=[]; nodes=0
 @torch.inference_mode()
 def flush():
  nonlocal sub,nodes
  if not sub:return
  batch=Batch.from_data_list(sub).to(device); batch.node_s=torch.zeros_like(batch.node_s)
  h=(batch.node_s,batch.node_v); e=(batch.edge_s,batch.edge_v)
  z=model.get_embedding(h,batch.edge_index,e)
  z=scatter_mean(z,batch.batch,dim=0).float().cpu(); z=torch.nn.functional.normalize(z,p=2,dim=1)
  labels.extend(int(x) for x in cluster.predict(z.numpy()).tolist()); sub=[]; nodes=0
 for anchor in range(len(graph.aa_seq)):
  item=convert_graph(generate_pos_subgraph(graph,subgraph_depth=None,max_distance=10,anchor_nodes=anchor,verbose=False,pure_subgraph=True)[anchor])
  n=int(item.node_s.shape[0])
  if sub and nodes+n>batch_nodes: flush()
  sub.append(item); nodes+=n
 flush(); return graph.aa_seq,labels

def prepare_prosst(args):
 authority,rows=load_units(args.structure_units); todo=selected(rows,args.side,args.shard_index,args.shard_count)
 if args.max_records is not None: todo=todo[:args.max_records]
 out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 model,cluster=load_prosst_quantizer(args); written=[]; skipped=[]
 for n,r in enumerate(todo,1):
  target=out/args.side/r['sequence_sha256'][:2]/f"{r['sequence_sha256']}.json"
  if target.exists():
   old=json.loads(target.read_text())
   if old.get('pdb_sha256')!=r['pdb_sha256']: raise ValueError('existing ProSST token has different source')
   skipped.append(r['sequence_sha256']); continue
  pdb=args.structure_units.parent/r['pdb_path']; aa,labels=quantize_one(pdb,model,cluster,args.prosst_source,args.device,args.batch_nodes)
  if aa!=r['sequence'] or len(labels)!=len(aa) or any(not 0<=x<2048 for x in labels): raise ValueError(f'ProSST quantizer mismatch {r["sequence_sha256"]}')
  atomic_json(target,{'format_version':'prosst_structural_tokens_v1','side':args.side,'sequence_sha256':r['sequence_sha256'],
    'sequence':aa,'structure_tokens':labels,'pdb_sha256':r['pdb_sha256'],'quantizer_source_commit':args.prosst_source_commit,
    'quantizer_ae_sha256':sha256_file(args.prosst_source/'prosst/structure/static/AE.pt'),
    'quantizer_cluster_sha256':sha256_file(args.prosst_source/'prosst/structure/static/2048.joblib'),
    'structure_authority_sha256':sha256_file(args.structure_units)})
  written.append(r['sequence_sha256'])
  if n%25==0 or n==len(todo): print(f'quantized {args.side} shard={args.shard_index} {n}/{len(todo)}',flush=True)
 receipt={'status':'PASS','format_version':'prosst_token_shard_receipt_v1','side':args.side,'shard_index':args.shard_index,'shard_count':args.shard_count,
  'requested':len(todo),'written':len(written),'skipped_exact_existing':len(skipped),'keys_sha256':hashlib.sha256(('\n'.join(sorted([r['sequence_sha256'] for r in todo]))+'\n').encode()).hexdigest(),
  'structure_units_sha256':sha256_file(args.structure_units),'test_loaded':False,'contact_labels_loaded':False}
 atomic_json(out/f'RECEIPT_{args.side}_{args.shard_index:03d}_of_{args.shard_count:03d}.json',receipt); print(json.dumps(receipt))

def merge_windows(sequence:str,token_payload:dict,model_kind:str,model,tokenizer,layers:list[int],device:str,max_residues:int,overlap:int):
 import torch
 length=len(sequence); sums={l:torch.zeros(length,int(model.config.hidden_size),dtype=torch.float32) for l in layers}; counts=torch.zeros(length,1)
 for start,end in windows(length,max_residues,overlap):
  seq=sequence[start:end]
  if model_kind=='saprot':
   combined=token_payload['combined_sequence'][2*start:2*end]
   encoded=tokenizer(combined,return_tensors='pt',add_special_tokens=True)
   if int(encoded['input_ids'].shape[1])!=len(seq)+2: raise ValueError('SaProt token count is not residue aligned')
   encoded={k:v.to(device) for k,v in encoded.items()}
   with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    outputs=model.esm(**encoded,output_hidden_states=True,return_dict=True)
  else:
   encoded=tokenizer(seq,return_tensors='pt',add_special_tokens=True)
   if int(encoded['input_ids'].shape[1])!=len(seq)+2: raise ValueError('ProSST amino-acid token count is not residue aligned')
   ss=[1]+[int(x)+3 for x in token_payload['structure_tokens'][start:end]]+[2]
   encoded={k:v.to(device) for k,v in encoded.items()}; ss_ids=torch.tensor([ss],dtype=torch.long,device=device)
   with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    outputs=model.prosst(input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],ss_input_ids=ss_ids,output_hidden_states=True,return_dict=True)
  hidden=outputs.hidden_states
  for layer in layers:
   x=hidden[layer][0,1:len(seq)+1].float().cpu()
   if tuple(x.shape)!=(len(seq),int(model.config.hidden_size)): raise ValueError('hidden state alignment failure')
   sums[layer][start:end]+=x
  counts[start:end]+=1
 if bool((counts==0).any()): raise AssertionError('window coverage gap')
 return {layer:(sums[layer]/counts).to(torch.float16).contiguous() for layer in layers},[(a,b) for a,b in windows(length,max_residues,overlap)]

def extract(args):
 import torch
 from transformers import AutoModelForMaskedLM,AutoTokenizer,EsmForMaskedLM,EsmTokenizer
 authority,rows=load_units(args.structure_units); todo=selected(rows,args.side,args.shard_index,args.shard_count)
 if args.max_records is not None: todo=todo[:args.max_records]
 if args.model_kind=='saprot':
  tokenizer=EsmTokenizer.from_pretrained(args.model_path,local_files_only=True)
  model=EsmForMaskedLM.from_pretrained(args.model_path,local_files_only=True).to(args.device).eval(); token_format='saprot_structural_tokens_v1'
 else:
  tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True)
  model=AutoModelForMaskedLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True).to(args.device).eval(); token_format='prosst_structural_tokens_v1'
 if max(args.layers)>int(model.config.num_hidden_layers) or min(args.layers)<0: raise ValueError('layer outside model')
 model_hashes={str(p.relative_to(args.model_path)):sha256_file(p) for p in sorted(args.model_path.rglob('*')) if p.is_file() and '.cache' not in p.parts}
 model_tree_sha=hashlib.sha256(json.dumps(model_hashes,sort_keys=True).encode()).hexdigest()
 receipts=[]
 for n,r in enumerate(todo,1):
  token_path=args.tokens/args.side/r['sequence_sha256'][:2]/f"{r['sequence_sha256']}.json"
  payload=json.loads(token_path.read_text())
  if payload.get('format_version')!=token_format or payload.get('sequence_sha256')!=r['sequence_sha256'] or payload.get('sequence')!=r['sequence'] or payload.get('pdb_sha256')!=r['pdb_sha256']:
   raise ValueError(f'token authority mismatch {r["sequence_sha256"]}')
  features,window_list=merge_windows(r['sequence'],payload,args.model_kind,model,tokenizer,args.layers,args.device,args.max_residues,args.overlap)
  for layer,tensor in features.items():
   root=args.output/f'{args.model_kind}_layer_{layer:02d}'/args.side
   model_name=f'{args.model_kind}_{args.model_revision}_layer_{layer}'
   target=representation_path(root,r['sequence_sha256'])
   if target.exists(): raise FileExistsError(f'refusing to overwrite {target}')
   artifact=build_artifact(RepresentationUnit(r['sequence']),tensor,model_name=model_name,
    model_revision=args.model_revision,chunk_residues=1024)
   artifact['source_pdb_sha256']=r['pdb_sha256']; artifact['structural_token_sha256']=sha256_file(token_path)
   artifact['model_tree_sha256']=model_tree_sha; artifact['layer_index']=layer; artifact['windows']=window_list
   validate_representation_artifact(artifact,expected_key=r['sequence_sha256'],expected_dim=int(model.config.hidden_size))
   write_artifact(root,artifact)
  receipts.append({'sequence_sha256':r['sequence_sha256'],'side':args.side,'length':len(r['sequence']),'token_sha256':sha256_file(token_path),'windows':window_list})
  if n%25==0 or n==len(todo): print(f'extracted {args.model_kind} {args.side} shard={args.shard_index} {n}/{len(todo)}',flush=True)
 receipt={'status':'PASS','format_version':'structural_plm_extract_shard_v1','model_kind':args.model_kind,'model_revision':args.model_revision,
  'model_tree_sha256':model_tree_sha,'layers':args.layers,'hidden_size':int(model.config.hidden_size),'side':args.side,
  'shard_index':args.shard_index,'shard_count':args.shard_count,'records':receipts,'record_count':len(receipts),
  'structure_units_sha256':sha256_file(args.structure_units),'test_loaded':False,'contact_labels_loaded':False,
  'dtype_on_disk':'float16','window_policy':{'max_residues':args.max_residues,'overlap':args.overlap,'merge':'uniform mean in overlap'}}
 atomic_json(args.output/f'RECEIPT_{args.model_kind}_{args.side}_{args.shard_index:03d}_of_{args.shard_count:03d}.json',receipt); print(json.dumps({k:receipt[k] for k in ['status','model_kind','side','record_count']}))

def build_parser():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('prepare-saprot'); a.add_argument('--structure-units',type=Path,required=True); a.add_argument('--foldseek',type=Path,required=True); a.add_argument('--foldseek-version',required=True); a.add_argument('--threads',type=int,default=8); a.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('prepare-prosst'); q.add_argument('--structure-units',type=Path,required=True); q.add_argument('--prosst-source',type=Path,required=True); q.add_argument('--prosst-source-commit',required=True); q.add_argument('--side',required=True); q.add_argument('--shard-index',type=int,required=True); q.add_argument('--shard-count',type=int,required=True); q.add_argument('--batch-nodes',type=int,default=8000); q.add_argument('--max-records',type=int); q.add_argument('--device',default='cuda'); q.add_argument('--output',type=Path,required=True)
 e=sub.add_parser('extract'); e.add_argument('--model-kind',choices=['saprot','prosst'],required=True); e.add_argument('--structure-units',type=Path,required=True); e.add_argument('--tokens',type=Path,required=True); e.add_argument('--model-path',type=Path,required=True); e.add_argument('--model-revision',required=True); e.add_argument('--layers',type=int,nargs='+',required=True); e.add_argument('--side',required=True); e.add_argument('--shard-index',type=int,required=True); e.add_argument('--shard-count',type=int,required=True); e.add_argument('--device',default='cuda'); e.add_argument('--max-records',type=int); e.add_argument('--max-residues',type=int,default=1000); e.add_argument('--overlap',type=int,default=128); e.add_argument('--output',type=Path,required=True)
 return p
if __name__=='__main__':
 a=build_parser().parse_args(); {'prepare-saprot':prepare_saprot,'prepare-prosst':prepare_prosst,'extract':extract}[a.cmd](a)
