import json,glob,pathlib,hashlib,numpy as np,datetime
R=pathlib.Path("/ssd/scxi253/frontier_multimodel_v1/frontier_performance_v1/rescue_qualification_v2_20260930/gated_rescue_v1")
files=sorted(R.glob("runs/seed_*.json")); ds=[json.load(open(p)) for p in files]; rec=[x for d in ds for x in d["records"]]
assert len(rec)==1953 and all(len(d["records"])==651 for d in ds)
seeds=sorted({(r["rescue_seed"],r["downstream_seed"]) for r in rec}); conds=sorted({r["condition"] for r in rec}); ids=sorted({r["instance_id"] for r in rec}); comps=sorted({r["component"] for r in rec}); assert len(seeds)==3 and len(ids)==217 and len(comps)==68
# exact complete grid
assert len({(r["instance_id"],r["rescue_seed"],r["downstream_seed"],r["condition"]) for r in rec})==1953
rng=np.random.default_rng(20260930)
def crossed(field_a,field_b,condition=None):
 rr=[r for r in rec if condition is None or r["condition"]==condition]
 mat=np.empty((len(seeds),len(comps))); raw=[]
 for si,(s,d) in enumerate(seeds):
  for ci,c in enumerate(comps):
   q=[r[field_a]-r[field_b] for r in rr if r["rescue_seed"]==s and r["component"]==c]
   assert q; mat[si,ci]=np.mean(q);raw+=q
 B=20000; vals=np.empty(B); comp_only=np.empty(B)
 for b in range(B):
  ss=rng.integers(len(seeds),size=len(seeds)); cc=rng.integers(len(comps),size=len(comps)); vals[b]=mat[ss][:,cc].mean();comp_only[b]=mat[:,cc].mean()
 return {"equal_complex":float(np.mean(raw)),"equal_component_equal_seed":float(mat.mean()),"crossed_seed_component_ci95":list(map(float,np.quantile(vals,[.025,.975]))),"component_only_ci95":list(map(float,np.quantile(comp_only,[.025,.975]))),"positive_component_seed_cell_fraction":float((mat>0).mean()),"seed_effects":{str(s):float(mat[i].mean()) for i,(s,d) in enumerate(seeds)},"component_effect_q10_q50_q90":list(map(float,np.quantile(mat.mean(0),[.1,.5,.9]))),"min_max_component_mean":list(map(float,[mat.mean(0).min(),mat.mean(0).max()]))}
# clean one row per iid/seed: duplicated identically across conditions
clean=[]
for s,d in seeds:
 for iid in ids:
  q=[r for r in rec if r["rescue_seed"]==s and r["instance_id"]==iid]; assert len(q)==3
  for k in ["aligned_ap","clean_rescued_ap","gated_clean_ap","clean_gate_score","apply_rescue_clean","component"]: assert len({str(x[k]) for x in q})==1
  clean.append(q[0])
def crossed_clean(a,b):
 global rec
 save=rec;rec=clean
 try:return crossed(a,b,None)
 finally:rec=save
out={"status":"COMPLETE_GATED_RESCUE_CROSSED_ANALYSIS","at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"record_count":1953,"unique_complexes":217,"independent_components":68,"paired_seeds":seeds,"conditions":{},"clean":{},"trigger_rates":{"calibration":{},"development":{}},"boundaries":{"synthetic_corruption_is_real_ood":False,"validation_trigger_rate_guaranteed_by_calibration":False,"contact_labels_used_for_gate":False,"test_labels_read":False,"ood_labels_read":False,"downstream_model_forward_performed":False}}
for c in conds:
 out["conditions"][c]={"gated_minus_corrupted":crossed("gated_corrupted_ap","corrupted_ap",c),"unconditional_minus_corrupted":crossed("rescued_ap","corrupted_ap",c),"gated_minus_unconditional":crossed("gated_corrupted_ap","rescued_ap",c),"no_rescue_reference":"corrupted_ap"}
out["clean"]={"gated_minus_aligned":crossed_clean("gated_clean_ap","aligned_ap"),"unconditional_minus_aligned":crossed_clean("clean_rescued_ap","aligned_ap"),"gated_minus_unconditional":crossed_clean("gated_clean_ap","clean_rescued_ap")}
for d in ds:
 s=str(d["rescue_seed"]);out["trigger_rates"]["calibration"][s]={"clean":d["calibration"]["clean_trigger_rate"],"corruption":d["calibration"]["corruption_trigger_rate"],"threshold":d["calibration"]["threshold"]}
 q=[r for r in clean if str(r["rescue_seed"])==s];out["trigger_rates"]["development"][s]={"clean":float(np.mean([r["apply_rescue_clean"] for r in q])),"corruption":{c:float(np.mean([r["apply_rescue_corrupted"] for r in rec if str(r["rescue_seed"])==s and r["condition"]==c])) for c in conds}}
# calibration shift ratios/deltas
out["trigger_shift"]={s:{"clean_absolute_delta":out["trigger_rates"]["development"][s]["clean"]-out["trigger_rates"]["calibration"][s]["clean"],"clean_ratio":out["trigger_rates"]["development"][s]["clean"]/out["trigger_rates"]["calibration"][s]["clean"]} for s in out["trigger_rates"]["development"]}
P=R/"GATED_RESCUE_CROSSED_ANALYSIS.json";P.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(hashlib.sha256(P.read_bytes()).hexdigest());print(json.dumps({"same":out["conditions"]["same_aa_shuffle:0"],"clean":out["clean"],"trigger":out["trigger_rates"],"shift":out["trigger_shift"]},indent=2))
