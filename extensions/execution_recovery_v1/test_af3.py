import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import af3

SEEDS=[42,100,18,37,79]

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.seq='ACDEFGHIK'
        self.payload={'name':'af3_unit','dialect':'alphafold3','version':1,
            'modelSeeds':SEEDS,'sequences':[{'protein':{'id':['A'],'sequence':self.seq,
            'pairedMsa':'','unpairedMsa':'','templates':[]}}]}
        self.input=self.root/'original.json'; self.save_input()
        self.plan={'tasks':[self.task], 'run_root':str(self.root/'runs'),
                   'legacy_plan_sha256':'a'*64,'model_seeds':SEEDS,'samples_per_seed':5,
                   'runner':'/runner.sh','runtime_environment':{}}
    def save_input(self):
        self.input.write_text(json.dumps(self.payload))
        self.task={'input_path':str(self.input),'input_sha256':af3.digest(self.input),
                   'sequence':self.seq,'length':len(self.seq),'name':'af3_unit',
                   'sequence_sha256':hashlib.sha256(self.seq.encode()).hexdigest(),'task_id':0}
    def valid_outputs(self):
        result=af3.task_dir(self.plan,0)/'results'/'af3_unit'; result.mkdir(parents=True)
        rows=[]
        for s in SEEDS:
            for i in range(5):
                d=result/('seed-%d_sample-%d'%(s,i)); d.mkdir()
                score=.831234 + i*.001 + s*.00001
                (d/'model.cif').write_bytes(('%s-%s'%(s,i)).encode())
                (d/'summary_confidences.json').write_text(json.dumps({'ranking_score':round(score,2)}))
                (d/'confidences.json').write_text('{}')
                rows.append((s,i,score))
        with (result/'ranking_scores.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['seed','sample','ranking_score']); w.writerows(rows)
        s,i,_=max(rows,key=lambda x:x[2])
        (result/'af3_unit_model.cif').write_bytes((result/('seed-%d_sample-%d'%(s,i))/'model.cif').read_bytes())
        (result/'af3_unit_data.json').write_text(json.dumps(self.payload))
        return result
    def test_input_identity(self):
        self.assertEqual(af3.check_input(self.task,SEEDS),self.payload)
    def test_input_hash_drift(self):
        self.input.write_text('{}')
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_msa_rejected(self):
        self.payload['sequences'][0]['protein']['unpairedMsa']='>x\nACD'; self.save_input()
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_template_rejected(self):
        self.payload['sequences'][0]['protein']['templates']=[{}]; self.save_input()
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_seed_change_rejected(self):
        self.payload['modelSeeds']=[42]; self.save_input()
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_foreign_protein_field(self):
        self.payload['sequences'][0]['protein']['modifications']=[{}]; self.save_input()
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_multiple_chains_rejected(self):
        self.payload['sequences']*=2; self.save_input()
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_unsafe_name(self):
        self.task['name']='../evil'
        with self.assertRaises(ValueError): af3.check_input(self.task,SEEDS)
    def test_task_index_validation(self):
        for value in (-1,1,True,'0'):
            with self.assertRaises(ValueError): af3.task_dir(self.plan,value)
    def test_staging_independent_exact_copy(self):
        old=af3.digest(self.input); d=af3.stage_task(self.plan,0)
        new=d/'inputs/original.json'
        self.assertEqual(af3.digest(new),old)
        self.assertNotEqual(new.stat().st_ino,self.input.stat().st_ino)
        self.assertTrue((d/'START.json').is_file())
    def test_staging_never_overwrites(self):
        af3.stage_task(self.plan,0)
        with self.assertRaises(FileExistsError): af3.stage_task(self.plan,0)
    def test_json_exclusive(self):
        p=self.root/'a.json'; af3.new_json(p,{'a':1})
        with self.assertRaises(FileExistsError): af3.new_json(p,{'a':2})
        self.assertEqual(af3.read_json(p),{'a':1})
    def test_duplicate_json_keys(self):
        p=self.root/'a.json'; p.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): af3.read_json(p)
    def test_nonfinite_json(self):
        p=self.root/'a.json'; p.write_text('{"a":NaN}')
        with self.assertRaises(ValueError): af3.read_json(p)
    def test_complete_25_sample_grid(self):
        r=self.valid_outputs()
        self.assertEqual(len(af3.sample_grid(r/'ranking_scores.csv',SEEDS,5)),25)
    def test_same_count_duplicate_grid_fails(self):
        r=self.valid_outputs(); p=r/'ranking_scores.csv'; lines=p.read_text().splitlines()
        lines[2]=lines[1]; p.write_text('\n'.join(lines))
        with self.assertRaises(ValueError): af3.sample_grid(p,SEEDS,5)
    def test_missing_grid_fails(self):
        r=self.valid_outputs(); p=r/'ranking_scores.csv'; lines=p.read_text().splitlines()
        p.write_text('\n'.join(lines[:-1]))
        with self.assertRaises(ValueError): af3.sample_grid(p,SEEDS,5)
    def test_rounded_summary_does_not_affect_selection(self):
        self.valid_outputs()
        with patch.object(af3,'protein_backbone',side_effect=lambda path,seq:{'residues':9,'backbone_atoms':36,'backbone_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}):
            a=af3.audit_task(self.plan,0)
        self.assertEqual((a['selected']['seed'],a['selected']['sample']),(100,4))
        self.assertEqual(a['sample_count'],25)

    def test_top_level_same_structure_different_bytes_is_admitted(self):
        r=self.valid_outputs()
        maximum=max(af3.sample_grid(r/'ranking_scores.csv',SEEDS,5))
        # Re-serialize top bytes while preserving a canonical structure identity.
        top=r/'af3_unit_model.cif'; top.write_bytes(b'different serialization of same structure')
        selected=r/'seed-100_sample-4/model.cif'
        def backbone(path,seq):
            raw=Path(path).read_bytes()
            identity=(selected.read_bytes() if Path(path)==top else raw)
            return {'residues':9,'backbone_atoms':36,'backbone_sha256':hashlib.sha256(identity).hexdigest()}
        with patch.object(af3,'protein_backbone',side_effect=backbone):
            a=af3.audit_task(self.plan,0)
        self.assertFalse(a['top_model_byte_identical_to_selected_sample'])
        self.assertEqual((a['selected']['seed'],a['selected']['sample']),(100,4))

    def test_foreign_top_model_fails(self):
        r=self.valid_outputs(); (r/'af3_unit_model.cif').write_bytes(b'not a sample')
        with patch.object(af3,'protein_backbone',side_effect=lambda path,seq:{'residues':9,'backbone_atoms':36,'backbone_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}), self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_missing_confidence_payload(self):
        r=self.valid_outputs(); (r/'seed-42_sample-0/confidences.json').unlink()
        with patch.object(af3,'protein_backbone',side_effect=lambda path,seq:{'residues':9,'backbone_atoms':36,'backbone_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}), self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_timestamped_result_directory_rejected(self):
        r=self.valid_outputs(); (r.parent/'af3_unit_timestamp').mkdir()
        with self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_processed_input_drift(self):
        r=self.valid_outputs(); self.payload['modelSeeds']=[9]
        (r/'af3_unit_data.json').write_text(json.dumps(self.payload))
        with patch.object(af3,'protein_backbone',side_effect=lambda path,seq:{'residues':9,'backbone_atoms':36,'backbone_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}), self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_empty_task_list(self):
        with self.assertRaises(ValueError): af3.validate_task_set([],40)
    def test_duplicate_task_list(self):
        with self.assertRaises(ValueError): af3.validate_task_set([1,1],40)
    def test_submit_persists_job_and_rejects_repeat(self):
        (self.root/'submissions').mkdir(); (self.root/'logs').mkdir()
        responses=[SimpleNamespace(stdout='',returncode=0),SimpleNamespace(stdout='123;cluster\n',stderr='',returncode=0)]
        with patch.object(af3,'load_plan',return_value=self.plan), patch.object(af3.subprocess,'run',side_effect=responses):
            af3.submit(self.root/'PLAN.json','a'*64,[0])
        self.assertEqual(af3.read_json(self.root/'task0_SUBMITTED.json')['job_id'],'123')
        with patch.object(af3,'load_plan',return_value=self.plan), self.assertRaises(FileExistsError):
            af3.submit(self.root/'PLAN.json','a'*64,[0])
    def test_ambiguous_acknowledgement_stays_locked(self):
        (self.root/'submissions').mkdir(); (self.root/'logs').mkdir()
        responses=[SimpleNamespace(stdout='',returncode=0),SimpleNamespace(stdout='network lost',stderr='',returncode=0)]
        with patch.object(af3,'load_plan',return_value=self.plan), patch.object(af3.subprocess,'run',side_effect=responses), self.assertRaises(RuntimeError):
            af3.submit(self.root/'PLAN.json','a'*64,[0])
        self.assertTrue((self.root/'submissions/task_000.json').is_file())
        self.assertFalse((self.root/'task0_SUBMITTED.json').exists())
    def test_active_same_plan_prevents_submission(self):
        response=SimpleNamespace(stdout='9_0|ab-af3r-aaaaaaaa|RUNNING\n',returncode=0)
        with patch.object(af3,'load_plan',return_value=self.plan), patch.object(af3.subprocess,'run',return_value=response), self.assertRaises(RuntimeError):
            af3.submit(self.root/'PLAN.json','a'*64,[0])
    def test_no_prediction_outside_scheduler(self):
        with patch.object(af3,'load_plan',return_value=self.plan), patch.dict(af3.os.environ,{},clear=True), self.assertRaises(RuntimeError):
            af3.run_task(self.root/'PLAN.json','a'*64,0)
    def test_requeue_rejected(self):
        with patch.object(af3,'load_plan',return_value=self.plan), patch.dict(af3.os.environ,{'SLURM_JOB_ID':'1','SLURM_RESTART_COUNT':'1'},clear=True), self.assertRaises(RuntimeError):
            af3.run_task(self.root/'PLAN.json','a'*64,0)

if __name__=='__main__': unittest.main(verbosity=2)
