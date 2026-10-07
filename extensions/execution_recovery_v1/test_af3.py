import csv
import hashlib
import importlib.util
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
    def fake_structure(self, path, seq):
        raw=Path(path).read_bytes()
        identity=hashlib.sha256(raw).hexdigest()
        return {'residues':len(seq),'atom_count':4*len(seq),
                'backbone_atoms':4*len(seq),'backbone_sha256':identity,
                'all_atom_sha256':identity}
    def valid_outputs(self):
        result=af3.task_dir(self.plan,0)/'results'/'af3_unit'; result.mkdir(parents=True)
        rows=[]
        for s in SEEDS:
            for i in range(5):
                d=result/('seed-%d_sample-%d'%(s,i)); d.mkdir()
                score=.831234 + i*.001 + s*.00001
                (d/'model.cif').write_bytes(('%s-%s'%(s,i)).encode())
                summary={'chain_iptm':[None],'chain_pair_iptm':[[.5]],
                         'chain_pair_pae_min':[[1.0]],'chain_ptm':[.5],
                         'fraction_disordered':.2,'has_clash':0.0,'iptm':None,
                         'ptm':.5,'ranking_score':round(score,2)}
                confidence={'atom_chain_ids':['A']*(4*len(self.seq)),
                            'atom_plddts':[80.]*(4*len(self.seq)),
                            'contact_probs':[[.5]*len(self.seq) for _ in self.seq],
                            'pae':[[1.0]*len(self.seq) for _ in self.seq],
                            'token_chain_ids':['A']*len(self.seq),
                            'token_res_ids':list(range(1,len(self.seq)+1))}
                (d/'summary_confidences.json').write_text(json.dumps(summary))
                (d/'confidences.json').write_text(json.dumps(confidence))
                rows.append((s,i,score))
        with (result/'ranking_scores.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['seed','sample','ranking_score']); w.writerows(rows)
        s,i,_=max(rows,key=lambda x:x[2])
        selected=result/('seed-%d_sample-%d'%(s,i))
        (result/'af3_unit_model.cif').write_bytes((selected/'model.cif').read_bytes())
        (result/'af3_unit_summary_confidences.json').write_bytes(
            (selected/'summary_confidences.json').read_bytes())
        (result/'af3_unit_confidences.json').write_bytes(
            (selected/'confidences.json').read_bytes())
        processed={'dialect':'alphafold3','version':3,'name':'af3_unit','modelSeeds':SEEDS,
                   'bondedAtomPairs':None,'userCCD':None,
                   'sequences':[{'protein':{'id':'A','sequence':self.seq,'modifications':[],
                   'unpairedMsa':'>Original query\n'+self.seq+'\n',
                   'pairedMsa':'>Original query\n'+self.seq+'\n','templates':[]}}]}
        (result/'af3_unit_data.json').write_text(json.dumps(processed))
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
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure):
            a=af3.audit_task(self.plan,0)
        self.assertEqual((a['selected']['seed'],a['selected']['sample']),(100,4))
        self.assertEqual(a['sample_count'],25)

    def test_top_level_same_structure_different_bytes_is_admitted(self):
        r=self.valid_outputs()
        top=r/'af3_unit_model.cif'; top.write_bytes(b'different serialization of same structure')
        selected=r/'seed-100_sample-4/model.cif'
        selected_identity=hashlib.sha256(selected.read_bytes()).hexdigest()
        def structure(path,seq):
            identity=selected_identity if Path(path)==top else hashlib.sha256(Path(path).read_bytes()).hexdigest()
            return {'residues':len(seq),'atom_count':4*len(seq),
                    'backbone_atoms':4*len(seq),'backbone_sha256':identity,
                    'all_atom_sha256':identity}
        comparison={'equivalent':True,'raw_difference_keys':['_ma_model_list.model_group_name'],
                    'normalized_difference_keys':[],'top_semantic_sha256':'s','sample_semantic_sha256':'s',
                    'normalised_fields':['_ma_model_list.model_group_name']}
        with patch.object(af3,'protein_backbone',side_effect=structure), \
             patch.object(af3,'mmcif_semantic_comparison',return_value=comparison):
            a=af3.audit_task(self.plan,0)
        self.assertFalse(a['top_model_byte_identical_to_selected_sample'])
        self.assertEqual((a['selected']['seed'],a['selected']['sample']),(100,4))
        self.assertEqual(a['top_model_comparison']['raw_difference_keys'],
                         ['_ma_model_list.model_group_name'])

    @unittest.skipUnless(importlib.util.find_spec('Bio'), 'Biopython required for mmCIF semantics')
    def test_real_af3_timestamp_only_regression_is_semantically_equal(self):
        # Exact field and timestamps from preserved failure 1641607_0.
        def cif(timestamp, x='1.000', model_name='Top ranked model'):
            return ('data_af3\n#\n'
                    '_ma_model_list.data_id 1\n'
                    '_ma_model_list.model_group_id 1\n'
                    '_ma_model_list.model_group_name '
                    '"AlphaFold-beta-20231127 (3.0.1 @ '+timestamp+')"\n'
                    '_ma_model_list.model_id 1\n'
                    '_ma_model_list.model_name "'+model_name+'"\n'
                    '_ma_model_list.model_type "Ab initio model"\n'
                    '_atom_site.Cartn_x '+x+'\n'
                    '_atom_site.Cartn_y 2.000\n'
                    '_atom_site.Cartn_z 3.000\n#\n')
        selected=self.root/'selected.cif'; top=self.root/'top.cif'
        selected.write_text(cif('2026-09-30 16:40:10'))
        top.write_text(cif('2026-09-30 16:40:11'))
        result=af3.mmcif_semantic_comparison(top,selected)
        self.assertTrue(result['equivalent'])
        self.assertEqual(result['raw_difference_keys'],['_ma_model_list.model_group_name'])
        self.assertEqual(result['normalized_difference_keys'],[])
        self.assertEqual(result['top_semantic_sha256'],result['sample_semantic_sha256'])
        top.write_text(cif('2026-09-30 16:40:11',x='1.001'))
        changed=af3.mmcif_semantic_comparison(top,selected)
        self.assertFalse(changed['equivalent'])
        self.assertIn('_atom_site.Cartn_x',changed['normalized_difference_keys'])
        top.write_text(cif('2026-09-30 16:40:11',model_name='Foreign model'))
        changed=af3.mmcif_semantic_comparison(top,selected)
        self.assertFalse(changed['equivalent'])
        self.assertIn('_ma_model_list.model_name',changed['normalized_difference_keys'])

    def test_foreign_top_model_fails(self):
        r=self.valid_outputs(); (r/'af3_unit_model.cif').write_bytes(b'not a sample')
        comparison={'equivalent':False,'raw_difference_keys':['foreign'],
                    'normalized_difference_keys':['foreign'],
                    'top_semantic_sha256':'x','sample_semantic_sha256':'y',
                    'normalised_fields':[]}
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure), \
             patch.object(af3,'mmcif_semantic_comparison',return_value=comparison), \
             self.assertRaises(ValueError):
            af3.audit_task(self.plan,0)
    def test_missing_confidence_payload(self):
        r=self.valid_outputs(); (r/'seed-42_sample-0/confidences.json').unlink()
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure), self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_confidence_shape_corruption_rejected(self):
        r=self.valid_outputs(); p=r/'seed-42_sample-0/confidences.json'
        payload=json.loads(p.read_text()); payload['token_chain_ids'].pop(); p.write_text(json.dumps(payload))
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure), self.assertRaises(ValueError):
            af3.audit_task(self.plan,0)
    def test_top_confidence_payload_must_match_selected_sample(self):
        r=self.valid_outputs(); p=r/'af3_unit_summary_confidences.json'
        payload=json.loads(p.read_text()); payload['ptm']=.1; p.write_text(json.dumps(payload))
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure), self.assertRaises(ValueError):
            af3.audit_task(self.plan,0)
    def test_timestamped_result_directory_rejected(self):
        r=self.valid_outputs(); (r.parent/'af3_unit_timestamp').mkdir()
        with self.assertRaises(ValueError): af3.audit_task(self.plan,0)
    def test_processed_input_drift(self):
        r=self.valid_outputs(); self.payload['modelSeeds']=[9]
        (r/'af3_unit_data.json').write_text(json.dumps(self.payload))
        with patch.object(af3,'protein_backbone',side_effect=self.fake_structure), self.assertRaises(ValueError): af3.audit_task(self.plan,0)
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
