"""Standard-library tests for presentation integrity; no GPU or scientific reruns."""
import gzip
import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote, urlparse
from build_atlas import ATLAS, ROOT, link, select, require_value

class AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs=json.loads((ATLAS/'SOURCE_REFS.json').read_text())
        cls.cards=json.loads((ATLAS/'RESULT_CARDS.json').read_text())['cards']
        cls.receipt=json.loads((ATLAS/'VERIFICATION.json').read_text())
        cls.rows=[json.loads(x) for x in gzip.decompress((ATLAS/'FILE_VERSIONS.jsonl.gz').read_bytes()).splitlines()]
    def test_all_thirty_branch_heads(self):
        self.assertEqual(len(self.refs['branches']),30)
        self.assertEqual(len({b['name'] for b in self.refs['branches']}),30)
        self.assertTrue(all(re.fullmatch('[0-9a-f]{40}',b['sha']) for b in self.refs['branches']))
    def test_deduplicated_versions_and_reconstructed_memberships(self):
        self.assertEqual(len(self.rows),4913)
        self.assertEqual(len({(r['path'],r['blob']) for r in self.rows}),4913)
        self.assertEqual(sum(len(r['branches']) for r in self.rows),103552)
        for i,b in enumerate(self.refs['branches']):
            self.assertEqual(sum(i in r['branches'] for r in self.rows),b['files'])
    def test_all_cards_have_complete_boundaries(self):
        self.assertEqual(len(self.cards),37)
        self.assertEqual(len({c['id'] for c in self.cards}),37)
        for c in self.cards:
            for key in ['cohort','inputs','seeds','estimand','boundary','sources']:
                self.assertTrue(c[key],(c['id'],key))
    def test_newer_report_not_silently_promoted(self):
        reports=[c for c in self.cards if c['level']=='report-only']
        self.assertEqual(len(reports),8)
        self.assertTrue(all(not c.get('legacy_metric_ids') for c in reports))
        self.assertTrue(all('REVIEWER_RESPONSE_20261004.md' in c['sources'][0]['path'] for c in reports))
    def test_every_source_matches_inventory(self):
        rows={(r['path'],r['blob']):r for r in self.rows}
        for c in self.cards:
            for s in c['sources']:
                self.assertEqual(rows[s['path'],s['blob']]['sha256'],s['sha256'])
                self.assertIn('/blob/'+s['commit']+'/',s['url'])
    def test_selector_rejects_ambiguous_named_rows(self):
        with self.assertRaises(ValueError):
            select([{'x':1},{'x':1}],[{'where':{'x':1}}])
        self.assertEqual(select([{'x':1,'v':4}],[{'where':{'x':1}},'v']),4)
    def test_metric_drift_rejected(self):
        with self.assertRaises(ValueError):require_value(.1,.2)
        require_value(.1,.1)
    def test_immutable_links(self):
        with self.assertRaises(ValueError):link('main','README.md')
        self.assertIn('%20',link('a'*40,'a b.md'))
    def test_reported_metric_checks(self):
        self.assertEqual(self.receipt['legacy_metric_checks'],59)
        self.assertEqual(self.receipt['additional_metric_checks'],28)
        self.assertEqual(self.receipt['exact_metric_field_checks'],87)
    def test_generated_hashes(self):
        for p,h in self.receipt['generated_files_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),h,p)
    def test_all_atlas_local_markdown_links(self):
        for p in ATLAS.rglob('*.md'):
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if urlparse(target).scheme or target.startswith('#'):continue
                relative=unquote(target.split('#',1)[0])
                if relative:self.assertTrue((p.parent/relative).exists(),(str(p),target))
    def test_html_data_and_no_remote_scripts(self):
        text=(ATLAS/'index.html').read_text()
        data=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',text,re.S).group(1))
        self.assertEqual(len(data['rows']),4913)
        self.assertEqual(len(data['cards']),37)
        self.assertNotRegex(text,r'<script[^>]+src=')
        self.assertNotIn('fetch(',text)
    def test_complete_centered_prediction_accounting(self):
        c=next(c for c in self.cards if c['id']=='reparank-centered')
        values={tuple(x['selector']):x['actual'] for x in c['checks']}
        self.assertEqual(values['reused_pair_prediction_count',]+values['fresh_model_forward_count',],values['instance_count',])
    def test_all_structural_layer_intervals_cross_zero(self):
        c=next(c for c in self.cards if c['id']=='structural-plm');s=c['sources'][0]
        data=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show',s['commit']+':'+s['path']]))
        contrasts=data['real_vs_width_matched_control'];self.assertEqual(len(contrasts),10)
        for row in contrasts:
            low,high=row['paired_complex_component_analysis']['crossed_seed_component_ci95']
            self.assertLessEqual(low,0);self.assertGreaterEqual(high,0)
        self.assertEqual(len(data['tasks']),36)

if __name__=='__main__':unittest.main(verbosity=2)
