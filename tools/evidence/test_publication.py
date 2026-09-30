#!/usr/bin/env python3
"""Offline regression checks for the publication layer; not model tests."""
import json
import tempfile
import unittest
from pathlib import Path
from core import ROOT, descend, equivalent, extract, load_registry, sha256
from render import products
from verify import local_links
from checkout_experiment import plan_command


class ExtractionTests(unittest.TestCase):
    def test_exact_scalar(self):
        self.assertEqual(descend({'a':{'b':2}},['a','b']),2)

    def test_unique_selector(self):
        self.assertEqual(descend([{'arm':'a','x':1},{'arm':'b','x':2}],[{'where':{'arm':'b'}},'x']),2)

    def test_ambiguous_selector_fails(self):
        with self.assertRaises(ValueError):
            descend([{'arm':'a'},{'arm':'a'}],[{'where':{'arm':'a'}}])

    def test_missing_selector_fails(self):
        with self.assertRaises(ValueError):
            descend([{'arm':'a'}],[{'where':{'arm':'b'}}])

    def test_wildcard(self):
        self.assertEqual(descend([{'x':1},{'x':2}],['*','x']),[1,2])

    def test_wildcard_rejects_object(self):
        with self.assertRaises(ValueError): descend({'x':1},['*'])

    def test_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json'; p.write_text('{"x":1}')
            record={'source':'source.json','source_sha256':sha256(p),'selector':['x']}
            self.assertEqual(extract(record,Path(d)),1)
            p.write_text('{"x":2}')
            with self.assertRaises(ValueError): extract(record,Path(d))

    def test_reduction(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json'; p.write_text('{"x":[1,3]}')
            record={'source':'source.json','source_sha256':sha256(p),'selector':['x'],'reduce':'mean'}
            self.assertEqual(extract(record,Path(d)),2)

    def test_no_nan_inf_comparisons(self):
        self.assertFalse(equivalent(float('nan'),float('nan')))
        self.assertFalse(equivalent(float('inf'),float('inf')))

    def test_vector_length_must_match(self):
        self.assertFalse(equivalent([1,2],[1]))


class RepositoryTests(unittest.TestCase):
    def test_registry_has_boundaries_and_values(self):
        registry,values=load_registry()
        self.assertEqual(len(values),59)
        self.assertTrue(all(r['source_sha256'] for r in registry['metrics']))

    def test_known_negative_evidence_retained(self):
        _,v=load_registry()
        self.assertLess(v['holdout_ESM-IF1_ci'][0],0)
        self.assertGreater(v['holdout_ESM-IF1_ci'][1],0)
        self.assertLess(v['rep_random_ci'][0],0)

    def test_generated_files_exact(self):
        for path,text in products().items():
            with self.subTest(path=path): self.assertEqual((ROOT/path).read_text(),text)

    def test_same_cohort_chart_zero_origin(self):
        svg=products()['figures/evidence/representation_transfer.svg']
        self.assertIn('The horizontal axis starts at zero',svg)
        self.assertIn('same 10 seeds',svg)
        self.assertIn('Not a model leaderboard',svg)

    def test_front_page_links(self):
        self.assertEqual(local_links(ROOT/'README.md'),[])

    def test_checkout_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError): plan_command('multimodel',d)

    def test_checkout_uses_exact_commit_and_no_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            command,entry=plan_command('multimodel',Path(d)/'new')
            self.assertEqual(command[-1],entry['commit'])
            self.assertIn('--detach',command)
            self.assertNotIn('sbatch',command)

    def test_unknown_experiment_fails(self):
        with self.assertRaises(ValueError): plan_command('not-real','/tmp/not-real-worktree')

    def test_broken_local_link_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'README.md'; p.write_text('[x](missing.md)')
            self.assertEqual(local_links(p),['missing.md'])

if __name__=='__main__': unittest.main(verbosity=2)
