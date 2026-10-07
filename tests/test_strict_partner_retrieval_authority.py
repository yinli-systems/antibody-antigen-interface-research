from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from strict_partner_retrieval_v1.authority import load_authorized_split


def _payloads():
    assignment={'authorized_for_training':True,'scientific_payload_sha256':'abc'}
    audit={'authorized_for_training':True,'authorized_for_nested_model_selection':True,'valid_audit_execution':True,'assignment_scientific_payload_sha256':'abc','claim_boundary':'foldwise direct exact four-view 80/80 C3/S2-like development; not global S2, transitive lineage independence, or test evaluation','authority':{'source_pool_split':'train','sealed_splits':['validation','test']},'data_firewall':{'validation_or_test_labels_read':0},'independent_gates':{k:True for k in ['all_four_views_completed_for_every_boundary','all_outer_and_inner_partition_invariants_passed','all_twelve_boundaries_present','authority_train_only_firewall_passed','builder_construction_gates_were_passed','exhaustive_alignment_accounting_passed','independent_inventory_exactly_matches_assignment','zero_direct_80_80_crossing_edges_in_every_boundary_and_view']}}
    return assignment,audit

def test_authority_accepts_only_full_gate_pass():
    a,b=_payloads()
    with TemporaryDirectory() as d:
        p=Path(d)/'a.json';q=Path(d)/'b.json';p.write_text(json.dumps(a));q.write_text(json.dumps(b));load_authorized_split(p,q)

def test_authority_rejects_any_failed_gate():
    a,b=_payloads();b['independent_gates']['zero_direct_80_80_crossing_edges_in_every_boundary_and_view']=False
    with TemporaryDirectory() as d:
        p=Path(d)/'a.json';q=Path(d)/'b.json';p.write_text(json.dumps(a));q.write_text(json.dumps(b))
        try:load_authorized_split(p,q)
        except PermissionError:pass
        else:raise AssertionError('failed split gate accepted')
