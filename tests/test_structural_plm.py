from __future__ import annotations
import json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extensions.frontier_multimodel_v1.structural_plm import load_units,selected,windows

@pytest.mark.parametrize("length,max_residues,overlap",[(1,1000,128),(1000,1000,128),(1001,1000,128),(2363,1000,128)])
def test_windows_cover_every_residue_without_out_of_range(length,max_residues,overlap):
    spans=windows(length,max_residues,overlap)
    assert spans[0][0]==0 and spans[-1][1]==length
    assert all(0<=a<b<=length and b-a<=max_residues for a,b in spans)
    covered=[0]*length
    for a,b in spans:
        for index in range(a,b):covered[index]+=1
    assert min(covered)>=1

def test_windows_are_deterministic_and_reject_invalid_contracts():
    assert windows(2363,1000,128)==windows(2363,1000,128)
    for args in [(10,0,0),(10,10,10),(10,10,-1)]:
        with pytest.raises(ValueError):windows(*args)

def fixture_payload():
    rows=[]
    for side in ["antibody","antigen"]:
        for number in range(30):
            rows.append({"side":side,"sequence_sha256":f"{number:064x}","sequence":"A","length":1})
    return {"format_version":"frontier_structure_units_v2","record_count":len(rows),"records":rows,"test_loaded":False,"contact_labels_loaded":False}

def test_selected_shards_are_disjoint_complete_and_side_specific():
    rows=fixture_payload()["records"]
    shards=[selected(rows,"antibody",index,7) for index in range(7)]
    keys=[row["sequence_sha256"] for shard in shards for row in shard]
    assert len(keys)==len(set(keys))==30
    assert all(row["side"]=="antibody" for shard in shards for row in shard)

def test_load_units_enforces_firewall_and_unique_side_sequence(tmp_path:Path):
    payload=fixture_payload();path=tmp_path/"units.json";path.write_text(json.dumps(payload))
    _,rows=load_units(path);assert len(rows)==60
    payload["test_loaded"]=True;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_units(path)
    payload=fixture_payload();payload["records"].append(payload["records"][0]);payload["record_count"]+=1;path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_units(path)


def test_saprot_allowed_missing_keys_are_only_non_hidden_state_heads():
    from extensions.frontier_multimodel_v1.structural_plm import SAPROT_ALLOWED_MISSING_KEYS
    assert SAPROT_ALLOWED_MISSING_KEYS == {
        "esm.embeddings.position_embeddings.weight",
        "esm.contact_head.regression.bias",
        "esm.contact_head.regression.weight",
    }
