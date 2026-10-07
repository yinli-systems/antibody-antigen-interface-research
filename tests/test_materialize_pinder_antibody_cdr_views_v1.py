from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = (
    Path(__file__).parents[1]
    / "remote_tools/abf_paircontact_v1/pinder_scale_v1/materialize_pinder_antibody_cdr_views_v1.py"
)
SPEC = spec_from_file_location("materialize_pinder_antibody_cdr_views_v1", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def numbering():
    return [((position, " "), "A") for position in range(1, 129)]


def test_extracts_three_imgt_cdr_ranges_in_order():
    observed = MODULE.extract_ordered_cdr_sequence(numbering())
    assert observed == "A" * (12 + 10 + 13)


def test_insertions_inside_cdr_are_retained():
    rows = numbering()
    rows.insert(112, ((112, "A"), "G"))
    observed = MODULE.extract_ordered_cdr_sequence(rows)
    assert len(observed) == 36
    assert "G" in observed


def test_rejects_missing_cdr_segment():
    rows = [row for row in numbering() if not 56 <= row[0][0] <= 65]
    with pytest.raises(ValueError, match="missing IMGT CDR range"):
        MODULE.extract_ordered_cdr_sequence(rows)


def test_require_domain_enforces_chain_type():
    with pytest.raises(ValueError, match="classification differs"):
        MODULE.require_domain(
            {"error": None, "chain_type": "K", "numbering": numbering()},
            {"H"},
            "heavy",
        )


class FakeAnarcii:
    def __init__(self, scfv_results):
        self.scfv_results = scfv_results

    def number(self, sequences, scfv=False):
        if scfv:
            return self.scfv_results
        return {}


def scfv_row():
    return {
        "INSTANCE": "physical-light-monomer",
        "topology": "scfv",
        "vh_seq": "A" * 128,
        "vl_seq": "",
    }


def test_scfv_physical_monomer_may_contain_one_light_domain():
    model = FakeAnarcii(
        {
            "scfv-000000-1": {
                "error": None,
                "chain_type": "K",
                "numbering": numbering(),
            }
        }
    )
    assert MODULE.numbered_domains(model, [scfv_row()]) == {
        "physical-light-monomer": ("", "A" * 35)
    }


def test_scfv_physical_monomer_rejects_duplicate_domain_type():
    model = FakeAnarcii(
        {
            key: {"error": None, "chain_type": "H", "numbering": numbering()}
            for key in ("scfv-000000-1", "scfv-000000-2")
        }
    )
    with pytest.raises(ValueError, match="scFv domain recovery differs"):
        MODULE.numbered_domains(model, [scfv_row()])
