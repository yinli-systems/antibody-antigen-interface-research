#!/usr/bin/env python3
"""Focused contracts for PINDER full-antibody system reconstruction."""

from __future__ import annotations

import unittest

import pandas as pd

from remote_tools.abf_paircontact_v1.pinder_scale_v1 import (
    reconstruct_pinder_full_antibody_systems_v1 as reconstruction,
)


def monomer(pdb_id: str, chain: str) -> str:
    return f"{pdb_id}__{chain}1_UNDEFINED"


def edge(first: str, second: str) -> str:
    return f"{first}--{second}"


def sabdab_row(
    pdb_id: str,
    *,
    heavy: str | None,
    light: str | None,
    antigen: str | None,
    scfv: bool = False,
    heavy_ids: tuple[str, ...] = (),
    light_ids: tuple[str, ...] = (),
    antigen_ids: tuple[str, ...] = (),
    heavy_monomers: tuple[str, ...] = (),
    light_monomers: tuple[str, ...] = (),
    antigen_monomers: tuple[str, ...] = (),
) -> dict[str, object]:
    def role_tokens(values: tuple[str, ...], body: str) -> str:
        return ";".join(f"{value}-{body}" for value in values)

    return {
        "pdb_id": pdb_id,
        "Hchain": heavy,
        "Lchain": light,
        "model": 0,
        "antigen_chain": antigen,
        "scfv": scfv,
        "pinder_Hchain_ids": ";".join(heavy_ids),
        "pinder_Hchain_monomers": role_tokens(heavy_monomers, "R"),
        "pinder_Lchain_ids": ";".join(light_ids),
        "pinder_Lchain_monomers": role_tokens(light_monomers, "L"),
        "pinder_antigen_chain_ids": ";".join(antigen_ids),
        "pinder_antigen_chain_monomers": role_tokens(antigen_monomers, "L"),
    }


class FullAntibodySystemReconstructionTests(unittest.TestCase):
    def fixture(self) -> tuple[pd.DataFrame, dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        h1, l1, a1 = (monomer("1abc", chain) for chain in ("H", "L", "A"))
        hl1, ha1 = edge(h1, l1), edge(h1, a1)
        v2, a2 = monomer("2abc", "H"), monomer("2abc", "A")
        va2 = edge(v2, a2)
        f3, a3 = monomer("3abc", "F"), monomer("3abc", "A")
        fa3 = edge(f3, a3)
        h4, l4, a4 = (monomer("4abc", chain) for chain in ("H", "L", "A"))
        ha4 = edge(h4, a4)
        h5 = monomer("5abc", "H")
        rows = [
            sabdab_row(
                "1abc",
                heavy="H",
                light="L",
                antigen="A",
                heavy_ids=(hl1, ha1),
                light_ids=(hl1,),
                antigen_ids=(ha1,),
                heavy_monomers=(h1,),
                light_monomers=(l1,),
                antigen_monomers=(a1,),
            ),
            # Exact duplicate metadata must collapse onto the same physical system.
            sabdab_row(
                "1abc",
                heavy="H",
                light="L",
                antigen="A",
                heavy_ids=(hl1, ha1),
                light_ids=(hl1,),
                antigen_ids=(ha1,),
                heavy_monomers=(h1,),
                light_monomers=(l1,),
                antigen_monomers=(a1,),
            ),
            sabdab_row(
                "2abc",
                heavy="H",
                light=None,
                antigen="A",
                heavy_ids=(va2,),
                antigen_ids=(va2,),
                heavy_monomers=(v2,),
                antigen_monomers=(a2,),
            ),
            sabdab_row(
                "3abc",
                heavy="H",
                light="L",
                antigen="A",
                scfv=True,
                heavy_ids=(fa3,),
                antigen_ids=(fa3,),
                heavy_monomers=(f3,),
                antigen_monomers=(a3,),
            ),
            # Incomplete Fab must not silently become a heavy-only system.
            sabdab_row(
                "4abc",
                heavy="H",
                light="L",
                antigen="A",
                heavy_ids=(ha4,),
                antigen_ids=(ha4,),
                heavy_monomers=(h4,),
                light_monomers=(l4,),
                antigen_monomers=(a4,),
            ),
            # Overlapping antibody/antigen roles must be excluded.
            sabdab_row(
                "5abc",
                heavy="H",
                light=None,
                antigen="H",
                heavy_monomers=(h5,),
                antigen_monomers=(h5,),
            ),
            # Light-only rows are unsupported at this gate.
            sabdab_row(
                "6abc",
                heavy=None,
                light="L",
                antigen="A",
            ),
        ]
        all_edges = (hl1, ha1, va2, fa3, ha4)
        index = {
            edge_id: {
                "split": "train",
                "id": edge_id,
                "pdb_id": edge_id.split("__", 1)[0],
                "cluster_id": f"cluster-{position}",
                "cluster_id_R": f"R-{position}",
                "cluster_id_L": f"L-{position}",
            }
            for position, edge_id in enumerate(all_edges)
        }
        inventory: dict[str, dict[str, str]] = {}
        for edge_id, antibody, antigen in (
            (ha1, h1, a1),
            (va2, v2, a2),
            (fa3, f3, a3),
            (ha4, h4, a4),
        ):
            inventory[edge_id] = {
                "id": edge_id,
                "pdb_id": antibody.split("__", 1)[0],
                "antibody_monomer": antibody,
                "antigen_monomer": antigen,
            }
        return pd.DataFrame(rows), index, inventory

    def test_reconstruction_requires_physical_copy_support_by_topology(self) -> None:
        sabdab, index, inventory = self.fixture()
        systems, diagnostics = reconstruction.reconstruct_systems(
            sabdab, index, inventory
        )
        self.assertEqual(len(systems), 3)
        self.assertEqual(
            {key[0] for key in systems}, {"fab", "heavy_only_or_vhh", "scfv"}
        )
        fab = next(value for key, value in systems.items() if key[0] == "fab")
        self.assertEqual(len(fab["support_hl_edge_ids"]), 1)
        self.assertEqual(len(fab["support_heavy_antigen_edge_ids"]), 1)
        self.assertEqual(len(fab["support_light_antigen_edge_ids"]), 0)
        self.assertEqual(diagnostics["raw_system_occurrence_count"], 4)
        self.assertEqual(diagnostics["duplicate_occurrences_collapsed"], 1)
        self.assertEqual(
            diagnostics["exclusion_counts"]["fab_missing_train_hl_support"], 1
        )
        self.assertEqual(
            diagnostics["exclusion_counts"][
                "heavy_only_or_vhh_antibody_antigen_role_overlap"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["exclusion_counts"][
                "unsupported_light_only_or_no_antibody_row"
            ],
            1,
        )

    def test_manifest_keeps_absent_light_contact_support_explicit(self) -> None:
        sabdab, index, inventory = self.fixture()
        systems, _ = reconstruction.reconstruct_systems(sabdab, index, inventory)
        sequences = {
            monomer_id: "ACDEFGHIK"
            for key in systems
            for monomer_id in key[1:]
            if monomer_id
        }
        rows = reconstruction.manifest_rows(systems, sequences, index)
        fab = next(row for row in rows if row["topology"] == "fab")
        self.assertTrue(fab["support_hl_edge_ids"])
        self.assertTrue(fab["support_heavy_antigen_edge_ids"])
        self.assertEqual(fab["support_light_antigen_edge_ids"], "")
        self.assertEqual(fab["support_antibody_antigen_edge_count"], 1)
        self.assertEqual(fab["source_sabdab_row_count"], 1)
        self.assertEqual(len({row["system_id"] for row in rows}), 3)

    def test_inventory_direction_mismatch_fails_closed(self) -> None:
        sabdab, index, inventory = self.fixture()
        support_id = next(iter(inventory))
        inventory[support_id] = dict(inventory[support_id])
        inventory[support_id]["antibody_monomer"], inventory[support_id][
            "antigen_monomer"
        ] = (
            inventory[support_id]["antigen_monomer"],
            inventory[support_id]["antibody_monomer"],
        )
        with self.assertRaisesRegex(
            reconstruction.ReconstructionError, "inventory role direction differs"
        ):
            reconstruction.reconstruct_systems(sabdab, index, inventory)

    def test_role_monomer_suffix_and_sequence_key_are_strict(self) -> None:
        with self.assertRaisesRegex(
            reconstruction.ReconstructionError, "invalid role monomer token"
        ):
            reconstruction.role_monomers("1abc__H1_UNDEFINED")
        self.assertEqual(
            reconstruction.sequence_monomer_key(
                "1abc__H1_UNDEFINED-R.pdb", "receptor"
            ),
            "1abc__H1_UNDEFINED",
        )
        with self.assertRaisesRegex(
            reconstruction.ReconstructionError, "suffix differs"
        ):
            reconstruction.sequence_monomer_key(
                "1abc__H1_UNDEFINED-L.pdb", "receptor"
            )


if __name__ == "__main__":
    unittest.main()
