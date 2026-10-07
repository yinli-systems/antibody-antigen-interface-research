#!/usr/bin/env python3
"""Visualize antibody-antigen interface predictions.

This script reads evaluation prediction parquet files (node_predictions.parquet,
pair_predictions.parquet), maps prediction probabilities into the B-factor column
of PDB structures, generates PyMOL batch scripts for 3D visualization, and
produces an HTML summary report.

Features:
1. PDB B-factor coloring:
   - Parses ATOM/HETATM records using robust fixed-column PDB format (cols 61-66).
   - Separates epitope (antigen) and paratope (antibody) chains.
   - Maps prediction probabilities (0.0 to 1.0) into B-factor columns.
   - Generates {instance_id}_epitope.pdb and {instance_id}_paratope.pdb.
2. PyMOL script generation:
   - Produces {instance_id}.pml script with automated dual-chain coloring and surface styling.
3. HTML summary report:
   - Self-contained, responsive report.html with global KPI summary.
   - Full instance table with mean scores, positive counts, and PDB mapping status.
   - Top-K ranked residue predictions for epitope and paratope with color-coded confidence.
   - Top-K contact pair predictions (when pair_predictions.parquet is provided).
   - Copyable PyMOL visualization commands and live table search filter.
4. Graceful handling:
   - Skips missing PDB structures with an informative warning without halting execution.
   - Automatically handles varying column naming conventions across evaluation pipelines.
   - Fallback chain identification and robust residue index alignment.
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visualize_predictions")

# ---------------------------------------------------------------------------
# Column Alias Dictionaries for Graceful Parquet Schema Handling
# ---------------------------------------------------------------------------

NODE_COLUMN_ALIASES: dict[str, list[str]] = {
    "instance_id": ["instance_id", "instance", "id", "complex_id", "pdb_id", "target_id", "entry_id"],
    "residue_index": ["residue_index", "res_idx", "residue_idx", "res_id", "resi", "residue", "pos", "position"],
    "role": ["role", "type", "chain_role", "entity", "region", "subunit"],
    "label": ["label", "target", "ground_truth", "truth", "y_true", "is_contact"],
    "score": ["score", "prob", "probability", "pred", "prediction", "y_pred", "logit_prob"],
    "chain": ["chain", "chain_id", "chain_name", "asym_id", "auth_asym_id"],
}

PAIR_COLUMN_ALIASES: dict[str, list[str]] = {
    "instance_id": ["instance_id", "instance", "id", "complex_id", "pdb_id", "target_id", "entry_id"],
    "antigen_index": ["antigen_index", "ag_res_idx", "antigen_res_idx", "ag_idx", "res_idx_ag", "ag_res", "i"],
    "antibody_index": ["antibody_index", "ab_res_idx", "antibody_res_idx", "ab_idx", "res_idx_ab", "ab_res", "j"],
    "label": ["label", "target", "ground_truth", "truth", "y_true", "is_contact"],
    "score": ["score", "prob", "probability", "pred", "prediction", "y_pred"],
}


def find_canonical_column(df_columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """Find a column in DataFrame that matches one of the candidate aliases (case-insensitive)."""
    normalized_cols = {re.sub(r"[\s_-]+", "", col.lower()): col for col in df_columns}
    for alias in aliases:
        norm_alias = re.sub(r"[\s_-]+", "", alias.lower())
        if norm_alias in normalized_cols:
            return normalized_cols[norm_alias]
    return None


def normalize_node_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize node predictions DataFrame into standard schema."""
    resolved: dict[str, str] = {}
    for canonical, aliases in NODE_COLUMN_ALIASES.items():
        matched = find_canonical_column(df.columns, aliases)
        if matched:
            resolved[canonical] = matched

    # Required columns
    for req in ("instance_id", "residue_index", "score"):
        if req not in resolved:
            raise ValueError(
                f"Required column '{req}' not found in node predictions. "
                f"Available columns: {list(df.columns)}"
            )

    normalized = pd.DataFrame()
    normalized["instance_id"] = df[resolved["instance_id"]].astype(str).str.strip()
    normalized["residue_index"] = pd.to_numeric(df[resolved["residue_index"]], errors="coerce").fillna(-1).astype(int)
    normalized["score"] = pd.to_numeric(df[resolved["score"]], errors="coerce").fillna(0.0).astype(float)
    normalized["score"] = normalized["score"].clip(0.0, 1.0)

    # Role column
    if "role" in resolved:
        raw_role = df[resolved["role"]].astype(str).str.strip().str.lower()
        # Canonicalize role names
        canonical_role = raw_role.apply(
            lambda r: "epitope" if "epi" in r or "ag" in r or "antigen" in r else ("paratope" if "para" in r or "ab" in r or "antibody" in r else r)
        )
        normalized["role"] = canonical_role
    else:
        logger.warning("No 'role' column detected in node predictions; defaulting all roles to 'epitope'")
        normalized["role"] = "epitope"

    # Optional label column
    if "label" in resolved:
        normalized["label"] = pd.to_numeric(df[resolved["label"]], errors="coerce").fillna(0).astype(int)
    else:
        normalized["label"] = None

    # Optional chain column
    if "chain" in resolved:
        normalized["chain"] = df[resolved["chain"]].astype(str).str.strip()
    else:
        normalized["chain"] = None

    return normalized


def normalize_pair_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pair predictions DataFrame into standard schema."""
    resolved: dict[str, str] = {}
    for canonical, aliases in PAIR_COLUMN_ALIASES.items():
        matched = find_canonical_column(df.columns, aliases)
        if matched:
            resolved[canonical] = matched

    # Required columns
    for req in ("instance_id", "antigen_index", "antibody_index", "score"):
        if req not in resolved:
            raise ValueError(
                f"Required column '{req}' not found in pair predictions. "
                f"Available columns: {list(df.columns)}"
            )

    normalized = pd.DataFrame()
    normalized["instance_id"] = df[resolved["instance_id"]].astype(str).str.strip()
    normalized["antigen_index"] = pd.to_numeric(df[resolved["antigen_index"]], errors="coerce").fillna(-1).astype(int)
    normalized["antibody_index"] = pd.to_numeric(df[resolved["antibody_index"]], errors="coerce").fillna(-1).astype(int)
    normalized["score"] = pd.to_numeric(df[resolved["score"]], errors="coerce").fillna(0.0).astype(float)
    normalized["score"] = normalized["score"].clip(0.0, 1.0)

    if "label" in resolved:
        normalized["label"] = pd.to_numeric(df[resolved["label"]], errors="coerce").fillna(0).astype(int)
    else:
        normalized["label"] = None

    return normalized


# ---------------------------------------------------------------------------
# PDB Data Structures and Plain-Text Parser
# ---------------------------------------------------------------------------

@dataclass
class AtomRecord:
    """Represents a single ATOM or HETATM record from a PDB file."""
    record_type: str  # "ATOM  " or "HETATM"
    serial: int
    name: str
    alt_loc: str
    res_name: str
    chain_id: str
    res_seq: int
    ins_code: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str
    charge: str
    raw_line: str
    model: int = 1


def parse_pdb_file(pdb_path: Path) -> list[AtomRecord]:
    """Parse ATOM and HETATM records line-by-line from a PDB file.

    PDB Column Format (1-based):
      1- 6: Record name "ATOM  " / "HETATM"
      7-11: Atom serial number
     13-16: Atom name
        17: Alternate location indicator
     18-20: Residue name
        22: Chain identifier
     23-26: Residue sequence number
        27: Code for insertion of residues
     31-38: Orthogonal coordinates for X in Angstroms
     39-46: Orthogonal coordinates for Y in Angstroms
     47-54: Orthogonal coordinates for Z in Angstroms
     55-60: Occupancy
     61-66: Temperature factor (B-factor)
     77-78: Element symbol
     79-80: Charge
    """
    records: list[AtomRecord] = []
    current_model = 1
    has_models = False

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                has_models = True
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    current_model = int(parts[1])
                continue
            if line.startswith("ENDMDL"):
                if has_models and current_model == 1:
                    # Multi-model structure: retain first model
                    break
                continue
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            # Need coordinates at least up to column 54
            line_len = len(line.rstrip("\r\n"))
            if line_len < 54:
                continue

            record_type = line[0:6]
            serial_str = line[6:11].strip() if line_len >= 11 else ""
            serial = int(serial_str) if serial_str.isdigit() else len(records) + 1
            name = line[12:16] if line_len >= 16 else "    "
            alt_loc = line[16:17] if line_len >= 17 else " "
            res_name = line[17:20].strip() if line_len >= 20 else "UNK"
            chain_id = line[21:22] if line_len >= 22 else " "
            res_seq_str = line[22:26].strip() if line_len >= 26 else ""
            try:
                res_seq = int(res_seq_str) if res_seq_str else 0
            except ValueError:
                res_seq = 0
            ins_code = line[26:27] if line_len >= 27 else " "

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            try:
                occupancy = float(line[54:60].strip()) if line_len >= 60 and line[54:60].strip() else 1.0
            except ValueError:
                occupancy = 1.0

            try:
                b_factor = float(line[60:66].strip()) if line_len >= 66 and line[60:66].strip() else 0.0
            except ValueError:
                b_factor = 0.0

            element = line[76:78].strip() if line_len >= 78 else ""
            charge = line[78:80].strip() if line_len >= 80 else ""

            records.append(
                AtomRecord(
                    record_type=record_type,
                    serial=serial,
                    name=name,
                    alt_loc=alt_loc,
                    res_name=res_name,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    ins_code=ins_code,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    b_factor=b_factor,
                    element=element,
                    charge=charge,
                    raw_line=line,
                    model=current_model,
                )
            )

    return records


def format_atom_with_bfactor(atom: AtomRecord, new_b_factor: float) -> str:
    """Format an ATOM/HETATM line with new B-factor in columns 61-66 (0-indexed 60:66)."""
    clamped = max(0.0, min(999.99, float(new_b_factor)))
    b_str = f"{clamped:6.2f}"

    raw = atom.raw_line.rstrip("\r\n")
    if len(raw) < 60:
        raw = raw.ljust(60)

    prefix = raw[:60]
    suffix = raw[66:] if len(raw) >= 66 else ""
    return f"{prefix}{b_str}{suffix}\n"


# ---------------------------------------------------------------------------
# Chain and Residue Mapping Engine
# ---------------------------------------------------------------------------

def infer_chains(
    atoms: Sequence[AtomRecord],
    epitope_indices: Set[int],
    paratope_indices: Set[int],
    explicit_epitope_chains: Optional[Sequence[str]] = None,
    explicit_paratope_chains: Optional[Sequence[str]] = None,
    role_by_chain: Optional[Mapping[str, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Determine which PDB chains represent epitope (antigen) vs paratope (antibody).

    Employs hierarchical resolution:
    1. Explicit CLI chains (--epitope-chains, --paratope-chains)
    2. Explicit chain annotations in node_predictions
    3. Standard antibody chain identifiers (H, L, V, etc.)
    4. Residue index overlap / sequence length matching
    5. Fallback heuristics
    """
    pdb_chains = list(dict.fromkeys(a.chain_id for a in atoms))
    if not pdb_chains:
        return ["A"], ["B"]

    if explicit_epitope_chains and explicit_paratope_chains:
        return [c for c in pdb_chains if c in explicit_epitope_chains], [
            c for c in pdb_chains if c in explicit_paratope_chains
        ]

    if role_by_chain:
        epi = [c for c, r in role_by_chain.items() if r == "epitope" and c in pdb_chains]
        para = [c for c, r in role_by_chain.items() if r == "paratope" and c in pdb_chains]
        if epi and para:
            return epi, para

    # Standard antibody chain identifiers
    antibody_names = {"H", "L", "V", "h", "l", "v", "A_B"}
    ab_chains = [c for c in pdb_chains if c in antibody_names]
    ag_chains = [c for c in pdb_chains if c not in antibody_names]

    if ab_chains and ag_chains:
        return ag_chains, ab_chains

    # Overlap with prediction residue indices
    chain_res_seqs: dict[str, set[int]] = {}
    for atom in atoms:
        chain_res_seqs.setdefault(atom.chain_id, set()).add(atom.res_seq)

    if len(pdb_chains) >= 2:
        # Score chain pair assignments
        c1, c2 = pdb_chains[0], pdb_chains[1]
        seqs_1 = chain_res_seqs.get(c1, set())
        seqs_2 = chain_res_seqs.get(c2, set())

        # Match direct res_seq overlap
        match_1_as_epi = len(seqs_1 & epitope_indices) + len(seqs_2 & paratope_indices)
        match_2_as_epi = len(seqs_2 & epitope_indices) + len(seqs_1 & paratope_indices)

        if match_1_as_epi > match_2_as_epi:
            return [c1], [c for c in pdb_chains if c != c1]
        elif match_2_as_epi > match_1_as_epi:
            return [c2], [c for c in pdb_chains if c != c2]

        # Size heuristic: antigens are usually longer than antibody variable fragments
        if len(seqs_1) >= len(seqs_2):
            return [c1], [c for c in pdb_chains if c != c1]
        else:
            return [c2], [c for c in pdb_chains if c != c2]

    # Single-chain structure fallback
    return pdb_chains, pdb_chains


def resolve_residue_scores(
    chain_atoms: Sequence[AtomRecord],
    prediction_scores: Mapping[int, float],
) -> Dict[Tuple[int, str], float]:
    """Map residue prediction scores to chain residues (res_seq, ins_code).

    Supports 3 indexing strategies:
    1. Direct res_seq matching: index matches PDB res_seq
    2. 0-based sequential offset: index 0, 1, 2, ... matches chain residue order
    3. 1-based sequential offset: index 1, 2, 3, ... matches chain residue order
    """
    unique_residues: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for a in chain_atoms:
        key = (a.res_seq, a.ins_code)
        if key not in seen:
            seen.add(key)
            unique_residues.append(key)

    if not unique_residues or not prediction_scores:
        return {}

    pred_keys = set(prediction_scores.keys())

    # Strategy 1: Direct res_seq matching
    matches_direct = sum(1 for res_seq, _ in unique_residues if res_seq in pred_keys)

    # Strategy 2: 0-based sequential
    matches_zero_based = sum(1 for idx in range(len(unique_residues)) if idx in pred_keys)

    # Strategy 3: 1-based sequential
    matches_one_based = sum(1 for idx in range(len(unique_residues)) if (idx + 1) in pred_keys)

    best_strategy = "direct"
    max_matches = matches_direct

    if matches_zero_based > max_matches:
        best_strategy = "zero_based"
        max_matches = matches_zero_based
    if matches_one_based > max_matches:
        best_strategy = "one_based"
        max_matches = matches_one_based

    res_score_map: dict[tuple[int, str], float] = {}
    for idx, key in enumerate(unique_residues):
        res_seq, _ = key
        if best_strategy == "direct":
            score = prediction_scores.get(res_seq, 0.0)
        elif best_strategy == "zero_based":
            score = prediction_scores.get(idx, 0.0)
        else:  # one_based
            score = prediction_scores.get(idx + 1, 0.0)
        res_score_map[key] = score

    return res_score_map


def write_colored_pdb(
    atoms: Sequence[AtomRecord],
    target_chains: Sequence[str],
    res_score_map_by_chain: Mapping[str, Mapping[Tuple[int, str], float]],
    output_pdb_path: Path,
) -> int:
    """Write an output PDB containing target_chains with B-factors replaced by prediction probabilities.

    Returns the count of ATOM/HETATM records written.
    """
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    target_chain_set = set(target_chains)
    written_count = 0
    last_chain: Optional[str] = None

    with output_pdb_path.open("w", encoding="utf-8") as out:
        out.write(f"REMARK 900 CREATED BY VISUALIZE_PREDICTIONS.PY\n")
        out.write(f"REMARK 900 B-FACTOR REPRESENTS PREDICTED INTERFACE PROBABILITY [0.0 - 1.0]\n")

        for atom in atoms:
            if atom.chain_id not in target_chain_set:
                continue

            if last_chain is not None and atom.chain_id != last_chain:
                out.write("TER\n")

            chain_scores = res_score_map_by_chain.get(atom.chain_id, {})
            score = chain_scores.get((atom.res_seq, atom.ins_code), 0.0)

            out_line = format_atom_with_bfactor(atom, score)
            out.write(out_line)
            written_count += 1
            last_chain = atom.chain_id

        if written_count > 0:
            out.write("TER\n")
            out.write("END\n")

    return written_count


# ---------------------------------------------------------------------------
# PyMOL Script Generator
# ---------------------------------------------------------------------------

def write_pymol_script(
    instance_id: str,
    output_pml_path: Path,
    epitope_pdb_filename: str,
    paratope_pdb_filename: str,
) -> None:
    """Generate PyMOL .pml visualization script for an instance."""
    output_pml_path.parent.mkdir(parents=True, exist_ok=True)
    script_content = f"""load {epitope_pdb_filename}, epitope
load {paratope_pdb_filename}, paratope
spectrum b, blue_white_red, epitope, minimum=0, maximum=1
spectrum b, blue_white_red, paratope, minimum=0, maximum=1
set cartoon_transparency, 0.3
show cartoon
show surface, epitope
set surface_color, white
set transparency, 0.7
orient
"""
    output_pml_path.write_text(script_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML Summary Report Generator
# ---------------------------------------------------------------------------

@dataclass
class InstanceReportData:
    """Container for instance metrics and top predictions."""
    instance_id: str
    pdb_found: bool
    epitope_pdb_name: Optional[str]
    paratope_pdb_name: Optional[str]
    pml_name: Optional[str]
    mean_epitope_score: float
    mean_paratope_score: float
    epitope_positives: int
    paratope_positives: int
    total_positives: int
    total_epitope_residues: int
    total_paratope_residues: int
    top_epitope: list[dict[str, Any]]
    top_paratope: list[dict[str, Any]]
    top_pairs: list[dict[str, Any]] = field(default_factory=list)
    pair_positives: int = 0
    total_pairs: int = 0


def format_confidence_badge(score: float) -> str:
    """Return HTML span with color-coded confidence badge."""
    if score >= 0.70:
        return '<span class="badge badge-high">High</span>'
    elif score >= 0.40:
        return '<span class="badge badge-med">Medium</span>'
    else:
        return '<span class="badge badge-low">Low</span>'


def format_label_badge(label: Optional[int]) -> str:
    """Return HTML span for ground truth label."""
    if label is None:
        return '<span class="label-none">&mdash;</span>'
    if label == 1:
        return '<span class="badge badge-pos">Pos (1)</span>'
    return '<span class="badge badge-neg">Neg (0)</span>'


def generate_html_report(
    instances: Sequence[InstanceReportData],
    output_html_path: Path,
    threshold: float,
    top_k: int,
    has_pair_data: bool,
) -> None:
    """Generate standalone, responsive HTML summary report."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    total_instances = len(instances)
    mapped_count = sum(1 for i in instances if i.pdb_found)
    skipped_count = total_instances - mapped_count

    avg_epi = np.mean([i.mean_epitope_score for i in instances]) if instances else 0.0
    avg_para = np.mean([i.mean_paratope_score for i in instances]) if instances else 0.0
    total_pos = sum(i.total_positives for i in instances)

    # HTML Header and embedded styling
    html_lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "  <title>Antibody-Antigen Interface Prediction Report</title>",
        "  <style>",
        """
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --high-conf: #10b981;
      --med-conf: #f59e0b;
      --low-conf: #64748b;
      --pos: #059669;
      --neg: #475569;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1400px; margin: 0 auto; }
    header {
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }
    h1 { font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 8px; }
    p.subtitle { color: var(--text-muted); font-size: 14px; }
    
    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px 20px;
    }
    .metric-title { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
    .metric-val { font-size: 26px; font-weight: 700; color: #fff; margin-top: 6px; }
    .metric-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
    
    /* Tables */
    .table-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow-x: auto;
      margin-bottom: 36px;
    }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background-color: #1a2333;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      white-space: nowrap;
    }
    td { padding: 12px 16px; border-bottom: 1px solid #283548; vertical-align: middle; }
    tr:hover { background-color: #24334a; }
    
    /* Search Bar */
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .search-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 14px;
      color: #fff;
      font-size: 13px;
      min-width: 280px;
      outline: none;
    }
    .search-box:focus { border-color: var(--accent); }
    
    /* Badges */
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .badge-high { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-med { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-low { background-color: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }
    .badge-pos { background-color: rgba(5, 150, 105, 0.3); color: #6ee7b7; }
    .badge-neg { background-color: rgba(71, 85, 105, 0.3); color: #94a3b8; }
    .badge-status-ok { background-color: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }
    .badge-status-skip { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    /* Progress Bars */
    .bar-container {
      width: 100px;
      height: 6px;
      background: #334155;
      border-radius: 3px;
      display: inline-block;
      vertical-align: middle;
      margin-left: 8px;
      overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 3px; }
    
    /* Instance Details Card */
    .instance-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      margin-bottom: 24px;
      padding: 20px;
    }
    .instance-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid #283548;
      padding-bottom: 12px;
      flex-wrap: wrap;
      gap: 8px;
    }
    .instance-title { font-size: 18px; font-weight: 700; color: var(--accent); }
    .instance-actions a {
      color: var(--accent);
      text-decoration: none;
      font-size: 12px;
      margin-left: 12px;
      font-weight: 500;
    }
    .instance-actions a:hover { text-decoration: underline; }
    
    /* PyMOL Code Box */
    .code-box {
      background: #090d16;
      border: 1px solid #1e293b;
      border-radius: 6px;
      padding: 12px 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      color: #93c5fd;
      position: relative;
      margin-bottom: 20px;
      overflow-x: auto;
      white-space: pre;
    }
    .copy-btn {
      position: absolute;
      top: 8px;
      right: 8px;
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text-muted);
      border-radius: 4px;
      font-size: 11px;
      padding: 4px 8px;
      cursor: pointer;
    }
    .copy-btn:hover { background: #334155; color: #fff; }
    
    /* Side-by-side Top-K */
    .split-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }
    .sub-card {
      background: #151d2c;
      border: 1px solid #283548;
      border-radius: 6px;
      padding: 14px;
    }
    .sub-title {
      font-size: 14px;
      font-weight: 600;
      color: #e2e8f0;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
    }
    a.inst-link { color: var(--accent); text-decoration: none; font-weight: 600; }
    a.inst-link:hover { text-decoration: underline; }
        """,
        "  </style>",
        "</head>",
        "<body>",
        '  <div class="container">',
        "    <header>",
        "      <h1>Antibody-Antigen Interface Prediction Report</h1>",
        f'      <p class="subtitle">Threshold: {threshold:.2f} | Top-K Residues: {top_k} | Generated by visualize_predictions.py</p>',
        "    </header>",
        "",
        '    <!-- Summary KPI Cards -->',
        '    <div class="metrics-grid">',
        '      <div class="metric-card">',
        '        <div class="metric-title">Total Instances</div>',
        f'        <div class="metric-val">{total_instances}</div>',
        f'        <div class="metric-sub">{mapped_count} structures mapped, {skipped_count} skipped</div>',
        "      </div>",
        '      <div class="metric-card">',
        '        <div class="metric-title">Mean Epitope Score</div>',
        f'        <div class="metric-val">{avg_epi:.3f}</div>',
        '        <div class="metric-sub">Dataset-wide average</div>',
        "      </div>",
        '      <div class="metric-card">',
        '        <div class="metric-title">Mean Paratope Score</div>',
        f'        <div class="metric-val">{avg_para:.3f}</div>',
        '        <div class="metric-sub">Dataset-wide average</div>',
        "      </div>",
        '      <div class="metric-card">',
        '        <div class="metric-title">Predicted Positives</div>',
        f'        <div class="metric-val">{total_pos}</div>',
        f'        <div class="metric-sub">Residues &ge; {threshold:.2f} threshold</div>',
        "      </div>",
        "    </div>",
        "",
        '    <!-- Overview Table Toolbar -->',
        '    <div class="toolbar">',
        '      <h2 style="font-size: 20px; font-weight: 700;">All Instances Summary</h2>',
        '      <input type="text" id="searchInput" class="search-box" placeholder="Filter instances by ID or status..." onkeyup="filterTable()">',
        "    </div>",
        "",
        '    <!-- All Instances Table -->',
        '    <div class="table-container">',
        '      <table id="summaryTable">',
        "        <thead>",
        "          <tr>",
        "            <th>#</th>",
        "            <th>Instance ID</th>",
        "            <th>PDB Status</th>",
        "            <th>Mean Epitope</th>",
        "            <th>Mean Paratope</th>",
        "            <th>Epitope Positives</th>",
        "            <th>Paratope Positives</th>",
        "            <th>Total Positives</th>",
    ]

    if has_pair_data:
        html_lines.append("            <th>Pair Positives</th>")

    html_lines.extend([
        "            <th>PyMOL Script</th>",
        "            <th>Action</th>",
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ])

    for row_idx, inst in enumerate(instances, 1):
        status_badge = (
            '<span class="badge badge-status-ok">Mapped</span>'
            if inst.pdb_found
            else '<span class="badge badge-status-skip">Missing</span>'
        )

        # Bar fill color based on score
        def bar_color(score: float) -> str:
            if score >= 0.7:
                return "#10b981"
            if score >= 0.4:
                return "#f59e0b"
            return "#64748b"

        epi_bar = f'<div class="bar-container"><div class="bar-fill" style="width: {min(100, int(inst.mean_epitope_score * 100))}%; background: {bar_color(inst.mean_epitope_score)};"></div></div>'
        para_bar = f'<div class="bar-container"><div class="bar-fill" style="width: {min(100, int(inst.mean_paratope_score * 100))}%; background: {bar_color(inst.mean_paratope_score)};"></div></div>'

        pml_cell = (
            f'<a href="{html.escape(inst.pml_name)}" download style="color: var(--accent); text-decoration: none;">{html.escape(inst.pml_name)}</a>'
            if inst.pml_name
            else '<span style="color: var(--text-muted);">&mdash;</span>'
        )

        esc_id = html.escape(inst.instance_id)
        html_lines.append("          <tr>")
        html_lines.append(f"            <td>{row_idx}</td>")
        html_lines.append(f'            <td><a class="inst-link" href="#inst-{esc_id}">{esc_id}</a></td>')
        html_lines.append(f"            <td>{status_badge}</td>")
        html_lines.append(f"            <td>{inst.mean_epitope_score:.3f} {epi_bar}</td>")
        html_lines.append(f"            <td>{inst.mean_paratope_score:.3f} {para_bar}</td>")
        html_lines.append(f"            <td>{inst.epitope_positives} / {inst.total_epitope_residues}</td>")
        html_lines.append(f"            <td>{inst.paratope_positives} / {inst.total_paratope_residues}</td>")
        html_lines.append(f"            <td><strong>{inst.total_positives}</strong></td>")

        if has_pair_data:
            html_lines.append(f"            <td>{inst.pair_positives} / {inst.total_pairs}</td>")

        html_lines.append(f"            <td>{pml_cell}</td>")
        html_lines.append(f'            <td><a href="#inst-{esc_id}" style="color: var(--accent); text-decoration: none;">View Details &darr;</a></td>')
        html_lines.append("          </tr>")

    html_lines.extend([
        "        </tbody>",
        "      </table>",
        "    </div>",
        "",
        '    <h2 style="font-size: 22px; font-weight: 700; margin-bottom: 20px;">Instance Details & Top-K Residues</h2>',
    ])

    # Per-Instance Sections
    for inst in instances:
        esc_id = html.escape(inst.instance_id)
        pml_snippet = (
            f"load {inst.epitope_pdb_name}, epitope\n"
            f"load {inst.paratope_pdb_name}, paratope\n"
            f"spectrum b, blue_white_red, epitope, minimum=0, maximum=1\n"
            f"spectrum b, blue_white_red, paratope, minimum=0, maximum=1\n"
            f"set cartoon_transparency, 0.3\n"
            f"show cartoon\n"
            f"show surface, epitope\n"
            f"set surface_color, white\n"
            f"set transparency, 0.7\n"
            f"orient"
            if inst.pdb_found
            else "# Structure file not found for this instance; PDB and PML generation skipped."
        )

        html_lines.extend([
            f'    <div class="instance-card" id="inst-{esc_id}">',
            '      <div class="instance-header">',
            f'        <div><span class="instance-title">{esc_id}</span>',
            f'          <span style="margin-left: 12px; font-size: 13px; color: var(--text-muted);">'
            f"Mean Epi: {inst.mean_epitope_score:.3f} | Mean Para: {inst.mean_paratope_score:.3f} | Positives: {inst.total_positives}</span></div>",
            '        <div class="instance-actions">',
        ])

        if inst.pdb_found:
            html_lines.append(f'          <a href="{html.escape(inst.epitope_pdb_name)}" download>&darr; Epitope PDB</a>')
            html_lines.append(f'          <a href="{html.escape(inst.paratope_pdb_name)}" download>&darr; Paratope PDB</a>')
            html_lines.append(f'          <a href="{html.escape(inst.pml_name)}" download>&darr; PyMOL .pml</a>')
        else:
            html_lines.append('          <span style="color: #f87171; font-size: 12px;">Structure Not Found</span>')

        html_lines.extend([
            "        </div>",
            "      </div>",
            "",
            '      <!-- PyMOL Script Box -->',
            '      <div class="code-box">',
            f'        <button class="copy-btn" onclick="copyCode(this)">Copy PyMOL</button><code>{html.escape(pml_snippet)}</code>',
            "      </div>",
            "",
            '      <!-- Side-by-Side Top-K Tables -->',
            '      <div class="split-grid">',
            '        <div class="sub-card">',
            f'          <div class="sub-title"><span>Top-{len(inst.top_epitope)} Epitope Residues</span><span>({inst.epitope_positives} &ge; {threshold:.2f})</span></div>',
            "          <table>",
            "            <thead>",
            "              <tr>",
            "                <th>Rank</th>",
            "                <th>Residue</th>",
            "                <th>Score</th>",
            "                <th>Confidence</th>",
            "                <th>Label</th>",
            "              </tr>",
            "            </thead>",
            "            <tbody>",
        ])

        for rank, r in enumerate(inst.top_epitope, 1):
            score_val = r["score"]
            badge = format_confidence_badge(score_val)
            lbl_badge = format_label_badge(r.get("label"))
            html_lines.append("              <tr>")
            html_lines.append(f"                <td>{rank}</td>")
            html_lines.append(f"                <td><strong>{r['residue_index']}</strong></td>")
            html_lines.append(f"                <td>{score_val:.4f}</td>")
            html_lines.append(f"                <td>{badge}</td>")
            html_lines.append(f"                <td>{lbl_badge}</td>")
            html_lines.append("              </tr>")

        if not inst.top_epitope:
            html_lines.append('              <tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No epitope predictions</td></tr>')

        html_lines.extend([
            "            </tbody>",
            "          </table>",
            "        </div>",
            "",
            '        <div class="sub-card">',
            f'          <div class="sub-title"><span>Top-{len(inst.top_paratope)} Paratope Residues</span><span>({inst.paratope_positives} &ge; {threshold:.2f})</span></div>',
            "          <table>",
            "            <thead>",
            "              <tr>",
            "                <th>Rank</th>",
            "                <th>Residue</th>",
            "                <th>Score</th>",
            "                <th>Confidence</th>",
            "                <th>Label</th>",
            "              </tr>",
            "            </thead>",
            "            <tbody>",
        ])

        for rank, r in enumerate(inst.top_paratope, 1):
            score_val = r["score"]
            badge = format_confidence_badge(score_val)
            lbl_badge = format_label_badge(r.get("label"))
            html_lines.append("              <tr>")
            html_lines.append(f"                <td>{rank}</td>")
            html_lines.append(f"                <td><strong>{r['residue_index']}</strong></td>")
            html_lines.append(f"                <td>{score_val:.4f}</td>")
            html_lines.append(f"                <td>{badge}</td>")
            html_lines.append(f"                <td>{lbl_badge}</td>")
            html_lines.append("              </tr>")

        if not inst.top_paratope:
            html_lines.append('              <tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No paratope predictions</td></tr>')

        html_lines.extend([
            "            </tbody>",
            "          </table>",
            "        </div>",
            "      </div>",
        ])

        # Top-K Contact Pairs (if available)
        if inst.top_pairs:
            html_lines.extend([
                '      <div style="margin-top: 16px;">',
                '        <div class="sub-card">',
                f'          <div class="sub-title"><span>Top-{len(inst.top_pairs)} Contact Pairs (Antigen &times; Antibody)</span><span>({inst.pair_positives} &ge; {threshold:.2f})</span></div>',
                "          <table>",
                "            <thead>",
                "              <tr>",
                "                <th>Rank</th>",
                "                <th>Antigen Res</th>",
                "                <th>Antibody Res</th>",
                "                <th>Score</th>",
                "                <th>Confidence</th>",
                "                <th>Label</th>",
                "              </tr>",
                "            </thead>",
                "            <tbody>",
            ])
            for rank, p in enumerate(inst.top_pairs, 1):
                p_score = p["score"]
                p_badge = format_confidence_badge(p_score)
                p_lbl = format_label_badge(p.get("label"))
                html_lines.append("              <tr>")
                html_lines.append(f"                <td>{rank}</td>")
                html_lines.append(f"                <td>Ag: {p['antigen_index']}</td>")
                html_lines.append(f"                <td>Ab: {p['antibody_index']}</td>")
                html_lines.append(f"                <td>{p_score:.4f}</td>")
                html_lines.append(f"                <td>{p_badge}</td>")
                html_lines.append(f"                <td>{p_lbl}</td>")
                html_lines.append("              </tr>")
            html_lines.extend([
                "            </tbody>",
                "          </table>",
                "        </div>",
                "      </div>",
            ])

        html_lines.append("    </div>")

    # JavaScript for Search Filter & Copy Button
    html_lines.extend([
        "  </div>",
        "  <script>",
        """
    function filterTable() {
      const input = document.getElementById("searchInput");
      const filter = input.value.toLowerCase();
      const table = document.getElementById("summaryTable");
      const trs = table.getElementsByTagName("tr");

      for (let i = 1; i < trs.length; i++) {
        const text = trs[i].textContent || trs[i].innerText;
        if (text.toLowerCase().indexOf(filter) > -1) {
          trs[i].style.display = "";
        } else {
          trs[i].style.display = "none";
        }
      }
    }

    function copyCode(btn) {
      const codeElem = btn.nextElementSibling;
      navigator.clipboard.writeText(codeElem.innerText).then(() => {
        const orig = btn.innerText;
        btn.innerText = "Copied!";
        btn.style.color = "#34d399";
        setTimeout(() => {
          btn.innerText = orig;
          btn.style.color = "";
        }, 2000);
      });
    }
        """,
        "  </script>",
        "</body>",
        "</html>",
    ])

    output_html_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Orchestration Pipeline
# ---------------------------------------------------------------------------

def locate_pdb_file(pdb_dir: Optional[Path], instance_id: str) -> Optional[Path]:
    """Search for structure file corresponding to instance_id in pdb_dir."""
    if not pdb_dir or not pdb_dir.exists():
        return None

    candidate_patterns = [
        f"{instance_id}.pdb",
        f"{instance_id.lower()}.pdb",
        f"{instance_id.upper()}.pdb",
        f"{instance_id}_complex.pdb",
        f"{instance_id}.ent",
    ]

    for pat in candidate_patterns:
        target = pdb_dir / pat
        if target.is_file():
            return target

    # Check subdirectory pattern {pdb_dir}/{instance_id}/{instance_id}.pdb
    sub_target = pdb_dir / instance_id / f"{instance_id}.pdb"
    if sub_target.is_file():
        return sub_target

    return None


def run_visualization(
    node_predictions_path: Path,
    output_dir: Path,
    pair_predictions_path: Optional[Path] = None,
    pdb_dir: Optional[Path] = None,
    top_k: int = 20,
    threshold: float = 0.5,
    explicit_epitope_chains: Optional[Sequence[str]] = None,
    explicit_paratope_chains: Optional[Sequence[str]] = None,
) -> None:
    """Execute complete end-to-end visualization pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read node predictions
    logger.info(f"Loading node predictions from: {node_predictions_path}")
    if not node_predictions_path.exists():
        raise FileNotFoundError(f"Node predictions file not found: {node_predictions_path}")

    node_df_raw = pd.read_parquet(node_predictions_path)
    node_df = normalize_node_predictions(node_df_raw)
    logger.info(f"Loaded {len(node_df)} node predictions across {node_df['instance_id'].nunique()} instances")

    # 2. Read pair predictions (if provided)
    pair_df: Optional[pd.DataFrame] = None
    if pair_predictions_path:
        if pair_predictions_path.exists():
            logger.info(f"Loading pair predictions from: {pair_predictions_path}")
            pair_df_raw = pd.read_parquet(pair_predictions_path)
            pair_df = normalize_pair_predictions(pair_df_raw)
            logger.info(f"Loaded {len(pair_df)} pair predictions")
        else:
            logger.warning(f"Specified pair predictions file not found: {pair_predictions_path}. Proceeding without pair data.")

    # Group node predictions by instance
    node_by_instance = {
        str(inst_id): group for inst_id, group in node_df.groupby("instance_id")
    }

    # Group pair predictions by instance
    pair_by_instance: dict[str, pd.DataFrame] = {}
    if pair_df is not None and not pair_df.empty:
        pair_by_instance = {
            str(inst_id): group for inst_id, group in pair_df.groupby("instance_id")
        }

    instance_ids = sorted(node_by_instance.keys())
    report_instances: list[InstanceReportData] = []

    # 3. Process each instance
    for instance_id in instance_ids:
        inst_node = node_by_instance[instance_id]
        epi_rows = inst_node[inst_node["role"] == "epitope"]
        para_rows = inst_node[inst_node["role"] == "paratope"]

        mean_epi_score = float(epi_rows["score"].mean()) if not epi_rows.empty else 0.0
        mean_para_score = float(para_rows["score"].mean()) if not para_rows.empty else 0.0

        epi_positives = int((epi_rows["score"] >= threshold).sum())
        para_positives = int((para_rows["score"] >= threshold).sum())
        total_positives = epi_positives + para_positives

        # Top-K residues
        top_epi_df = epi_rows.sort_values(by="score", ascending=False).head(top_k)
        top_para_df = para_rows.sort_values(by="score", ascending=False).head(top_k)

        top_epi_list = top_epi_df.to_dict(orient="records")
        top_para_list = top_para_df.to_dict(orient="records")

        # Top-K pairs
        top_pairs_list: list[dict[str, Any]] = []
        pair_positives = 0
        total_pairs = 0
        if instance_id in pair_by_instance:
            inst_pairs = pair_by_instance[instance_id]
            total_pairs = len(inst_pairs)
            pair_positives = int((inst_pairs["score"] >= threshold).sum())
            top_pairs_df = inst_pairs.sort_values(by="score", ascending=False).head(top_k)
            top_pairs_list = top_pairs_df.to_dict(orient="records")

        # Check PDB structure
        pdb_file = locate_pdb_file(pdb_dir, instance_id)
        pdb_found = False
        epitope_pdb_name: Optional[str] = None
        paratope_pdb_name: Optional[str] = None
        pml_name: Optional[str] = None

        if pdb_file:
            try:
                atoms = parse_pdb_file(pdb_file)
                if not atoms:
                    logger.warning(f"No ATOM records found in {pdb_file}. Skipping PDB generation for {instance_id}.")
                else:
                    # Residue index sets for chain inference
                    epi_res_set = set(epi_rows["residue_index"])
                    para_res_set = set(para_rows["residue_index"])

                    # Role by chain if available
                    role_by_chain: dict[str, str] = {}
                    if "chain" in inst_node.columns and inst_node["chain"].notna().any():
                        for _, row in inst_node.iterrows():
                            if pd.notna(row["chain"]):
                                role_by_chain[str(row["chain"])] = str(row["role"])

                    epitope_chains, paratope_chains = infer_chains(
                        atoms=atoms,
                        epitope_indices=epi_res_set,
                        paratope_indices=para_res_set,
                        explicit_epitope_chains=explicit_epitope_chains,
                        explicit_paratope_chains=explicit_paratope_chains,
                        role_by_chain=role_by_chain,
                    )

                    # Score dictionaries
                    epi_scores = dict(zip(epi_rows["residue_index"], epi_rows["score"]))
                    para_scores = dict(zip(para_rows["residue_index"], para_rows["score"]))

                    # Map residue scores per chain
                    epi_score_map_by_chain: dict[str, dict[tuple[int, str], float]] = {}
                    for c in epitope_chains:
                        chain_atoms = [a for a in atoms if a.chain_id == c]
                        epi_score_map_by_chain[c] = resolve_residue_scores(chain_atoms, epi_scores)

                    para_score_map_by_chain: dict[str, dict[tuple[int, str], float]] = {}
                    for c in paratope_chains:
                        chain_atoms = [a for a in atoms if a.chain_id == c]
                        para_score_map_by_chain[c] = resolve_residue_scores(chain_atoms, para_scores)

                    # Write {instance_id}_epitope.pdb
                    epitope_pdb_name = f"{instance_id}_epitope.pdb"
                    paratope_pdb_name = f"{instance_id}_paratope.pdb"
                    pml_name = f"{instance_id}.pml"

                    epi_out_path = output_dir / epitope_pdb_name
                    para_out_path = output_dir / paratope_pdb_name
                    pml_out_path = output_dir / pml_name

                    write_colored_pdb(atoms, epitope_chains, epi_score_map_by_chain, epi_out_path)
                    write_colored_pdb(atoms, paratope_chains, para_score_map_by_chain, para_out_path)

                    # Generate PyMOL script
                    write_pymol_script(
                        instance_id=instance_id,
                        output_pml_path=pml_out_path,
                        epitope_pdb_filename=epitope_pdb_name,
                        paratope_pdb_filename=paratope_pdb_name,
                    )

                    pdb_found = True
                    logger.info(f"Generated colored PDBs and PyMOL script for: {instance_id}")

            except Exception as exc:
                logger.warning(f"Error processing PDB structure for {instance_id}: {exc}. Skipping PDB generation.")
                pdb_found = False
        else:
            logger.warning(f"Structure not found for instance '{instance_id}' in {pdb_dir}. Skipping PDB generation.")

        report_instances.append(
            InstanceReportData(
                instance_id=instance_id,
                pdb_found=pdb_found,
                epitope_pdb_name=epitope_pdb_name,
                paratope_pdb_name=paratope_pdb_name,
                pml_name=pml_name,
                mean_epitope_score=mean_epi_score,
                mean_paratope_score=mean_para_score,
                epitope_positives=epi_positives,
                paratope_positives=para_positives,
                total_positives=total_positives,
                total_epitope_residues=len(epi_rows),
                total_paratope_residues=len(para_rows),
                top_epitope=top_epi_list,
                top_paratope=top_para_list,
                top_pairs=top_pairs_list,
                pair_positives=pair_positives,
                total_pairs=total_pairs,
            )
        )

    # 4. Generate HTML Report
    report_html_path = output_dir / "report.html"
    logger.info(f"Generating HTML summary report: {report_html_path}")
    generate_html_report(
        instances=report_instances,
        output_html_path=report_html_path,
        threshold=threshold,
        top_k=top_k,
        has_pair_data=bool(pair_df is not None and not pair_df.empty),
    )

    logger.info("=" * 60)
    logger.info("Visualization pipeline completed successfully!")
    logger.info(f"Total instances processed: {len(report_instances)}")
    logger.info(f"PDB structures mapped:     {sum(1 for i in report_instances if i.pdb_found)}")
    logger.info(f"PDB structures skipped:    {sum(1 for i in report_instances if not i.pdb_found)}")
    logger.info(f"HTML Summary Report:       {report_html_path}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize antibody-antigen interface predictions with PDB B-factor coloring, PyMOL scripts, and HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--node-predictions",
        type=Path,
        required=True,
        help="Path to node_predictions.parquet file containing residue-level predictions.",
    )
    parser.add_argument(
        "--pair-predictions",
        type=Path,
        default=None,
        help="Optional path to pair_predictions.parquet file containing residue pair contact predictions.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing input PDB structure files named {instance_id}.pdb.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where colored PDBs, PyMOL scripts, and report.html will be saved.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-scoring residues (and contact pairs) to include in the HTML report per instance.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for counting positive interface residues.",
    )
    parser.add_argument(
        "--epitope-chains",
        type=str,
        default=None,
        help="Optional comma-separated list of chain IDs to treat as epitope/antigen (overrides auto-detection).",
    )
    parser.add_argument(
        "--paratope-chains",
        type=str,
        default=None,
        help="Optional comma-separated list of chain IDs to treat as paratope/antibody (overrides auto-detection).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    explicit_epitope = [c.strip() for c in args.epitope_chains.split(",")] if args.epitope_chains else None
    explicit_paratope = [c.strip() for c in args.paratope_chains.split(",")] if args.paratope_chains else None

    run_visualization(
        node_predictions_path=args.node_predictions,
        output_dir=args.output_dir,
        pair_predictions_path=args.pair_predictions,
        pdb_dir=args.pdb_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        explicit_epitope_chains=explicit_epitope,
        explicit_paratope_chains=explicit_paratope,
    )


if __name__ == "__main__":
    main()
