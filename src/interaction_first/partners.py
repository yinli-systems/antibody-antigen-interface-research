"""Deterministic false-negative-aware partner sampling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PartnerRecord:
    """Metadata used to construct and prioritize non-cognate partners."""

    instance_id: str
    antibody_key: str
    antigen_key: str
    antibody_length: int
    antigen_length: int
    antibody_group: str | None = None
    antigen_group: str | None = None
    epitope_group: str | None = None


class PartnerSampler:
    """Sample reproducible real-sequence partner negatives without self matches."""

    def __init__(
        self,
        records: Iterable[PartnerRecord],
        known_positive_pairs: Iterable[tuple[str, str]] = (),
        seed: int = 42,
    ) -> None:
        ordered = sorted(records, key=lambda record: record.instance_id)
        if len(ordered) < 2:
            raise ValueError("Partner sampling requires at least two instances")
        if len({record.instance_id for record in ordered}) != len(ordered):
            raise ValueError("Partner records contain duplicate instance IDs")
        self.records = tuple(ordered)
        self.by_id = {record.instance_id: record for record in ordered}
        self.known_positive_pairs = {
            (str(antibody_key), str(antigen_key))
            for antibody_key, antigen_key in known_positive_pairs
        }
        self.known_positive_pairs.update(
            (record.antibody_key, record.antigen_key) for record in ordered
        )
        self.seed = int(seed)

    @staticmethod
    def _relative_length_gap(left: int, right: int) -> float:
        return abs(left - right) / max(left, right, 1)

    def candidates(
        self,
        anchor_id: str,
        swap: str = "antibody",
        require_different_antigen_group: bool = True,
    ) -> list[PartnerRecord]:
        """Return valid candidates ordered from harder to easier negatives."""
        if anchor_id not in self.by_id:
            raise KeyError(f"Unknown anchor instance: {anchor_id}")
        if swap not in {"antibody", "antigen"}:
            raise ValueError("swap must be either 'antibody' or 'antigen'")
        anchor = self.by_id[anchor_id]
        candidates: list[PartnerRecord] = []
        for candidate in self.records:
            if candidate.instance_id == anchor_id:
                continue
            if swap == "antibody":
                proposed = (candidate.antibody_key, anchor.antigen_key)
                if candidate.antibody_key == anchor.antibody_key:
                    continue
                if require_different_antigen_group:
                    if (
                        anchor.antigen_group is not None
                        and candidate.antigen_group == anchor.antigen_group
                    ):
                        continue
            else:
                proposed = (anchor.antibody_key, candidate.antigen_key)
                if candidate.antigen_key == anchor.antigen_key:
                    continue
                if require_different_antigen_group:
                    if (
                        anchor.antigen_group is not None
                        and candidate.antigen_group == anchor.antigen_group
                    ):
                        continue
            if proposed in self.known_positive_pairs:
                continue
            candidates.append(candidate)

        def hardness(candidate: PartnerRecord) -> tuple[float, float, str]:
            if swap == "antibody":
                antibody_group_match = float(
                    anchor.antibody_group is not None
                    and candidate.antibody_group == anchor.antibody_group
                )
                length_gap = self._relative_length_gap(
                    anchor.antibody_length, candidate.antibody_length
                )
                group_priority = -antibody_group_match
            else:
                antigen_group_match = float(
                    anchor.antigen_group is not None
                    and candidate.antigen_group == anchor.antigen_group
                )
                length_gap = self._relative_length_gap(
                    anchor.antigen_length, candidate.antigen_length
                )
                group_priority = -antigen_group_match
            tie = hashlib.sha256(
                f"{self.seed}:{anchor_id}:{swap}:{candidate.instance_id}".encode("utf-8")
            ).hexdigest()
            return group_priority, length_gap, tie

        ordered = sorted(candidates, key=hardness)
        unique = []
        seen_partner_keys = set()
        for candidate in ordered:
            partner_key = (
                candidate.antibody_key if swap == "antibody" else candidate.antigen_key
            )
            if partner_key in seen_partner_keys:
                continue
            seen_partner_keys.add(partner_key)
            unique.append(candidate)
        return unique

    def sample(
        self,
        anchor_id: str,
        count: int,
        epoch: int,
        swap: str = "antibody",
        hard_pool_size: int = 32,
        require_different_antigen_group: bool = True,
    ) -> list[PartnerRecord]:
        """Sample up to count unique partners from the prioritized candidate pool."""
        if count <= 0:
            raise ValueError("Negative count must be positive")
        if hard_pool_size < count:
            raise ValueError("hard_pool_size cannot be smaller than count")
        candidates = self.candidates(
            anchor_id,
            swap=swap,
            require_different_antigen_group=require_different_antigen_group,
        )
        if len(candidates) < count and require_different_antigen_group:
            candidates = self.candidates(
                anchor_id,
                swap=swap,
                require_different_antigen_group=False,
            )
        if not candidates:
            raise ValueError(f"No valid negative partner exists for {anchor_id}")
        selected_count = min(count, len(candidates))
        pool = candidates[: min(len(candidates), hard_pool_size)]
        digest = hashlib.sha256(
            f"{self.seed}:{epoch}:{anchor_id}:{swap}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return rng.sample(pool, selected_count)


def records_from_rows(rows: Sequence[Mapping[str, object]]) -> list[PartnerRecord]:
    """Create validated partner records from tabular row mappings."""
    records = []
    for row in rows:
        records.append(
            PartnerRecord(
                instance_id=str(row["instance_id"]),
                antibody_key=str(row["antibody_key"]),
                antigen_key=str(row["antigen_key"]),
                antibody_length=int(row["antibody_length"]),
                antigen_length=int(row["antigen_length"]),
                antibody_group=(
                    None if row.get("antibody_group") in (None, "") else str(row["antibody_group"])
                ),
                antigen_group=(
                    None if row.get("antigen_group") in (None, "") else str(row["antigen_group"])
                ),
                epitope_group=(
                    None if row.get("epitope_group") in (None, "") else str(row["epitope_group"])
                ),
            )
        )
    return records
