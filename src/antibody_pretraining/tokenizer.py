"""Fixed residue-level tokenization for antibody variable regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SPECIAL_TOKENS = ("[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]")
STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUES = STANDARD_RESIDUES + "X"
VOCABULARY = SPECIAL_TOKENS + tuple(RESIDUES)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
ID_TO_TOKEN = dict(enumerate(VOCABULARY))

PAD_ID = TOKEN_TO_ID["[PAD]"]
MASK_ID = TOKEN_TO_ID["[MASK]"]
CLS_ID = TOKEN_TO_ID["[CLS]"]
SEP_ID = TOKEN_TO_ID["[SEP]"]
UNK_ID = TOKEN_TO_ID["[UNK]"]
FIRST_RESIDUE_ID = TOKEN_TO_ID["A"]
STANDARD_RESIDUE_COUNT = len(STANDARD_RESIDUES)

CHAIN_TYPE_TO_ID = {
    "padding": 0,
    "heavy": 1,
    "kappa": 2,
    "lambda": 3,
    "light_unknown": 4,
    "paired_heavy": 5,
    "paired_light": 6,
}
ID_TO_CHAIN_TYPE = {value: key for key, value in CHAIN_TYPE_TO_ID.items()}


class TokenizerError(ValueError):
    """Report a fail-closed sequence or chain-tokenization error."""


@dataclass(frozen=True)
class EncodedAntibody:
    """Represent one unpadded residue-token sequence and its chain types."""

    input_ids: tuple[int, ...]
    type_ids: tuple[int, ...]


def normalize_locus(value: str) -> str:
    """Normalize strict OAS locus aliases to model chain types."""
    if not isinstance(value, str):
        raise TokenizerError("Locus must be text")
    normalized = value.strip().upper()
    aliases = {
        "H": "heavy",
        "IGH": "heavy",
        "K": "kappa",
        "IGK": "kappa",
        "L": "lambda",
        "IGL": "lambda",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise TokenizerError(f"Unsupported antibody locus: {value!r}") from error


def validate_residue_sequence(
    sequence: str,
    *,
    minimum_length: int = 80,
    maximum_length: int = 192,
) -> str:
    """Return one canonical sequence or reject ambiguity and invalid lengths."""
    if not isinstance(sequence, str):
        raise TokenizerError("Sequence must be text")
    canonical = sequence.strip().upper()
    if not minimum_length <= len(canonical) <= maximum_length:
        raise TokenizerError("Sequence length is outside the frozen boundary")
    invalid = sorted(set(canonical) - set(RESIDUES))
    if invalid:
        raise TokenizerError(f"Sequence contains unsupported residues: {invalid}")
    return canonical


def encode_unpaired(sequence: str, chain_type: str) -> EncodedAntibody:
    """Encode one validated unpaired sequence with explicit boundary tokens."""
    canonical = validate_residue_sequence(sequence)
    if chain_type not in {"heavy", "kappa", "lambda", "light_unknown"}:
        raise TokenizerError("Unpaired chain type is invalid")
    token_ids = (CLS_ID,) + tuple(TOKEN_TO_ID[residue] for residue in canonical) + (
        SEP_ID,
    )
    type_id = CHAIN_TYPE_TO_ID[chain_type]
    return EncodedAntibody(
        input_ids=token_ids,
        type_ids=(type_id,) * len(token_ids),
    )


def pad_encoded(
    encoded: Iterable[EncodedAntibody],
    *,
    pad_to_length: int | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[bool]]]:
    """Pad a nonempty batch and return tokens, types, and validity masks."""
    items = tuple(encoded)
    if not items:
        raise TokenizerError("Cannot pad an empty batch")
    longest = max(len(item.input_ids) for item in items)
    target = longest if pad_to_length is None else pad_to_length
    if target < longest:
        raise TokenizerError("pad_to_length truncates an encoded sequence")
    input_batch: list[list[int]] = []
    type_batch: list[list[int]] = []
    mask_batch: list[list[bool]] = []
    for item in items:
        if len(item.input_ids) != len(item.type_ids):
            raise TokenizerError("Token and type lengths disagree")
        padding = target - len(item.input_ids)
        input_batch.append(list(item.input_ids) + [PAD_ID] * padding)
        type_batch.append(
            list(item.type_ids) + [CHAIN_TYPE_TO_ID["padding"]] * padding
        )
        mask_batch.append([True] * len(item.input_ids) + [False] * padding)
    return input_batch, type_batch, mask_batch


def decode_residues(input_ids: Iterable[int]) -> str:
    """Decode residue tokens while ignoring boundaries and padding."""
    output: list[str] = []
    for token_id in input_ids:
        token = ID_TO_TOKEN.get(int(token_id))
        if token in RESIDUES:
            output.append(token)
        elif token in {"[PAD]", "[CLS]", "[SEP]"}:
            continue
        else:
            raise TokenizerError("Cannot decode a masked or unknown token as a residue")
    return "".join(output)
