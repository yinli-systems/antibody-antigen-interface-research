import numpy as np
import torch

from paired_pretraining import (
    CLS_ID,
    REGION_CDR3,
    REGION_SPECIAL,
    SEP_ID,
    apply_cdr3_balanced_mlm,
    encode_pair,
    focal_mlm_loss_sum,
)


HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS"
LIGHT = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"


def test_pair_encoding_maps_boundaries_and_cdr3() -> None:
    encoded = encode_pair(
        HEAVY,
        LIGHT,
        ("GYDFTHY", "INTYTGEPT", "AKYPYYYGTSHWYFDV"),
        ("SASQDISNY", "FTS", "QQYSTVPWT"),
    )
    assert encoded.input_ids[0] == CLS_ID
    assert encoded.input_ids[len(HEAVY) + 1] == SEP_ID
    assert encoded.input_ids[-1] == SEP_ID
    assert encoded.region_ids[0] == REGION_SPECIAL
    assert int((encoded.region_ids == REGION_CDR3).sum()) == 25
    assert len(encoded.input_ids) == len(HEAVY) + len(LIGHT) + 3


def test_cdr3_masking_is_deterministic_and_balanced_in_expectation() -> None:
    encoded = encode_pair(
        HEAVY,
        LIGHT,
        ("GYDFTHY", "INTYTGEPT", "AKYPYYYGTSHWYFDV"),
        ("SASQDISNY", "FTS", "QQYSTVPWT"),
    )
    width = len(encoded.input_ids)
    tokens = np.tile(encoded.input_ids, (256, 1))
    regions = np.tile(encoded.region_ids, (256, 1))
    keys = np.arange(256, dtype=np.uint64) + 1
    first = apply_cdr3_balanced_mlm(tokens, regions, keys, update=3, seed=17)
    second = apply_cdr3_balanced_mlm(tokens, regions, keys, update=3, seed=17)
    assert np.array_equal(first.input_ids, second.input_ids)
    assert np.array_equal(first.labels, second.labels)
    expected = 0.15 * 256 * (len(HEAVY) + len(LIGHT))
    assert abs(first.selected_target_count - expected) / expected < 0.08
    assert first.labels.shape == (256, width)
    cdr3_selected = first.selected_by_region[REGION_CDR3]
    cdr3_total = 256 * 25
    assert abs(cdr3_selected / cdr3_total - 0.25) < 0.025


def test_focal_gamma_zero_matches_cross_entropy_sum() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, -100, 4, -100], [2, 3, -100, 0]])
    expected = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
    )
    observed = focal_mlm_loss_sum(logits, labels, gamma=0.0)
    torch.testing.assert_close(observed, expected)
