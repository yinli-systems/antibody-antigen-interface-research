# Stage three: the effect does not transfer

> **Status in the paper: supplementary.** This line is not a main claim and does
> not get main-text space. Its one sentence there is: *a checkpoint-selection
> improvement observed and replicated on the group validation split did not
> transfer to an independently evaluated ratio split.* The table below goes in
> the appendix. Nine pages are worth more than half a page spent explaining an
> effect that did not reproduce across splits.

The second and final registered reading of the AsEP `epitope_ratio` test split,
taken once, with the configuration fixed before the split was touched.

## Result

Selection metric `total`, `d_model` 256 — the value the mechanical rule chose
from group validation — ten seeds, threshold selected per seed on ratio
validation and applied unchanged to ratio test.

| | prior reading (frozen configuration) | this reading |
|---|---|---|
| MCC | 0.2584 | **0.2551**  CI [0.2490, 0.2606] |
| AUROC | 0.7644 | 0.7630 |
| precision | 0.2454 | 0.2423 |
| recall | 0.580 | 0.5905 |
| F1 | 0.328 | 0.3258 |

Both readings use ten seeds and the identical convention: threshold chosen on
validation to maximise per-complex MCC, applied to test, metrics averaged per
complex. The comparison is like for like.

**The difference is −0.0033 and the interval contains the prior value.** Not
worse; indistinguishable.

The ten-seed mean-score ensemble reaches MCC 0.2670. It is reported here as
registered and is **not** the headline figure, because the published rows this
sits beside are single models.

## What this settles

On group **validation**, selecting the checkpoint on total loss rather than on
`partner_rank` was worth +0.0278 mean per-complex MCC over ten paired seeds,
with the interval excluding zero. That replication holds and is not in doubt.

It does not transfer. On an independent split's test set, with everything fixed
in advance, the configuration chosen that way performs the same as the one it
was meant to improve on.

Under the registered outcomes this is the second: *the misalignment is
pronounced on group-split validation and does not transfer.* The
group-validation replication is therefore reported as **split-specific**, and
the checkpoint-selection observation is a property of that split's validation
set rather than of the model.

## What does not follow

No further reading of either test split. The `epitope_group` test split remains
permanently frozen — it was deliberately not re-evaluated after the
checkpoint-selection hypothesis was discovered, in accordance with the
pre-specified test-access protocol, and this result does not reopen it. The
`epitope_ratio` split has now had both of its registered readings and has none
left.

Nothing here is retuned in response to the number. `d_model` was chosen by a
rule stated in advance, the threshold policy was fixed in advance, the primary
figure was designated in advance, and the ensemble does not become the headline
because it is larger.

It is worth being plain about what the negative result costs: the most promising
development-set finding of this line of work did not survive an independent
confirmation. That is what the confirmation was for.

The accurate summary is not that the optimisation failed. It is that **seltotal
was a real but non-transferable validation effect** — replicated at eight of ten
seeds where it was found, absent where it was tested independently. It cuts a
story that was not general enough, and the way it was cut is itself worth more
to this paper than the half-point it would have added: a promising validation
finding was put to an independent confirmation with every choice locked
beforehand, the confirmation failed, and the failure was kept rather than
discarded. No `d_model` was retried, no threshold convention revisited, no
ensemble promoted to rescue the number.
