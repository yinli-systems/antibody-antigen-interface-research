# Stage three: the effect does not transfer

> **Status in the paper: supplementary.** This line is not a main claim and does
> not get main-text space.
>
> **Corrected 2026-09-11.** Stage two varied `d_model` at a fixed
> `--selection-metric total`, so what stage three confirms is a **configuration
> bundle** — `total` selection *and* `d_model` 256 together, against a frozen
> configuration that used `partner_rank` selection at `d_model` 128. The two
> changes cannot be separated by this design. The one sentence for the main text
> is therefore: *a configuration selected on development data did not show an
> improvement under a pre-registered confirmation on an independently evaluated
> ratio split.* The table goes in the appendix. Nine pages are worth more than half a page spent explaining an
> effect that did not reproduce across splits.

The second and final registered reading of the AsEP `epitope_ratio` test split,
taken once, with the configuration fixed before the split was touched.

## Result

Selection metric `total`, `d_model` 256 — the value the mechanical rule chose
from group validation — ten seeds, threshold selected per seed on ratio
validation and applied unchanged to ratio test. The prior reading's frozen
configuration is `partner_rank` selection at `d_model` 128, so **both** knobs
differ between the two rows.

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

**The difference is −0.0033 and the interval contains the prior value.** No
improvement was shown. That an interval contains a value is not evidence the two
are equal — it is the absence of evidence that they differ, and with one arm
having ten seeds and the other being a previously recorded mean, this design
cannot support an equivalence claim either way.

The ten-seed mean-score ensemble reaches MCC 0.2670. It is reported here as
registered and is **not** the headline figure, because the published rows this
sits beside are single models.

## What this settles

On group **validation**, at a fixed `total` selection metric, `d_model` 256 beat
`d_model` 128 by +0.0278 mean per-complex MCC over ten paired seeds, with the
interval excluding zero. That replication holds and is not in doubt — as a
statement about **width**, which is what it measured.

The configuration built from it showed no improvement on an independent split's
test set, with everything fixed in advance.

Under the registered outcomes this is the second. What is recorded is that a
configuration chosen on development data did not confirm, and that the
development-set effect is reported as **specific to the split and the metric it
was found on**. It is not recorded that checkpoint selection has some
split-specific mechanism: the design changed two things at once and cannot
attribute the outcome to either.

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

The accurate summary is not that the optimisation failed. It is that **a
development-set configuration effect was real where it was found and did not
confirm where it was tested** — eight of ten seeds on group validation, no
improvement on the ratio test. Calling it a "checkpoint-selection effect"
overstates what varied. It cuts a
story that was not general enough, and the way it was cut is itself worth more
to this paper than the half-point it would have added: a promising validation
finding was put to an independent confirmation with every choice locked
beforehand, the confirmation failed, and the failure was kept rather than
discarded. No `d_model` was retried, no threshold convention revisited, no
ensemble promoted to rescue the number.
