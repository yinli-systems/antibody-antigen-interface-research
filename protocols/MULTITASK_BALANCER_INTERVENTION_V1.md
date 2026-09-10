# Intervening on gradient scale, and on gradient conflict

Registered 2026-09-10, **before any balancer was implemented**.

## The prediction, and where it comes from

Three measurements already recorded:

* `cos(∇node, ∇pair)` on the shared trunk is **+0.70**, with **0 of 240**
  complexes negative (`results/gradient_conflict_v1/`);
* `‖∇pair‖ / ‖∇node‖` is **1.37–1.50**;
* quartering the pair weight improves node AUPRC by 13.3% **and** pair AUPRC by
  10.8% (`results/multitask_coupling_v1/`).

Those say the failure is scale, not conflict. If that is right, the two families
of multi-task optimiser should behave differently in a way fixed in advance:

> **Methods that rebalance gradient magnitude — GradNorm, FAMO — should recover
> most of what hand-tuning λ recovers. Methods that remove gradient conflict —
> PCGrad, CAGrad — should do close to nothing, because there is almost no
> conflict for them to remove.**

PCGrad projects `g_i` off `g_j` only when `g_i·g_j < 0`. Measured here, that
condition fires on essentially no complex, so PCGrad should be near-identical to
the unmodified objective. **That makes it a negative control, not a competitor.**

## The arms

| arm | what it does |
|---|---|
| `none` (λ=1) | the default objective, already run |
| `lambda_0.25` | hand-tuned weight, already run — the target to recover |
| `gradnorm` | rebalances task weights toward equal gradient norm |
| `famo` | balances loss decrease rates |
| `pcgrad` | projects conflicting gradients apart — **negative control** |

Ten seeds each, AsEP `epitope_group`, **validation only**. The frozen group test
split is not read.

## What each outcome means

**GradNorm/FAMO recover most of λ=0.25's gain and PCGrad recovers little**:
the mechanism is established by intervention, not only by correlation.
Intervening on scale fixes the failure; intervening on conflict does not.

**All three help equally**: the balancers are doing something other than what
their descriptions claim on this problem, and the mechanism is not isolated by
this experiment. Reported as such.

**PCGrad helps as much as GradNorm**: the conflict account is not dead after
all, and the cosine measurement needs re-examining — most likely per-layer
conflict cancelling in the aggregate, which the gradient record already lists as
a limit.

**None of them reach λ=0.25**: automatic balancing underperforms a hand-set
constant here, which is worth reporting on its own and is a known failure mode
of these methods rather than a defect of the analysis.

## Implementation constraints, fixed now

Every arm uses the same seeds, schedule, epoch budget, representations and
checkpoint-selection metric as the λ arms. Balancers act on the **shared trunk**
gradients only — the same 863,494 parameters the cosine was measured on — since
head-specific gradients cannot conflict or compete by construction.

If a balancer cannot be implemented faithfully in the time available it is
dropped and its absence recorded, rather than shipped as an approximation whose
behaviour is not the published method's.
