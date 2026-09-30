# FrameContact anonymous research draft

This directory is an Overleaf-ready, venue-neutral anonymous working draft. No
conference, workshop, review mode, or page limit is assumed until a destination
venue is selected.

## Upload to Overleaf

1. Upload `framecontact_venue_neutral_overleaf.zip` as a new project.
2. Set the compiler to pdfLaTeX and the main document to `main.tex`.
3. Apply the destination venue's style and options only after it is fixed.
4. Keep the author block anonymous until the destination review policy is known.
5. Invite collaborators from Overleaf's **Share** dialog.

## Local build

```bash
python3 make_figure.py
tectonic main.tex
```

## Result map

- `fresh156`: frozen antigen-cluster-disjoint **development** holdout that
  confirms the screen-stage routing decision; it does not evaluate the later
  full-development retrained checkpoints.
- `2,565/549`: full-development retraining; the 549 validation complexes both
  select checkpoints and supply the reported development summary.
- Predicted-geometry sensitivity: the same 2,565/549 protocol with antibody and
  antigen geometry predicted separately from sequence. Experimental structures
  are not model inputs and are not required at inference; the completed run
  still uses bound-derived contact labels and auxiliary distance supervision.
- Historical SAbDab2 benchmark: previously exposed, non-blind, single-seed,
  measurement-only metrics reported in the appendix; not used for selection.
- Final-retrained-checkpoint independent evaluation: not yet performed.
- Temporal holdout: sealed and not evaluated.
- ESM-C: the same-protocol base-vs-published-AbCDR development control has
  completed and is indexed under `../../results/esmc_representation_v1/`; it is
  not folded into the paper's primary ESM-2 task-routing claim.

The compact machine-readable source of every number and evidence role used in
the draft is `data/frozen_metrics.json`. Bibliographic metadata and verification
sources are recorded in `data/reference_audit.json`.

## Suggested titles

1. **FrameContact: Contact-Derived Task Routing for Joint Antibody–Antigen Interface Prediction**
2. **One Interface, Three Tasks: Contact-Derived Routing for Epitope, Paratope, and Contact Prediction**
3. **From Pair Contacts to Epitopes: Geometry-Aware Task Routing for Antibody–Antigen Interfaces**

The first title is used in `main.tex` because it names the method and its central
technical contribution.
