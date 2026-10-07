# SAbDab2 predicted-geometry development sensitivity

This directory preserves the launchers, analyzers, immutable protocol freezers,
and fail-closed recovery receipts for the development-only replacement of
experimental bound-complex backbone coordinates with sequence-predicted
geometry.

Scientific boundaries:

- the frozen development split contains 2,565 training and 549 validation
  complexes;
- old SAbDab2 test and temporal structures are not materialized or accessed;
- predicted coordinates are chain-internal inputs only, with no cross-partner
  geometry;
- existing contact targets are reused byte-for-byte under an explicit
  provenance adapter rather than regenerated from predicted structures;
- the completed corrected-distance study retains bound-derived distance bins
  only as auxiliary training supervision, never as model input;
- the frozen `no_privileged_distance_v1` follow-up supplies no distance-label
  cache and sets the distance loss weight to zero, while holding the model,
  optimizer, split, task labels, seeds, and checkpoint rule fixed;
- the geometry-quality mechanism analysis aligns predicted and experimental
  backbones independently within each physical chain and relates chain-internal
  RMSD/graph changes to seed-matched development contact-ranking changes;
- recovery files document systems and schema failures; they are not additional
  model-selection attempts.

The compact public result and receipt live in
`results/sabdab2_predicted_geometry_v1/`. Large structure predictions,
checkpoints, caches, and scheduler logs remain excluded from Git.
