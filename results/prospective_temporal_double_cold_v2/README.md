# Prospective temporal double-cold v2

This directory contains the compact public preregistration and the initial
metadata-only screen for a future antibody-and-antigen double-cold cohort.

- Freeze time: `2026-08-30T20:37:46Z`
- Eligible release date: strictly after `2026-08-30`
- Opening gate: at least 36 qualified instances and 30 unique PDB IDs
- Protocol SHA-256: `41cad95072a3189826877dfbbb6c0ff493c9fd37c66a0132e5ae3fea9579ee7e`
- Frozen checkpoint binding SHA-256: `c8be8d83fbedd6cce0809b6df5258f3a9be9d02a653d3285a6eed74ded9621f7`
- Initial official SAbDab2 metadata SHA-256: `00b9ab8864155fc34592e14eaf9bb92d38f4953e08fed10e1400be68233dccfb`
- Initial screen status: `WAITING_FOR_POST_CUTOFF_RELEASES`
- Initial counts: 21,914 metadata rows; zero post-cutoff candidates
- Bound-label opening count: zero

## Release-date authority correction

SAbDab2 documents the summary `date` column as the date of initial PDB
deposition, whereas the frozen protocol requires the initial public release
date. The v2.1 amendment preserves the original v2 files and changes no cutoff,
gate, model, checkpoint, task router, metric, or label rule. It replaces only
the metadata authority with the official RCSB
`rcsb_accession_info.initial_release_date` field.

- Amendment SHA-256: `e1d8dad751c213a93e7ebabb5da38d3177deea92331ab6011edb8ac090e82398`
- Exact RCSB snapshot SHA-256 (uncompressed canonical JSON):
  `74fe9b1638d0f3b44c128db8eae9ac031c9b0c6aba7ac089bbc41c67f7ae9f05`
- Resolved RCSB entries: 11,458 / 11,458
- Corrected screen SHA-256: `6404ba5d2b79d0d93c4c4cdb255d5055c610bdc2524634ad1f212c17915d45c7`
- Corrected screen status: `WAITING_FOR_POST_CUTOFF_RELEASES`
- Corrected post-cutoff candidates: zero
- Labels, bound coordinates, and cross-chain distances read: none

## Gate precision audit

The already analyst-exposed SABLE per-entry AP values were used only as a
design prior, not as another test or for model selection. At 30 unique PDBs,
the simulated median bootstrap 95% CI widths were approximately `0.051`
(contact), `0.215` (epitope), and `0.092` (paratope). At 60 unique PDBs they
were `0.037`, `0.153`, and `0.066`. This supports keeping 30 PDBs as a minimum
usable opening gate while describing it as precision-limited; sample size
alone does not remove the strong epitope heterogeneity.

- Precision-audit SHA-256: `8665644bffaafb0a957d65c7f8c12813db4e4f8e59bacd8b22e32f9c3c97e51c`
- Scientific test result: false
- Future temporal labels read: false

## Immutable-registration bundle

The deterministic compact bundle is prepared but has not been published to an
external registry. It contains the protocol, append-only correction, screens,
design audit, sealed-execution specification, implementation scripts, and
tests; it excludes weights, secrets, future candidates, and labels.

- Archive SHA-256: `83274adf0fdf21435f5d35ed6b760f9790c5543130eb02d43b096d9753ec1503`
- Manifest SHA-256: `dbf7c17b3111f3669355905de7b947c551b0695b97af1a1a0fff319807e9903c`
- External registration completed: false

The weekly watcher is metadata-only and cannot authorize label opening. SABLE
v0.2.0, the old exposed SAbDab2 test, and temporal v1 are explicitly closed and
cannot be reused as blind tests. The preferred final execution is by an
independent sealed steward; an internal fallback is limited to one irreversible
opening after both frozen minima pass.

Official design references:

- SAbDab2: <https://sabdab.opig.stats.ox.ac.uk/about>
- RCSB Data API: <https://data.rcsb.org/>
- SABLE v0.2.0: <https://zenodo.org/records/21759053>
- wwPDB release schedule: <https://www1.rcsb.org/pages/unreleased>
