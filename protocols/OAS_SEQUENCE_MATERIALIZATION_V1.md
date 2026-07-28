# OAS sequence materialization v1

**Status:** real-data pilot shard passed; production external dedup pending

The materializer consumes only plan-bound, hash-verified OAS download receipts.
It applies the rational plan sampling threshold before quality filtering,
requires productive and in-frame records without stop codons, enforces locus
agreement, accepts only the frozen residue alphabet and length range, and
rejects fatal ANARCI conditions. ANARCI deletion, insertion, and truncated-end
warnings are retained as bounded audit categories because they do not by
themselves imply a failed antibody-domain assignment. Their prevalence is
reported rather than hidden.

The fixed-width shard stores 194 uint8 token positions per example, encoded
lengths, chain types, and deterministic sample keys. This consumes about 205
bytes per sequence before filesystem overhead and allows direct memory-mapped
GPU input without runtime CSV parsing.

## Real-data pilot

The first shard used 20 completed data units from the frozen 250M plan:

- source rows: 8,399
- selected by plan fraction: 8,399
- accepted unique sequences: 8,112
- fatal ANARCI rejections: 201
- length rejections: 82
- exact duplicate rejections: 4
- dense-shard payload SHA-256: `0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6`
- dense-shard manifest file SHA-256: `16bc7901df9d288d3d812fad9815eecc186a969449073badd3b8a0b96ede9e15`

All 8,112 accepted pilot records carry a truncated-end warning, and 8,106
carry an ANARCI deletion warning. This reflects the early small units selected
for systems validation and is not evidence about the final 250M corpus.

The pilot performs exact deduplication only across its 20 input units. Its
manifest therefore sets `global_production_dedup_complete=false`. Production
materialization requires hash-bucketed full-corpus deduplication and removal
of any validation duplicate from training before a scientific pretraining run
is admitted.
