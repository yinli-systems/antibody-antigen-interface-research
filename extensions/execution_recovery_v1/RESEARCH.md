# Targeted primary-source research — 30 September 2026

## Sources actually consulted

- Abramson et al., Nature 2024, *Accurate structure prediction of biomolecular
  interactions with AlphaFold 3*, DOI 10.1038/s41586-024-07487-w.
  https://www.nature.com/articles/s41586-024-07487-w
  Read Methods and antibody sampling discussion. Sampling changes conclusions;
  preserve the registered seed/sample grid rather than shrinking it to pass.
- Official AF3 input specification:
  https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
  `--input_dir` consumes JSON files in the named directory; explicit modelSeeds,
  unpairedMsa, pairedMsa and templates encode the intended information budget.
- Official AF3 outputs and source:
  https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md
  https://github.com/google-deepmind/alphafold3/blob/main/run_alphafold.py
  Five samples/seed; top-ranked sample at root; existing directories can get
  timestamp suffixes. Ranking_score is for structure ranking, not binding proof.
  Current upstream naming differs from the installed normalizing wrapper.
  Inspected the actual SIF's run_alphafold.py, post_processing.py and version.py:
  3.0.1, prefixed sample filenames, 5 diffusion samples, 10 recycles by default.
- Official Slurm semantics:
  https://slurm.schedmd.com/sbatch.html
  https://slurm.schedmd.com/job_array.html
  `--parsable` returns job ID optionally followed by semicolon+cluster; a failed
  dependency remains failed even after requeue. Preserve submitted intents when
  transport acknowledgement is uncertain. afterok requires every array task.
- Official SaProt contract:
  https://github.com/westlake-repl/SaProt
  35M/650M frozen embeddings require structure-aware tokens. Current 1.3B
  availability does not justify changing a running 650M frozen sweep.
- Official ProSST contract:
  https://github.com/ai4protein/ProSST
  The independent structure quantizer is required. No substitution of 3Di or
  sequence-only tokens. No extra encoder search during the fixed layer sweep.

## Decisions, counterevidence and limits

Repair execution authority first, not a larger architecture. Neither the full
122x119 partner diagnostic nor rejected OOD admission supports a new specificity
claim. The running layer sweep has not completed its controls. A new model or
score chosen now would add unregistered validation selection. We therefore leave
its training and source hashes untouched, and repair the I/O/runtime contract
plus receipt parsing outside its frozen source tree.

Online docs describe current upstream, not necessarily the installed build.
The local original SIF, actual failure log and captured receipt schema override
assumptions based on directory names. No source, model weights or restricted
third-party assets are redistributed with this patch.
