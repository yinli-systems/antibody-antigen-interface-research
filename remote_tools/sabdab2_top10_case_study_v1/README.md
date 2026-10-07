# SAbDab2 top-10 contact case study

This package freezes and executes the teacher-requested descriptive analysis:

- choose the contact checkpoint using development MPC-AP only;
- sample 20 of the 549 historical SAbDab2 test instances with a fixed seed;
- rank every residue pair by the frozen model probability;
- treat the top 10 pairs as predicted positives and report `[[TN, FP], [FN, TP]]`;
- nominate three case-study candidates within the fixed sample.

The test partition was previously exposed and this analysis is not an independent
performance estimate. Experimental structures supply contact labels only. Model
inputs are ESM-2 embeddings and independently sequence-predicted monomer geometry.
