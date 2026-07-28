# Antibody training engine v1

**Status:** sampler and scheduler semantics frozen; GPU evidence pending

The authoritative progress coordinate is the number of admitted antibody
residue tokens consumed. Each update selects complete sequences in canonical
512-sequence quanta until it reaches the frozen global residue-token target.
This makes update membership independent of rank partitioning while avoiding
cross-sequence attention. The observed token count, rather than padded tensor
width or an estimated epoch, advances the learning-rate schedule.

Within an update, masked-language-model loss is the sum over every selected
target on every rank divided by the exact global target count. Distributed
gradient averaging must be compensated explicitly and verified against a
single-process reference. Padding, boundary tokens, and unselected residues do
not enter the denominator.

The epoch-wise sampler is a counter-addressable affine bijection. Its cursor is
part of every checkpoint. Changing world size may repartition a frozen update,
but cannot change its global membership. Production resume additionally
requires the same corpus, protocol, model, and tokenizer hashes.

The 30-billion-token budget is a phase-A ceiling, not evidence that those
tokens have been downloaded or trained. Promotion past engineering pilots
requires the separately frozen corpus and validation gates.
