# Immutable preregistration publication guide

The compact bundle is ready for external registration, but no external record
has been created yet. Publishing or embargoing it changes external state and
therefore requires the project owner's explicit choice.

Recommended order:

1. Register the bundle on OSF before the first eligible post-cutoff candidate
   exists. Choose either public registration or an embargo. OSF registrations
   are time-stamped and immutable after submission.
2. Record the final OSF registration URL/DOI and deposited archive SHA-256 in a
   small local receipt. Do not replace or edit the frozen local protocol.
3. Optionally mirror the same byte-identical archive on Zenodo for a DOI. A
   published Zenodo record does not permit later file replacement.
4. Optionally submit only the archive digest to Sigstore Rekor to add a public,
   append-only transparency timestamp without exposing candidate data.

Do not publish model weights, future candidate sequences, labels, per-complex
predictions, credentials, or cluster paths. Future protocol amendments must be
new append-only records that name their parent SHA-256 and disclose whether any
candidate or label had already been seen.

Official references:

- OSF registrations: <https://help.osf.io/article/330-welcome-to-registrations>
- Zenodo publication: <https://help.zenodo.org/docs/get-started/quickstart/>
- Sigstore Rekor transparency log: <https://docs.sigstore.dev/about/security/>
