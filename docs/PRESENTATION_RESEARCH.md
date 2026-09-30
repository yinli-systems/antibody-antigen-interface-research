# Research informing this repository's presentation

Reviewed 30 September 2026. These sources guide packaging and attribution; they
do not certify this repository, award it badges or verify its biological claims.

| Primary source | Relevant principle | Applied here |
|---|---|---|
| [GitHub: About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) | Explain usefulness, starting points and navigation; prefer relative links for repository content | Short landing page; detailed evidence in linked catalogues |
| [GitHub: Permanent links](https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files) | A branch link can change; a commit link identifies the exact file version | All historical branch evidence links use full commit IDs |
| [GitHub: Backing up a repository](https://docs.github.com/en/repositories/archiving-a-github-repository/backing-up-a-repository) | Preserve revision history, distinguish Git data from LFS/metadata | Original refs retained, full tracked-tree manifest and offline Git bundle |
| [Git: git-bundle](https://git-scm.com/docs/git-bundle) | Bundles carry refs and reachable objects, not every local working-tree or external state | Restore test, explicit exclusions, no claim that cluster tensors are bundled |
| [ACM: Artifact review and badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current) | Available, functional/reusable and independently validated results are different properties | Reporting verification is labelled as such, not “independent scientific replication” |

The ACM policy was readable through ACM's publishing host while its main policy
URL returned an access error in this session; no badge is claimed.

## Presentation rules

Use completed, central, comparable evidence rather than whichever isolated
number is largest. State cohort, model provenance and aggregation beside the
number. Keep negative findings visible, link every headline to its artifact,
and separate the numerical result from its interpretation.

Do not manufacture a single leaderboard across mismatched tasks, cohorts,
checkpoints, input modalities or budgets. Do not treat the official baseline as
a newly trained project model. Do not erase failed studies to shorten a README.

The chart uses a common zero-origin AP axis and the same ten-seed experiment for
all four bars. It displays absolute means; paired-effect uncertainty remains in
the linked registry rather than being substituted with unrelated error bars.
