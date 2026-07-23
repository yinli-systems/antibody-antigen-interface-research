# Security and data handling

Do not commit credentials, access tokens, private keys, cluster usernames, internal
hostnames, raw structural data, model weights, or participant-linked information.

Before every release, run a secret scan and inspect all newly tracked files larger
than 10 MB. Environment-specific paths in cluster templates must use placeholders.
