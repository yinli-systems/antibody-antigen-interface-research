# Minimal Azure placeholders — not a full production stack.
# Prefer docker compose locally. Fill via azd / ARM later.

variable "location" {
  type    = string
  default = "uksouth"
}

output "note" {
  value = "AbCite Azure OpenAI + AI Search wiring lives in providers/azure.py; this file is a stub, not a deployable environment."
}
