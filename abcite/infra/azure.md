# Azure starter (UK South)

## Python

| File | Role |
|---|---|
| `providers/azure.py` | Azure OpenAI chat completions for **/query phrasing** |
| `abcite_core/azure_search.py` | Hybrid + semantic ranker query body (`queryType=semantic`, `vectorQueries`) |
| `tests/test_azure_search.py` | Mocked HTTP — no live Azure |

```bash
export AZURE_SEARCH_ENDPOINT=https://srch-abcite-demo.search.windows.net
export AZURE_SEARCH_API_KEY=...          # from `az search admin-key`, not git
export AZURE_OPENAI_ENDPOINT=https://aoai-abcite-demo.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

## Terraform

```bash
cd abcite/infra/terraform
cp terraform.tfvars.example terraform.tfvars   # edit unique names
terraform init
terraform plan
```

Resources: `azurerm_resource_group`, `azurerm_search_service` (hybrid-capable SKU), `azurerm_cognitive_account` kind=OpenAI. Outputs: `search_endpoint`, `openai_endpoint`. No secrets in the repo.
