# Azure starter (no subscription required to read)

AbCite’s cloud path is the `LLMProvider` adapter, not a full Foundry/Terraform product.

## What exists

| File | Role |
|---|---|
| `providers/azure.py` | Chat completions against Azure OpenAI (`/openai/deployments/{name}/chat/completions`) |
| `providers/openai.py` | Same phrasing role for public OpenAI |
| `abcite_core/settings.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `infra/terraform/main.tf` | Non-deployable note only |

## Wire it

```bash
export ABCITE_PROVIDER=azure   # or auto
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
cd abcite
python -m uvicorn apps.api.main:app --port 8000
```

`/query` will call Azure for **phrasing** and still attach citations from retrieved chunks. Without keys, `ABCITE_PROVIDER=auto` falls back to local extractive.

## Next (not in this commit)

- Azure AI Search hybrid + semantic ranker instead of in-process BM25
- Private endpoint / Key Vault for the API key
- `azd` bicep for UK South
