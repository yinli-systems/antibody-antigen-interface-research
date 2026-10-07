# Azure vs Bedrock (same `LLMProvider.complete` interface)

| | Azure OpenAI | AWS Bedrock |
|---|---|---|
| Adapter | `providers/azure.py` | `providers/bedrock.py` |
| Env | `AZURE_OPENAI_*` (via ABCITE_ prefix on shared settings + raw names in .env.example) | `AWS_REGION`, `AWS_BEDROCK_MODEL_ID` |
| Retrieval | Intended: Azure AI Search hybrid + semantic ranker | Intended: Bedrock Knowledge Bases |
| This commit | Chat completions used for **/query phrasing** when keys set (`compose_with_provider`) | `converse` if boto3 present |
| Local default | `ABCITE_PROVIDER=auto` → extractive if no keys | same |

UK jobs cite both clouds; keep adapters thin and identical so the LangGraph does not branch.
