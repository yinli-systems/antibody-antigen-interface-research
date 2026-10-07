output "resource_group_name" {
  value = azurerm_resource_group.abcite.name
}

output "search_endpoint" {
  value       = "https://${azurerm_search_service.abcite.name}.search.windows.net"
  description = "Set AZURE_SEARCH_ENDPOINT to this value."
}

output "openai_endpoint" {
  value       = azurerm_cognitive_account.openai.endpoint
  description = "Set AZURE_OPENAI_ENDPOINT to this value."
}

output "note" {
  value = "Keys stay in Key Vault / az search admin-key. Never commit terraform.tfvars with secrets."
}
