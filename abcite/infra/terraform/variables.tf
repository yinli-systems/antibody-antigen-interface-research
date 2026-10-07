variable "location" {
  type        = string
  description = "Azure region (UK hiring default: uksouth)."
  default     = "uksouth"
}

variable "resource_group_name" {
  type        = string
  description = "Resource group for AbCite demo (no secrets)."
  default     = "rg-abcite-demo"
}

variable "search_service_name" {
  type        = string
  description = "Globally unique Azure AI Search name."
  default     = "srch-abcite-demo"
}

variable "search_sku" {
  type        = string
  description = "Search SKU. basic supports hybrid; semantic ranker needs supported SKU."
  default     = "basic"
}

variable "openai_account_name" {
  type        = string
  description = "Azure OpenAI account name (custom subdomain)."
  default     = "aoai-abcite-demo"
}

variable "openai_sku" {
  type    = string
  default = "S0"
}

variable "tags" {
  type = map(string)
  default = {
    project = "abcite"
    owner   = "yin-li"
  }
}
