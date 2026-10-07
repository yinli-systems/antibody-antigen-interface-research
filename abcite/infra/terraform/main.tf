terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.117"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "abcite" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

resource "azurerm_search_service" "abcite" {
  name                = var.search_service_name
  resource_group_name = azurerm_resource_group.abcite.name
  location            = azurerm_resource_group.abcite.location
  sku                 = var.search_sku
  replica_count       = 1
  partition_count     = 1
  semantic_search_sku = "free"
  tags                = var.tags
}

resource "azurerm_cognitive_account" "openai" {
  name                  = var.openai_account_name
  location              = azurerm_resource_group.abcite.location
  resource_group_name   = azurerm_resource_group.abcite.name
  kind                  = "OpenAI"
  sku_name              = var.openai_sku
  custom_subdomain_name = var.openai_account_name
  tags                  = var.tags
}
