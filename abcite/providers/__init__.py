from abcite_core.settings import Settings
from abcite_core.vision import LocalImageEncoder
from providers.azure import AzureOpenAILLM
from providers.bedrock import BedrockLLM
from providers.local import LocalLLM
from providers.openai import OpenAILLM, OpenAIVision


def get_llm(settings: Settings):
    """Keys present → that cloud provider for phrasing. Otherwise local extractive."""
    name = (settings.provider or "auto").lower()
    if name == "local":
        return LocalLLM()
    if name == "openai" or (name == "auto" and settings.openai_api_key):
        if settings.openai_api_key:
            return OpenAILLM(settings.openai_api_key)
        return LocalLLM()
    if name == "azure" or (name == "auto" and settings.azure_openai_api_key):
        if settings.azure_openai_api_key and settings.azure_openai_endpoint:
            return AzureOpenAILLM(
                settings.azure_openai_endpoint,
                settings.azure_openai_api_key,
                settings.azure_openai_deployment,
            )
        return LocalLLM()
    if name == "bedrock":
        return BedrockLLM(settings.aws_region, settings.aws_bedrock_model_id)
    return LocalLLM()


def get_vision(settings: Settings):
    if settings.openai_api_key:
        return OpenAIVision(settings.openai_api_key, settings.openai_vision_model)
    return LocalImageEncoder()
