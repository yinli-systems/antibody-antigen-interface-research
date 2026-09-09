from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    provider: str = Field(default="auto", alias="ABCITE_PROVIDER")
    log_level: str = Field(default="INFO", alias="ABCITE_LOG_LEVEL")
    abstain_threshold: float = Field(default=0.12, alias="ABCITE_ABSTAIN_THRESHOLD")
    hitl_high_stakes: bool = Field(default=True, alias="ABCITE_HITL_HIGH_STAKES")
    corpus_dir: str = Field(default="corpus", alias="ABCITE_CORPUS_DIR")
    figures_dir: str = Field(default="corpus/figures", alias="ABCITE_FIGURES_DIR")
    host: str = Field(default="0.0.0.0", alias="ABCITE_HOST")
    port: int = Field(default=8000, alias="ABCITE_PORT")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_vision_model: str = Field(default="gpt-4o-mini", alias="OPENAI_VISION_MODEL")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")
    aws_region: str = Field(default="eu-west-2", alias="AWS_REGION")
    aws_bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        alias="AWS_BEDROCK_MODEL_ID",
    )

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="abcite", alias="LANGSMITH_PROJECT")
    rcsb_live: bool = Field(default=False, alias="ABCITE_RCSB_LIVE")
    azure_search_endpoint: str = Field(default="", alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(default="", alias="AZURE_SEARCH_API_KEY")
    azure_search_index: str = Field(default="abcite-chunks", alias="AZURE_SEARCH_INDEX")
    vision_fusion: bool = Field(default=False, alias="ABCITE_VISION_FUSION")
    vision_rerank: bool = Field(default=False, alias="ABCITE_VISION_RERANK")
    vision_rerank_k: int = Field(default=50, alias="ABCITE_VISION_RERANK_K")


def get_settings() -> Settings:
    return Settings()
