"""Azure AI Search hybrid+semantic retriever.

Same `search(query, top_k) -> list[Hit]` surface as `HybridIndex`.
Live HTTP only when AZURE_SEARCH_ENDPOINT + key are set; tests inject a transport.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import Hit
from abcite_core.vision import hashing_vector

API_VERSION = "2024-07-01"
SEMANTIC_CONFIG = "abcite-semantic"
VECTOR_FIELD = "embedding"
VECTOR_DIM = 256


def index_schema(name: str = "abcite-chunks") -> dict[str, Any]:
    """Index definition matching Microsoft hybrid + semantic ranker docs."""
    return {
        "name": name,
        "fields": [
            {"name": "chunk_id", "type": "Edm.String", "key": True, "filterable": True},
            {"name": "text", "type": "Edm.String", "searchable": True},
            {"name": "title", "type": "Edm.String", "searchable": True},
            {"name": "modality", "type": "Edm.String", "filterable": True, "facetable": True},
            {"name": "pdb_id", "type": "Edm.String", "filterable": True},
            {"name": "figure_id", "type": "Edm.String", "filterable": True},
            {
                "name": VECTOR_FIELD,
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": VECTOR_DIM,
                "vectorSearchProfile": "abcite-hnsw-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [{"name": "abcite-hnsw", "kind": "hnsw"}],
            "profiles": [{"name": "abcite-hnsw-profile", "algorithm": "abcite-hnsw"}],
        },
        "semantic": {
            "defaultConfiguration": SEMANTIC_CONFIG,
            "configurations": [
                {
                    "name": SEMANTIC_CONFIG,
                    "prioritizedFields": {
                        "titleField": {"fieldName": "title"},
                        "contentFields": [{"fieldName": "text"}],
                    },
                }
            ],
        },
    }


def hybrid_search_body(query: str, vector: list[float], top_k: int = 6) -> dict[str, Any]:
    """Hybrid keyword + vector query with semantic ranker enabled."""
    return {
        "search": query,
        "queryType": "semantic",
        "semanticConfiguration": SEMANTIC_CONFIG,
        "captions": "extractive",
        "answers": "extractive|count-3",
        "top": top_k,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": vector,
                "fields": VECTOR_FIELD,
                "k": top_k,
            }
        ],
        "select": "chunk_id,text,title,modality,pdb_id,figure_id",
    }


def document_from_chunk(chunk: Chunk) -> dict[str, Any]:
    blob = f"{chunk.title} {chunk.text} {chunk.caption or ''}"
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "title": chunk.title,
        "modality": chunk.modality.value,
        "pdb_id": chunk.pdb_id or "",
        "figure_id": chunk.chunk_id if chunk.modality.value == "figure" else "",
        VECTOR_FIELD: hashing_vector(blob).tolist(),
    }


class AzureSearchIndex:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str = "abcite-chunks",
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self._post = post or httpx.post

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        if not (self.endpoint and self.api_key):
            raise RuntimeError("AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_API_KEY missing")
        vec = hashing_vector(query).tolist()
        url = (
            f"{self.endpoint}/indexes/{self.index_name}/docs/search"
            f"?api-version={API_VERSION}"
        )
        r = self._post(
            url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            json=hybrid_search_body(query, vec, top_k),
            timeout=30.0,
        )
        r.raise_for_status()
        payload = r.json()
        hits: list[Hit] = []
        for doc in payload.get("value", []):
            chunk = Chunk(
                chunk_id=doc["chunk_id"],
                parent_id=doc.get("parent_id") or doc["chunk_id"],
                modality=Modality(doc.get("modality") or "text"),
                title=doc.get("title") or doc["chunk_id"],
                text=doc.get("text") or "",
                source_url=doc.get("source_url") or "",
                source_label="azure-search",
                pdb_id=doc.get("pdb_id") or None,
            )
            score = float(doc.get("@search.rerankerScore") or doc.get("@search.score") or 0.0)
            hits.append(Hit(chunk=chunk, score=score, bm25=score, dense=score))
        return hits[:top_k]


def from_env() -> AzureSearchIndex | None:
    from abcite_core.settings import Settings

    s = Settings()
    if s.azure_search_endpoint and s.azure_search_api_key:
        return AzureSearchIndex(s.azure_search_endpoint, s.azure_search_api_key, s.azure_search_index)
    return None
