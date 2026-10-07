from abcite_core.azure_search import (
    AzureSearchIndex,
    document_from_chunk,
    hybrid_search_body,
    index_schema,
)
from abcite_core.models import Chunk, Modality


class _Resp:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "value": [
                {
                    "chunk_id": "txt-1n8z-parent",
                    "title": "1N8Z",
                    "text": "trastuzumab HER2 domain IV",
                    "modality": "text",
                    "pdb_id": "1N8Z",
                    "figure_id": "",
                    "@search.score": 1.2,
                    "@search.rerankerScore": 2.4,
                }
            ]
        }


def test_index_schema_hybrid_semantic():
    schema = index_schema()
    assert schema["semantic"]["defaultConfiguration"] == "abcite-semantic"
    assert schema["vectorSearch"]["algorithms"][0]["kind"] == "hnsw"
    names = {f["name"] for f in schema["fields"]}
    assert {"chunk_id", "text", "embedding", "modality", "pdb_id", "figure_id"} <= names
    emb = next(f for f in schema["fields"] if f["name"] == "embedding")
    assert emb["dimensions"] == 256


def test_hybrid_search_body_flags():
    body = hybrid_search_body("trastuzumab HER2", [0.1] * 256, top_k=5)
    assert body["queryType"] == "semantic"
    assert body["semanticConfiguration"] == "abcite-semantic"
    assert body["vectorQueries"][0]["kind"] == "vector"
    assert body["vectorQueries"][0]["k"] == 5


def test_azure_search_mocked_http():
    seen = {}

    def post(url, **kwargs):
        seen["url"] = url
        seen["json"] = kwargs["json"]
        seen["headers"] = kwargs["headers"]
        return _Resp()

    idx = AzureSearchIndex("https://srch.example.search.windows.net", "key", post=post)
    hits = idx.search("trastuzumab HER2 1N8Z", top_k=5)
    assert hits[0].chunk.chunk_id == "txt-1n8z-parent"
    assert hits[0].chunk.pdb_id == "1N8Z"
    assert "docs/search" in seen["url"]
    assert seen["headers"]["api-key"] == "key"
    assert seen["json"]["queryType"] == "semantic"


def test_document_from_chunk_metadata():
    c = Chunk(
        chunk_id="fig-1n8z",
        parent_id="txt-1n8z-parent",
        modality=Modality.FIGURE,
        title="fig",
        text="caption",
        source_url="https://example.org",
        source_label="t",
        pdb_id="1N8Z",
    )
    doc = document_from_chunk(c)
    assert doc["figure_id"] == "fig-1n8z"
    assert doc["modality"] == "figure"
    assert len(doc["embedding"]) == 256
