"""Interaction-first components for antibody-antigen interface prediction."""

from .contacts import FORMAT_VERSION as CONTACT_FORMAT_VERSION
from .intra_chain_geometry import (
    DualIntraChainGeometryEncoder,
    GeometryBiasedSelfAttentionBlock,
    IntraChainGeometryEncoder,
    build_backbone_frames,
    build_intra_chain_relative_geometry,
)
from .partners import PartnerRecord, PartnerSampler
from .runtime import EmbeddingStore
from .sabdab2_structure import (
    IntraChainResidueClassifier,
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    StructureCacheError,
    materialize_structure_cache,
)
from .splits import SimilarityEdge, build_components, deterministic_component_split

__all__ = [
    "CONTACT_FORMAT_VERSION",
    "DualIntraChainGeometryEncoder",
    "GeometryBiasedSelfAttentionBlock",
    "IntraChainGeometryEncoder",
    "IntraChainResidueClassifier",
    "PartnerRecord",
    "PartnerSampler",
    "Sabdab2StructureBatchCollator",
    "Sabdab2StructureEmbeddingDataset",
    "EmbeddingStore",
    "SimilarityEdge",
    "StructureCacheError",
    "build_backbone_frames",
    "build_components",
    "build_intra_chain_relative_geometry",
    "deterministic_component_split",
    "materialize_structure_cache",
]
