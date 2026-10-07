"""Compatibility entry point for the preserved optional multimodal stack.

Frozen core modules keep their registered bytes. The optional stack has its
own enhanced base modules; no model algorithms are combined during integration.
"""
from .multimodal_v1.multimodal_pair_aware_esm2 import (
    StructureConditioningConfig,
    StructureReconstructionLoss,
    PredictedStructureAdapter,
    StructureConditionedPairAwareESM2,
)

__all__ = ['StructureConditioningConfig', 'StructureReconstructionLoss', 'PredictedStructureAdapter', 'StructureConditionedPairAwareESM2']
