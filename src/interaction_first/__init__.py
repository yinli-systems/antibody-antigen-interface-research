"""Antibody-antigen interface prediction improvements.

Three high-priority modules:
1. inter_chain_geometry — Cross-chain geometric features (distance RBF, relative frames)
2. graph_community_split — Graph community detection dual-axis splitting
3. mipe_contact_head — Direct supervision contact prediction head for MIPE
"""

from .inter_chain_geometry import (
    InterChainGeometryBias,
    InterChainPairFeatureProjection,
    InterChainRelativeGeometry,
    build_inter_chain_geometry,
)
from .graph_community_split import (
    DualAxisSimilarityGraph,
    build_antibody_similarity_graph,
    build_antigen_family_graph,
    build_joint_constraint_graph,
    louvain_dual_axis_split,
    validate_dual_axis_split,
    spectral_fallback_split,
)
from .mipe_contact_head import (
    MIPEContactHead,
    MIPEContactEvaluator,
    focal_loss,
    mipe_contact_loss,
)
