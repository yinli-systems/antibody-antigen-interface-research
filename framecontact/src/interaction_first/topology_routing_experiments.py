"""Isolated registry for adaptive topology-routing development experiments.

The condition was designed only after reading the authenticated AsEP validation
topology diagnostic.  It therefore cannot enter either the frozen
confirmatory registry or the earlier local-pose exploratory registry.
"""

from __future__ import annotations

from types import MappingProxyType


TOPOLOGY_ROUTING_EXPERIMENTS = MappingProxyType(
    {
        "topology_routed_multiscale": MappingProxyType(
            {
                "geometry_feature_mode": "full_pose",
                "geometry_bias_mode": "topology_routed_multiscale",
                "geometry_fusion_mode": "shared_qkv_multiscale",
                "geometry": True,
                "cross": False,
                "fusion_mode": "late",
            }
        ),
    }
)


__all__ = ["TOPOLOGY_ROUTING_EXPERIMENTS"]
