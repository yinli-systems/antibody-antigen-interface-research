"""Isolated registry for post-confirmatory local-pose experiments.

This registry is intentionally not imported by ``train_framecontact.py``.
The frozen eight-condition confirmatory registry therefore cannot acquire a new
condition as a side effect of importing the core trainer.
"""

from __future__ import annotations

from types import MappingProxyType


EXPLORATORY_EXPERIMENTS = MappingProxyType(
    {
        "local_pose_multibasis": MappingProxyType(
            {
                "geometry_feature_mode": "full_pose",
                "geometry_bias_mode": "local_pose_multibasis",
                "geometry": True,
                "cross": False,
                "fusion_mode": "late",
            }
        ),
        "local_pose_rotary": MappingProxyType(
            {
                "geometry_feature_mode": "full_pose",
                "geometry_bias_mode": "local_pose_rotary",
                "geometry": True,
                "cross": False,
                "fusion_mode": "late",
            }
        ),
    }
)


__all__ = ["EXPLORATORY_EXPERIMENTS"]
