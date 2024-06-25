from .base import (BatchedTensors, MultiModalData, MultiModalInputs,
                   MultiModalPlugin)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to its modality and the target model.
"""

__all__ = [
    "BatchedTensors", "MultiModalData", "MultiModalInputs", "MultiModalPlugin",
    "MULTIMODAL_REGISTRY", "MultiModalRegistry"
]
