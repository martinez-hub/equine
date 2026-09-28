# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

from . import architectures as _architectures  # noqa: F401  registers equine.* builders
from .architectures import MLP
from .equine import Equine, EquineOutput
from .equine_gp import EquineGP
from .equine_protonet import CovType, EquineProtonet
from .load_equine_model import load_equine_model
from .registry import (
    embedding_architecture,
    embedding_recipe,
    register_embedding_architecture,
    registered_architectures,
)
from .utils import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    generate_episode,
    generate_model_metrics,
    generate_model_summary,
    generate_support,
    generate_train_summary,
    mahalanobis_distance_nosq,
)

if not TYPE_CHECKING:  # pragma: no cover
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str

__all__ = [
    "Equine",
    "EquineOutput",
    "EquineGP",
    "EquineProtonet",
    "CovType",
    "MLP",
    "embedding_architecture",
    "embedding_recipe",
    "register_embedding_architecture",
    "registered_architectures",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "load_equine_model",
    "generate_support",
    "generate_episode",
    "generate_model_metrics",
    "generate_train_summary",
    "generate_model_summary",
    "mahalanobis_distance_nosq",
]
