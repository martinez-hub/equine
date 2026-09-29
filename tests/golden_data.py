# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Deterministic data and models for golden-value and cross-version tests.

Everything here is seeded so that the same code produces the same numbers on
CPU across runs. Golden literals in tests/test_golden.py and the fixture files
in tests/fixtures/ are produced by these functions.
"""

import torch
from conftest import BasicEmbeddingModel

import equine as eq

ROWS, FEATURES, CLASSES = 120, 6, 3
SEED = 0


def separable_dataset(seed: int = SEED):
    """Uniform noise plus a class-dependent shift of 3 on the first `CLASSES` features.

    Labels are float, which is what ``EquineProtonet.train_model`` expects;
    ``EquineGP.train_model`` casts them to long itself.
    """
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES).repeat_interleave(ROWS // CLASSES)
    x = torch.rand(ROWS, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return torch.utils.data.TensorDataset(x, y.float()), x, y


def query_batch(seed: int = SEED + 1) -> torch.Tensor:
    """Fixed in-distribution queries: one per class, same construction as the training data."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES)
    x = torch.rand(CLASSES, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return x


def far_ood_batch() -> torch.Tensor:
    """Queries shifted by +6 on every feature: far from every class."""
    return query_batch() + 6.0


def trained_protonet(seed: int = SEED) -> eq.EquineProtonet:
    torch.manual_seed(seed)
    dataset, _, _ = separable_dataset(seed)
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(
        dataset,
        num_episodes=20,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model


def trained_gp(seed: int = SEED) -> eq.EquineGP:
    """Small GP trained long enough to be confident on in-distribution queries.

    Five epochs at lr=0.01 left every class probability near 1/3 and every
    OOD score near 1 (degenerate golden values); 40 epochs at lr=0.05 gives
    argmax-correct predictions and a clear in-distribution vs far-OOD gap.
    """
    torch.manual_seed(seed)
    dataset, _, _ = separable_dataset(seed)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=40,
        batch_size=32,
        vis_support=True,
        support_size=10,
    )
    return model
