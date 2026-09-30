# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Inference-path tests for both model classes (#173, #182, #212).

``predict`` must run the embedding model once per call and return tensors
that carry no autograd graph (it runs under ``torch.no_grad()``, not
``torch.inference_mode()``, so a caller can still feed the outputs into
autograd). ``update_support`` must embed each support class once and, for
the Protonet, the calibration set once. CPU only, seeded, short training
like ``tests/test_devices.py``.
"""

import pytest
import torch
from conftest import CountingEmbedding
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq


def _protonet():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    emb = CountingEmbedding(FEATURES, CLASSES)
    model = eq.EquineProtonet(emb, CLASSES, use_temperature=True)
    model.train_model(
        dataset,
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, emb, x, y


def _gp():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    emb = CountingEmbedding(FEATURES, CLASSES)
    model = eq.EquineGP(emb, CLASSES, CLASSES, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    return model, emb, x, y


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


@_BUILDERS
def test_predict_embeds_once(build):
    model, emb, x, _ = build()
    emb.calls = 0
    model.predict(x[:5])
    assert emb.calls == 1


@_BUILDERS
def test_predict_outputs_carry_no_autograd(build):
    model, _, x, _ = build()
    out = model.predict(x[:5])
    assert out.classes.grad_fn is None
    assert out.embeddings.grad_fn is None
    assert not out.ood_scores.requires_grad


@_BUILDERS
def test_predict_outputs_are_plain_tensors_usable_in_autograd(build):
    """Guards against ``torch.inference_mode()``: its tensors cannot enter autograd."""
    model, _, x, _ = build()
    out = model.predict(x[:5])
    leaf = out.embeddings.clone().requires_grad_()
    (leaf * 2).sum().backward()


def test_gp_update_support_embeds_once_per_class():
    model, emb, x, y = _gp()
    emb.calls = 0
    model.update_support(x, y.long(), 10)
    assert emb.calls == CLASSES


def test_protonet_update_support_embeds_calibration_set_once():
    """One pass over the calibration set plus one per support class."""
    model, emb, x, y = _protonet()
    emb.calls = 0
    model.update_support(x, y.float(), 0.5)
    assert emb.calls == 1 + CLASSES
