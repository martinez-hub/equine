# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Train/eval mode handling for both model classes (#209, #179).

``predict`` and ``EquineProtonet.update_support`` are inference entry points
and must run in eval mode whatever mode the caller left the model in; a GP
``predict`` in training mode accumulates into the Laplace precision matrix
(#209). ``Protonet.update_support`` must compute the global moments in either
mode so that a freshly constructed (training-mode) model can be given a
support set (#179). ``train_model`` must leave the wrapper and the inner
module in the same (eval) mode. CPU only, seeded, short training like
``tests/test_inference.py``.
"""

import pytest
import torch
from conftest import BasicEmbeddingModel, assert_valid_prediction
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq


def _protonet():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    model.train_model(
        dataset,
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, x, y


def _gp():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    return model, x, y


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


def test_protonet_update_support_on_untrained_model():
    """A freshly constructed EquineProtonet accepts a support set (#179)."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    assert model.training  # nn.Module default: this is the failing configuration
    model.update_support(x, y.float(), 0.5)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_protonet_update_support_in_train_mode_computes_global_moments():
    """The inner Protonet recomputes the global moments in training mode too (#179).

    ``EquineProtonet.train_model`` calls the inner ``update_support`` once per
    episode in training mode; the moments must track the support in that mode
    as well, not only in eval mode.
    """
    model, x, y = _protonet()
    stale_mean = model.model.global_mean.clone()
    model.train()
    support = eq.utils.generate_support(
        x, y, support_size=5, selected_labels=list(range(CLASSES))
    )
    model.model.update_support(support)
    assert model.model.global_mean.shape == stale_mean.shape
    assert not torch.equal(model.model.global_mean, stale_mean)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_gp_predict_after_train_mode_does_not_touch_precision():
    """predict() after model.train() leaves the Laplace buffers alone (#209)."""
    model, x, _ = _gp()
    model.train()
    precision = model.model.precision.clone()
    seen_data = model.model.seen_data.clone()
    for _ in range(3):
        assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    torch.testing.assert_close(model.model.precision, precision, atol=0, rtol=0)
    torch.testing.assert_close(model.model.seen_data, seen_data, atol=0, rtol=0)


def test_gp_train_model_leaves_wrapper_and_inner_in_eval():
    model, _, _ = _gp()
    assert model.training is False
    assert model.model.training is False


def test_protonet_train_model_leaves_wrapper_and_inner_in_eval():
    model, _, _ = _protonet()
    assert model.training is False
    assert model.model.training is False


@_BUILDERS
def test_predict_leaves_model_in_eval(build):
    """predict() switches the wrapper and the inner module to eval mode (#209)."""
    model, x, _ = build()
    model.train()
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    assert model.training is False
    assert model.model.training is False
