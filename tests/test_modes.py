# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Train/eval mode handling for both model classes (#209, #179).

``predict`` and ``update_support`` are inference entry points: they compute
in eval mode whatever mode the caller left the model in, and leave the model
in that mode afterwards; a GP ``predict`` in training mode used to accumulate
into the Laplace precision matrix (#209). ``Protonet.update_support`` must compute the global moments in either
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
    stale_cov = model.model.global_covariance.clone()
    model.train()
    support = eq.utils.generate_support(
        x, y, support_size=5, selected_labels=list(range(CLASSES))
    )
    model.model.update_support(support)
    assert model.model.global_mean.shape == stale_mean.shape
    assert not torch.equal(model.model.global_mean, stale_mean)
    assert model.model.global_covariance.shape == stale_cov.shape
    assert not torch.equal(model.model.global_covariance, stale_cov)
    assert torch.isfinite(model.model.global_covariance).all()
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


_MODES = pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])


def _set_mode(model, training):
    model.train(training)
    assert model.training is training and model.model.training is training


def _assert_mode(model, training):
    assert model.training is training
    assert model.model.training is training


@_MODES
@_BUILDERS
def test_predict_restores_callers_mode(build, training):
    """predict() computes in eval mode and leaves wrapper and inner module in
    the mode the caller had them in (#209)."""
    model, x, _ = build()
    _set_mode(model, training)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    _assert_mode(model, training)


@_MODES
def test_protonet_update_support_restores_callers_mode(training):
    model, x, y = _protonet()
    _set_mode(model, training)
    model.update_support(x, y.float(), 0.5)
    _assert_mode(model, training)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@_MODES
def test_gp_update_support_restores_callers_mode(training):
    model, x, y = _gp()
    _set_mode(model, training)
    precision = model.model.precision.clone()
    seen_data = model.model.seen_data.clone()
    model.update_support(x, y.long(), 10)
    _assert_mode(model, training)
    torch.testing.assert_close(model.model.precision, precision, atol=0, rtol=0)
    torch.testing.assert_close(model.model.seen_data, seen_data, atol=0, rtol=0)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_gp_manual_fine_tune_loop_with_predict_per_epoch():
    """A hand-written fine-tune loop (model.train(), reset_precision_matrix(),
    training batches through model(xs), predict on validation data at the end
    of each epoch) runs for two epochs: predict does not consume the training
    budget of the precision matrix and does not knock the model out of
    training mode (#209)."""
    model, x, y = _gp()
    val = x[:5]
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    out = None
    for _ in range(2):
        model.model.reset_precision_matrix()
        for start in range(0, x.shape[0], 32):
            opt.zero_grad()
            xs = x[start : start + 32]
            ys = y[start : start + 32].long()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
        assert model.training and model.model.training
        out = model.predict(val)
        assert model.training and model.model.training
    assert out is not None
    assert_valid_prediction(out, 5, CLASSES)
