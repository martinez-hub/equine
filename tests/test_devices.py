# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Device-parametrized behaviour tests for both model classes (#188).

Every test runs on CPU and on whichever accelerator this machine has (see
``conftest.available_devices()``); on CI (Ubuntu, no accelerator) only the
CPU case is collected. The CPU case is never marked. The accelerator case
carries a *strict* xfail only where the path is known to fail today (the
fact table in docs/superpowers/plans/2026-09-29-stack-2-device-correctness.md);
the fixes in PR-2b flip those cases to passing by removing the marks, and a
strict xfail that starts passing fails the run so a mark cannot go stale.
"""

import os

import pytest
import torch
from conftest import (
    BasicEmbeddingModel,
    assert_on_device,
    assert_valid_prediction,
    available_devices,
)
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq

pytestmark = pytest.mark.device

# What the accelerator paths raise today: device-mismatch RuntimeErrors, the
# missing MPS kernel for cholesky_inverse, the MPS float64 TypeError and the
# assertion in assert_on_device. PR-2b tightens this per test as it fixes them.
_FAILS_TODAY = (RuntimeError, NotImplementedError, TypeError, AssertionError)


def devices(*, accelerator_xfail: str | None = None) -> list:
    """Parametrization over ``available_devices()``.

    The CPU case is always plain. The accelerator case gets a strict xfail
    with ``accelerator_xfail`` as its reason when one is given, so every test
    states next to its signature whether (and why) it fails on the
    accelerator today.
    """
    params = []
    for device in available_devices():
        marks = []
        if device != "cpu" and accelerator_xfail is not None:
            marks.append(
                pytest.mark.xfail(
                    strict=True, raises=_FAILS_TODAY, reason=accelerator_xfail
                )
            )
        params.append(pytest.param(device, marks=marks))
    return params


def _protonet(device: str, use_temperature: bool = False):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        use_temperature=use_temperature,
        device=device,
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


def _gp(device: str, **train_kwargs):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
        device=device,
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
        **train_kwargs,
    )
    return model, x, y


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


def _assert_same_predictions(a: eq.EquineOutput, b: eq.EquineOutput) -> None:
    torch.testing.assert_close(b.classes.cpu(), a.classes.cpu(), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        b.ood_scores.cpu(), a.ood_scores.cpu(), atol=1e-5, rtol=0
    )


# --- EquineProtonet -----------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_protonet_trains_and_predicts(device):
    model, x, _ = _protonet(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize(
    "device", devices(accelerator_xfail="#170: temperature buffer stays on CPU")
)
def test_protonet_with_temperature_predicts(device):
    model, x, _ = _protonet(device, use_temperature=True)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_update_support(device):
    model, x, y = _protonet(device)
    model.update_support(x, y.float(), 0.5)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_save_load_round_trip(device, tmp_path):
    model, x, _ = _protonet(device)
    before = model.predict(x[:5])
    path = os.path.join(tmp_path, "protonet.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))

    explicit = eq.EquineProtonet.load(path, device)
    _assert_same_predictions(before, explicit.predict(x[:5]))


# --- both classes -------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize(
    "device",
    devices(
        accelerator_xfail="#170: temperature registered after the module was moved"
    ),
)
def test_stored_tensors_on_device(build, device):
    model, _, _ = build(device)
    assert_on_device(model, device)


# --- EquineGP -----------------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_gp_trains(device):
    model, _, _ = _gp(device)
    assert model.model.precision.device.type == torch.device(device).type


# EquineGP.forward already moves X to the device, so the predict paths get
# past compute_embeddings (#177) and fail on the missing MPS kernel (#173).
_GP_NO_CHOLESKY_KERNEL = (
    "#173: aten::cholesky_inverse has no MPS kernel (predict moves its input)"
)


@pytest.mark.parametrize("device", devices(accelerator_xfail=_GP_NO_CHOLESKY_KERNEL))
def test_gp_predicts(device):
    model, x, _ = _gp(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize(
    "device",
    devices(accelerator_xfail="#177: compute_embeddings does not move its input"),
)
def test_gp_update_support(device):
    model, x, y = _gp(device)
    model.update_support(x, y.long(), 10)
    assert set(model.support) == set(range(CLASSES))
    assert model.prototypes.shape[0] == CLASSES  # one prototype per class


@pytest.mark.parametrize(
    "device",
    devices(accelerator_xfail="#177: compute_embeddings does not move its input"),
)
def test_gp_vis_support_training(device):
    model, _, _ = _gp(device, vis_support=True, support_size=10)
    assert set(model.support) == set(range(CLASSES))


@pytest.mark.parametrize("device", devices(accelerator_xfail=_GP_NO_CHOLESKY_KERNEL))
def test_gp_save_load_round_trip(device, tmp_path):
    model, x, _ = _gp(device)
    before = model.predict(x[:5])
    path = os.path.join(tmp_path, "gp.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))


@pytest.mark.skip(reason="#188: EquineGP.load(device=) is added in PR-2b")
@pytest.mark.parametrize("device", devices())
def test_gp_load_onto_device(device, tmp_path):
    model, x, _ = _gp("cpu")
    before = model.predict(x[:5])
    path = os.path.join(tmp_path, "gp.eq")
    model.save(path)

    loaded = eq.EquineGP.load(path, device)
    assert_on_device(loaded, device)
    _assert_same_predictions(before, loaded.predict(x[:5]))


# --- input dtype --------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize(
    "device",
    devices(
        accelerator_xfail=("MPS has no float64; PR-2b casts inputs to the model dtype")
    ),
)
def test_predict_accepts_float64_input(build, device):
    model, x, _ = build(device)
    assert_valid_prediction(model.predict(x[:5].double()), 5, CLASSES)


# --- device attribute ---------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda: eq.EquineProtonet(
                BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, device="cpu"
            ),
            id="protonet",
        ),
        pytest.param(
            lambda: eq.EquineGP(
                BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, device="cpu"
            ),
            id="gp",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="#216: EquineGP.device is a torch.device",
            ),
        ),
    ],
)
def test_device_attribute_is_a_string(make):
    assert isinstance(make().device, str)
