# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Device-parametrized behaviour tests for both model classes (#188).

Every test runs on CPU and on whichever accelerator this machine has (see
``conftest.available_devices()``); on CI (Ubuntu, no accelerator) only the
CPU case is collected. PR-2a marked, with a *strict* xfail, exactly the
cases that failed then (the fact table in
docs/superpowers/plans/2026-09-29-stack-2-device-correctness.md) and PR-2b
flipped them by removing the marks as it fixed each path; ``devices(xfail=)``
stays for later layers. A strict xfail that starts passing fails the run, so
a mark cannot go stale.
"""

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


def devices(
    *,
    xfail: str | None = None,
    raises: type[BaseException] | tuple[type[BaseException], ...] = RuntimeError,
    every_device: bool = False,
) -> list:
    """Parametrization over ``available_devices()``.

    When ``xfail`` is given, the accelerator case (every case, with
    ``every_device=True``) gets a strict xfail with that reason and the
    exception type(s) observed today in ``raises``, so every test states next
    to its signature whether (and why) it fails today. Otherwise the CPU case
    is plain; it is never marked for an accelerator-only failure.
    """
    params = []
    for device in available_devices():
        marks = []
        if xfail is not None and (every_device or device != "cpu"):
            marks.append(pytest.mark.xfail(strict=True, raises=raises, reason=xfail))
        params.append(pytest.param(device, marks=marks))
    return params


def _protonet(device: str, use_temperature: bool = False):
    """Short float32 training on ``device``; the float64 ``golden_data.trained_protonet`` must stay byte-stable and takes no device."""
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
    """Short float32 training on ``device``; the float64 ``golden_data.trained_gp`` must stay byte-stable and takes no device."""
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


def _assert_same_predictions(
    expected: eq.EquineOutput, actual: eq.EquineOutput
) -> None:
    torch.testing.assert_close(
        actual.classes.cpu(), expected.classes.cpu(), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        actual.ood_scores.cpu(), expected.ood_scores.cpu(), atol=1e-5, rtol=0
    )


# --- EquineProtonet -----------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_protonet_trains_and_predicts(device):
    model, x, _ = _protonet(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
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
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))

    explicit = eq.EquineProtonet.load(path, device)
    _assert_same_predictions(before, explicit.predict(x[:5]))


# --- both classes -------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_stored_tensors_on_device(build, device):
    model, _, _ = build(device)
    assert_on_device(model, device)


# --- EquineGP -----------------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_gp_trains(device):
    model, _, _ = _gp(device)
    assert model.model.precision.device.type == torch.device(device).type


@pytest.mark.parametrize("device", devices())
def test_gp_predicts(device):
    model, x, _ = _gp(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_update_support(device):
    model, x, y = _gp(device)
    model.update_support(x, y.long(), 10)
    assert set(model.support) == set(range(CLASSES))
    assert model.prototypes.shape[0] == CLASSES  # one prototype per class


@pytest.mark.parametrize("device", devices())
def test_gp_vis_support_training(device):
    model, _, _ = _gp(device, vis_support=True, support_size=10)
    assert set(model.support) == set(range(CLASSES))


@pytest.mark.parametrize("device", devices())
def test_gp_save_load_round_trip(device, tmp_path):
    model, x, _ = _gp(device)
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))


@pytest.mark.parametrize("device", devices())
def test_gp_load_onto_device(device, tmp_path):
    model, x, _ = _gp("cpu")
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    loaded = eq.EquineGP.load(path, device)
    assert loaded.device == device
    assert_on_device(loaded, device)
    _assert_same_predictions(before, loaded.predict(x[:5]))

    generic = eq.load_equine_model(path, device=device)
    assert generic.device == device
    assert_on_device(generic, device)
    _assert_same_predictions(before, generic.predict(x[:5]))


# --- input dtype --------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_predict_accepts_float64_input(build, device):
    """Inputs are cast to the embedding's parameter dtype at the model boundary."""
    model, x, _ = build(device)
    assert_valid_prediction(model.predict(x[:5].double()), 5, CLASSES)


class _TokenEmbedding(torch.nn.Module):
    """Embedding model whose input is integer token ids, mean-pooled then projected."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=50, embedding_dim=8)
        self.linear = torch.nn.Linear(8, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(x).mean(dim=1))


@pytest.mark.parametrize("device", devices())
def test_integer_inputs_are_moved_but_not_cast(device):
    """The dtype cast applies to floating inputs only: nn.Embedding indices stay integer."""
    torch.manual_seed(0)
    y = torch.arange(90) % CLASSES
    # class-specific ids, all below the 50 rows of the embedding table
    x = torch.randint(0, 47, (90, 4)) // CLASSES * CLASSES + y[:, None]
    model = eq.EquineProtonet(_TokenEmbedding(), CLASSES, device=device)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert x.dtype == torch.int64
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


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
        ),
    ],
)
def test_device_attribute_is_a_string(make):
    assert isinstance(make().device, str)
