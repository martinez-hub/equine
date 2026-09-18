# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Tests that EQUINE model files are read without unrestricted unpickling, so
that opening an untrusted file cannot run pickle payloads (see issue #168).
"""

import io
import os
import warnings

import numpy as np
import pytest
import torch
from conftest import BasicEmbeddingModel

import equine as eq
from equine.utils import prepare_jit_module


class _MakesDirectoryWhenUnpickled:
    """Stand-in for a malicious pickle payload: unpickling it creates a directory."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def _tiny_dataset(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    return torch.utils.data.TensorDataset(X, Y), X


def _trained_protonet(cov_type=eq.CovType.UNIT, use_temperature=False):
    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(6, 3), 3, cov_type=cov_type, use_temperature=use_temperature
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _trained_gp():
    dataset, X = _tiny_dataset()
    model = eq.EquineGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
    )
    return model, X


def _assert_same_predictions(a, b, X):
    out_a, out_b = a.predict(X[1:10]), b.predict(X[1:10])
    assert torch.allclose(out_a.classes, out_b.classes, atol=1e-6)
    assert torch.allclose(out_a.ood_scores, out_b.ood_scores, atol=1e-6)


def _jit_buffer(module: torch.nn.Module) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(prepare_jit_module(module)), buffer)
    buffer.seek(0)
    return buffer


def _write_legacy_protonet_file(model, path: str) -> None:
    """Reproduce the pre-#168 on-disk layout: BytesIO, Enum and scipy objects."""
    torch.save(
        {
            "embed_jit_save": _jit_buffer(model.model.embedding_model),
            "feature_names": model.feature_names,
            "label_names": model.label_names,
            "model_head_save": model.model.model_head.state_dict(),
            "outlier_kde": model.outlier_score_kde,
            "settings": {
                "cov_type": model.cov_type,
                "emb_out_dim": model.emb_out_dim,
                "use_temperature": model.use_temperature,
                "init_temperature": model.temperature.item(),
                "relative_mahal": model.relative_mahal,
                "device": model.device,
            },
            "support": model.model.support,
            "train_summary": model.train_summary,
        },
        path,
    )


def _write_legacy_gp_file(model, path: str) -> None:
    """Reproduce the pre-#168 EquineGP layout: BytesIO buffer and raw OrderedDicts."""
    laplace_sd = {
        k: v
        for k, v in model.model.state_dict().items()
        if "feature_extractor" not in k
    }
    torch.save(
        {
            "embed_jit_save": _jit_buffer(model.model.feature_extractor),
            "feature_names": model.feature_names,
            "label_names": model.label_names,
            "laplace_model_save": laplace_sd,
            "num_data": model.model.num_data,
            "settings": {
                "emb_out_dim": model.num_deep_features,
                "num_classes": model.num_outputs,
                "num_random_features": model.num_random_features,
                "init_temperature": model.temperature.item(),
                "device": model.device_type,
            },
            "support": model.support,
            "train_batch_size": model.model.train_batch_size,
            "train_summary": model.train_summary,
        },
        path,
    )


LEGACY_CASES = [
    pytest.param(_trained_protonet, _write_legacy_protonet_file, id="protonet"),
    pytest.param(_trained_gp, _write_legacy_gp_file, id="gp"),
]


# ---------------------------------------------------------------------------
# Untrusted files must not run pickle payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader",
    [
        lambda p: eq.load_equine_model(p),
        lambda p: eq.EquineProtonet.load(p),
        lambda p: eq.EquineGP.load(p),
    ],
    ids=["load_equine_model", "EquineProtonet.load", "EquineGP.load"],
)
def test_loading_untrusted_file_does_not_execute_payload(tmp_path, loader) -> None:
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.eq"
    torch.save(
        {
            "train_summary": {"modelType": "EquineProtonet"},
            "payload": _MakesDirectoryWhenUnpickled(str(marker)),
        },
        path,
    )

    with pytest.raises(ValueError, match="safely"):
        loader(str(path))

    assert not marker.exists(), "unpickling the model file executed its payload"


# ---------------------------------------------------------------------------
# Files written by save() must be loadable with torch.load(weights_only=True)
# ---------------------------------------------------------------------------


def _assert_weights_only_safe_layout(path: str) -> dict:
    checkpoint = torch.load(path, weights_only=True)  # must not raise
    # `bytes` is only accepted by the weights-only unpickler from torch 2.5 on,
    # so the TorchScript archive has to travel as a uint8 tensor.
    archive = checkpoint["embed_jit_save"]
    assert isinstance(archive, torch.Tensor) and archive.dtype == torch.uint8
    assert checkpoint["equine_format_version"] == 2
    return checkpoint


def test_protonet_save_format_is_weights_only_safe(tmp_path) -> None:
    model, X = _trained_protonet(cov_type=eq.CovType.DIAGONAL, use_temperature=True)
    # A non-default bandwidth on one class makes the test sensitive to the
    # bandwidth factor being dropped or misapplied on reload.
    model.outlier_score_kde[1].set_bandwidth(0.3)
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    checkpoint = _assert_weights_only_safe_layout(path)
    assert checkpoint["settings"]["cov_type"] == "diag"

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the safe path must not warn
        reloaded = eq.load_equine_model(path)

    assert isinstance(reloaded, eq.EquineProtonet)
    assert reloaded.cov_type is eq.CovType.DIAGONAL
    assert reloaded.use_temperature is True
    assert list(reloaded.outlier_score_kde.keys()) == list(
        model.outlier_score_kde.keys()
    )
    for label, kde in model.outlier_score_kde.items():
        rebuilt = reloaded.outlier_score_kde[label]
        assert rebuilt.factor == pytest.approx(kde.factor)
        np.testing.assert_allclose(rebuilt.covariance, kde.covariance)
        np.testing.assert_array_equal(rebuilt.dataset, kde.dataset)
    assert reloaded.outlier_score_kde[1].factor == pytest.approx(0.3)
    _assert_same_predictions(model, reloaded, X)


def test_gp_save_format_is_weights_only_safe(tmp_path) -> None:
    model, X = _trained_gp()
    path = str(tmp_path / "gp.eq")
    model.save(path)

    _assert_weights_only_safe_layout(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = eq.load_equine_model(path)

    assert isinstance(reloaded, eq.EquineGP)
    assert list(reloaded.support.keys()) == list(model.support.keys())
    _assert_same_predictions(model, reloaded, X)


# ---------------------------------------------------------------------------
# Legacy files (pre-safe format) need an explicit opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_legacy_file_is_rejected_by_default(tmp_path, train, write_legacy) -> None:
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with pytest.raises(ValueError, match="allow_unsafe_legacy_format"):
        eq.load_equine_model(path)
    with pytest.raises(ValueError, match="allow_unsafe_legacy_format"):
        type(model).load(path)


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_legacy_file_loads_with_opt_in_and_resaves_safely(
    tmp_path, train, write_legacy
) -> None:
    model, X = train()
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        reloaded = eq.load_equine_model(legacy_path, allow_unsafe_legacy_format=True)
    unsafe = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(unsafe) == 1, "exactly one unsafe-load warning per load"
    assert "unsafe" in str(unsafe[0].message)
    assert unsafe[0].filename == __file__, "warning must point at the caller"
    _assert_same_predictions(model, reloaded, X)

    safe_path = str(tmp_path / "resaved.eq")
    reloaded.save(safe_path)
    _assert_weights_only_safe_layout(safe_path)
    _assert_same_predictions(model, eq.load_equine_model(safe_path), X)


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_class_load_opt_in_warning_points_at_caller(
    tmp_path, train, write_legacy
) -> None:
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        type(model).load(path, allow_unsafe_legacy_format=True)
    unsafe = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(unsafe) == 1
    assert unsafe[0].filename == __file__


def test_opt_in_still_uses_safe_path_for_safe_files(tmp_path) -> None:
    """The flag is a fallback, not a switch: a safe-format file is never unpickled
    without restrictions, so it must load silently even when the flag is set."""
    model, X = _trained_protonet()
    path = str(tmp_path / "safe.eq")
    model.save(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = eq.load_equine_model(path, allow_unsafe_legacy_format=True)
    _assert_same_predictions(model, reloaded, X)


def test_opt_in_does_not_bypass_safe_path_for_safe_layout_with_payload(
    tmp_path,
) -> None:
    """A file that declares the safe format but smuggles a payload is not an
    EQUINE model; with the flag set it falls back to unrestricted unpickling by
    design, so callers must only set the flag for files they trust. This test
    pins that the *default* (flag off) never reaches the payload."""
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.eq"
    torch.save(
        {
            "equine_format_version": 2,
            "train_summary": {"modelType": "EquineProtonet"},
            "payload": _MakesDirectoryWhenUnpickled(str(marker)),
        },
        path,
    )
    with pytest.raises(ValueError, match="safely"):
        eq.load_equine_model(str(path))
    assert not marker.exists()


def test_load_jit_archive_accepts_all_stored_forms() -> None:
    """The archive loader reads the uint8 tensor written today as well as the
    bytes and BytesIO forms found in files written by earlier versions."""
    from equine.utils import jit_archive_to_tensor, load_jit_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.rand(4, 6)
    expected = module(x)

    for archive in (jit_archive_to_tensor(buffer), buffer.getvalue(), buffer):
        rebuilt = load_jit_archive(archive)
        assert torch.allclose(rebuilt(x), expected, atol=1e-6)
