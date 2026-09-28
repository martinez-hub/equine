# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""D-191 spike: embedding architectures as recipes rebuilt through a registry."""

import subprocess
import sys

import pytest
import torch

import equine as eq


def _dataset(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    return torch.utils.data.TensorDataset(X, Y), X


def _train(embedding):
    dataset, X = _dataset()
    model = eq.EquineProtonet(embedding, 3)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _same(a, b, X):
    oa, ob = a.predict(X[:8]), b.predict(X[:8])
    assert torch.allclose(oa.classes, ob.classes, atol=1e-6)
    assert torch.allclose(oa.ood_scores, ob.ood_scores, atol=1e-6)


class UnregisteredNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(6, 3)

    def forward(self, x):
        return self.lin(x)


# --- recipes ---------------------------------------------------------------


def test_shipped_mlp_records_its_recipe() -> None:
    m = eq.MLP(6, [16], 3)
    assert eq.embedding_recipe(m) == {
        "builder": "equine.mlp",
        "kwargs": {
            "in_features": 6,
            "hidden_sizes": [16],
            "out_features": 3,
            "activation": "relu",
        },
    }
    assert "equine.mlp" in eq.registered_architectures()


def test_non_plain_constructor_argument_is_rejected() -> None:
    @eq.embedding_architecture("equine.tests.bad_args")
    class TakesTensor(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

    with pytest.raises(TypeError, match="plain value"):
        TakesTensor(torch.zeros(2))


def test_duplicate_name_for_a_different_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        eq.register_embedding_architecture("equine.mlp", UnregisteredNet)


# --- save / load -------------------------------------------------------------


def test_registered_model_saves_data_only_and_round_trips(tmp_path) -> None:
    model, X = _train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)

    ckpt = torch.load(path, weights_only=True)
    assert ckpt["contains_executable"] is False
    assert "embed_jit_save" not in ckpt
    assert ckpt["embedding_recipe"]["builder"] == "equine.mlp"

    reloaded = eq.load_equine_model(path)
    assert isinstance(reloaded.embedding_model, eq.MLP)
    _same(model, reloaded, X)


def test_unregistered_model_cannot_be_saved_without_opt_in(tmp_path) -> None:
    model, _ = _train(UnregisteredNet())
    with pytest.raises(ValueError, match="allow_executable=True"):
        model.save(str(tmp_path / "m.eq"))


def test_executable_opt_in_is_flagged_and_needs_trust(tmp_path) -> None:
    model, X = _train(torch.jit.script(UnregisteredNet()))  # what the web app supplies
    path = str(tmp_path / "m.eq")
    model.save(path, allow_executable=True)

    ckpt = torch.load(path, weights_only=True)  # still safe to *read*
    assert ckpt["contains_executable"] is True

    with pytest.raises(ValueError, match="trust_executable=True"):
        eq.load_equine_model(path)
    with pytest.raises(ValueError, match="trust_executable=True"):
        eq.EquineProtonet.load(path)

    _same(model, eq.load_equine_model(path, trust_executable=True), X)
    _same(model, eq.EquineProtonet.load(path, trust_executable=True), X)


def test_caller_supplied_architecture_overrides_recipe(tmp_path) -> None:
    model, X = _train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    fresh = eq.MLP(6, [16], 3)  # weights from the file are loaded into it
    _same(model, eq.load_equine_model(path, embedding_model=fresh), X)


def test_unknown_recipe_gives_actionable_error_in_another_process(tmp_path) -> None:
    """A class registered in this process is not known to a fresh interpreter."""

    @eq.embedding_architecture("equine.tests.only_here")
    class OnlyHere(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)

        def forward(self, x):
            return self.lin(x)

    model, _ = _train(OnlyHere())
    path = str(tmp_path / "m.eq")
    model.save(path)

    code = (
        "import equine as eq, sys\n"
        "try:\n"
        f"    eq.load_equine_model({path!r})\n"
        "except ValueError as e:\n"
        "    print(e); sys.exit(3)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert "equine.tests.only_here" in proc.stdout
    assert "not registered" in proc.stdout
    assert "embedding_model=" in proc.stdout
