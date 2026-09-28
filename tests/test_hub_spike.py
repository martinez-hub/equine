# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""D-191 spike: Hugging Face Hub layout (config.json + model.safetensors)."""

import json
import os

import pytest
import torch

import equine as eq

pytest.importorskip("safetensors")


def _train(embedding, use_temperature=False):
    torch.manual_seed(0)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    dataset = torch.utils.data.TensorDataset(X, Y)
    model = eq.EquineProtonet(embedding, 3, use_temperature=use_temperature)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _same(a, b, X):
    oa, ob = a.predict(X[:8]), b.predict(X[:8])
    assert torch.allclose(oa.classes, ob.classes, atol=1e-6)
    assert torch.allclose(oa.ood_scores, ob.ood_scores, atol=1e-6)


def test_save_pretrained_writes_hub_layout_and_round_trips(tmp_path) -> None:
    model, X = _train(eq.MLP(6, [16], 3), use_temperature=True)
    model.label_names = ["a", "b", "c"]
    out = str(tmp_path / "repo")
    model.save_pretrained(out)

    assert sorted(os.listdir(out)) == ["README.md", "config.json", "model.safetensors"]

    config = json.load(open(os.path.join(out, "config.json")))
    assert config["model_type"] == "EquineProtonet"
    assert config["library_name"] == "equine"
    assert config["embedding_recipe"]["builder"] == "equine.mlp"
    assert config["contains_executable"] is False
    assert config["label_names"] == ["a", "b", "c"]
    assert "tensor(" not in json.dumps(config)  # nothing but plain JSON

    from safetensors import safe_open

    with safe_open(os.path.join(out, "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert any('"embedding_state_dict"' in k for k in keys)
    assert any('"support", 0' in k for k in keys)  # int label keys preserved in paths
    assert any('"outlier_kde", 0, "dataset"' in k for k in keys)

    reloaded = eq.EquineProtonet.from_pretrained(out)
    assert isinstance(reloaded.embedding_model, eq.MLP)
    assert list(reloaded.model.support.keys()) == [0, 1, 2]  # ints, not strings
    assert reloaded.label_names == ["a", "b", "c"]
    assert reloaded.use_temperature is True
    _same(model, reloaded, X)

    # the generic loader accepts the directory too
    _same(model, eq.load_equine_model(out), X)


def test_executable_models_cannot_be_exported(tmp_path) -> None:
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(6, 3)

        def forward(self, x):
            return self.lin(x)

    model, _ = _train(torch.jit.script(Net()))
    with pytest.raises(ValueError, match="not a registered architecture"):
        model.save_pretrained(str(tmp_path / "repo"))


def test_from_pretrained_accepts_caller_architecture(tmp_path) -> None:
    model, X = _train(eq.MLP(6, [16], 3))
    out = str(tmp_path / "repo")
    model.save_pretrained(out)
    _same(model, eq.EquineProtonet.from_pretrained(out, embedding_model=eq.MLP(6, [16], 3)), X)
