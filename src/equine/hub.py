# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Hugging Face Hub layout for EQUINE models (D-191 spike).

A model directory holds ``config.json`` (every plain value of the checkpoint,
including the embedding recipe) and ``model.safetensors`` (every tensor). That
is the layout the Hub scans as safe and every PyTorch library on the Hub uses;
nothing in it can execute code. Models whose embedding is an executable
TorchScript archive have no recipe and cannot be exported.

Requires the ``hub`` extra: ``pip install "equine[hub]"``.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

import torch

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
CARD_FILE = "README.md"
_TENSOR = "__tensor__"
_INT_KEY_DICTS = "__int_key_dicts__"


def _require_safetensors():
    try:
        from safetensors import torch as safetensors_torch
    except ImportError as err:  # pragma: no cover - only without the extra
        raise ImportError(
            "Hub export needs the 'safetensors' package: pip install 'equine[hub]'"
        ) from err
    return safetensors_torch


def _split(
    node: Any,
    path: list[Any],
    tensors: dict[str, torch.Tensor],
    int_key_dicts: list[str],
) -> Any:
    """Replace every tensor with a placeholder, collecting tensors and int-keyed dicts."""
    if isinstance(node, torch.Tensor):
        key = json.dumps(path)
        tensors[key] = node.detach().cpu().contiguous()
        return {_TENSOR: key}
    if isinstance(node, dict):
        if any(isinstance(k, int) for k in node):
            int_key_dicts.append(json.dumps(path))
        return {
            str(k): _split(v, path + [k], tensors, int_key_dicts)
            for k, v in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [
            _split(v, path + [i], tensors, int_key_dicts) for i, v in enumerate(node)
        ]
    return node


def _merge(
    node: Any,
    path: list[Any],
    tensors: dict[str, torch.Tensor],
    int_key_dicts: set[str],
) -> Any:
    """Inverse of ``_split``: restore tensors and int dict keys."""
    if isinstance(node, dict) and set(node) == {_TENSOR}:
        return tensors[node[_TENSOR]]
    if isinstance(node, dict):
        int_keys = json.dumps(path) in int_key_dicts
        out: dict[Any, Any] = {}
        for k, v in node.items():
            key: Any = int(k) if int_keys else k
            out[key] = _merge(v, path + [key], tensors, int_key_dicts)
        return out
    if isinstance(node, list):
        return [
            _merge(v, path + [i], tensors, int_key_dicts) for i, v in enumerate(node)
        ]
    return node


def save_pretrained(model: Any, save_directory: str, card: bool = True) -> None:
    """
    Write ``model`` as a Hub-style directory: ``config.json`` + ``model.safetensors``.

    Parameters
    ----------
    model : EquineProtonet
        A model whose embedding is a registered architecture.
    save_directory : str
        Directory to create; existing files with the same names are overwritten.
    card : bool
        Also write a minimal model card (``README.md``) with library metadata.

    Raises
    ------
    ValueError
        If the model's embedding is an executable archive (no recipe).
    """
    safetensors_torch = _require_safetensors()
    checkpoint = model._to_checkpoint(
        allow_executable=False
    )  # refuses without a recipe
    tensors: dict[str, torch.Tensor] = {}
    int_key_dicts: list[str] = []
    config = _split(checkpoint, [], tensors, int_key_dicts)
    config[_INT_KEY_DICTS] = int_key_dicts
    config["model_type"] = type(model).__name__
    config["library_name"] = "equine"

    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, CONFIG_FILE), "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
    safetensors_torch.save_file(
        tensors, os.path.join(save_directory, WEIGHTS_FILE), metadata={"format": "pt"}
    )
    if card:
        builder = checkpoint["embedding_recipe"]["builder"]
        name = type(model).__name__
        with open(os.path.join(save_directory, CARD_FILE), "w") as fh:
            fh.write(
                "---\nlibrary_name: equine\ntags:\n- uncertainty-quantification\n"
                f"- {name.lower()}\n---\n\n# EQUINE {name}\n\n"
                f"Embedding architecture: `{builder}`. Load with:\n\n```python\n"
                f'import equine as eq\nmodel = eq.{name}.from_pretrained("<repo or directory>")\n```\n'
            )


def load_checkpoint_dir(load_directory: str) -> dict[str, Any]:
    """Read a Hub-style directory into the dictionary ``_from_checkpoint`` expects."""
    safetensors_torch = _require_safetensors()
    with open(os.path.join(load_directory, CONFIG_FILE)) as fh:
        config = json.load(fh)
    config.pop("model_type", None)
    config.pop("library_name", None)
    int_key_dicts = set(config.pop(_INT_KEY_DICTS, []))
    tensors = safetensors_torch.load_file(os.path.join(load_directory, WEIGHTS_FILE))
    return _merge(config, [], tensors, int_key_dicts)


def model_type_of(load_directory: str) -> Optional[str]:
    """The ``model_type`` recorded in a Hub-style directory's ``config.json``."""
    with open(os.path.join(load_directory, CONFIG_FILE)) as fh:
        return json.load(fh).get("model_type")


def is_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, CONFIG_FILE))


def push_to_hub(
    model: Any,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    **kwargs: Any,
) -> str:
    """
    Export ``model`` and upload the directory to the Hugging Face Hub.

    Thin wrapper over ``huggingface_hub.HfApi.upload_folder``; extra keyword
    arguments (``commit_message``, ``revision``, ...) are forwarded.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "push_to_hub needs 'huggingface_hub': pip install 'equine[hub]'"
        ) from err

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=private)
    with tempfile.TemporaryDirectory() as tmp:
        save_pretrained(model, tmp)
        info = api.upload_folder(
            repo_id=repo_id,
            folder_path=tmp,
            commit_message=kwargs.pop("commit_message", "Upload EQUINE model"),
            **kwargs,
        )
    return getattr(info, "commit_url", str(info))
