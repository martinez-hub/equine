# PR-0b: Persistence as Architecture-as-Code — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save an EQUINE model's embedding network as a recipe (registered architecture name plus plain constructor arguments) and a `state_dict`, rebuilt through an allowlist registry, so a default `.eq` file contains nothing executable; keep TorchScript only as a flagged, opt-in read/write bridge for one release.

**Architecture:** A new `equine.registry` module records constructor arguments on instances of decorated classes and rebuilds them by name. Each model class defines its file format in one place (`_to_checkpoint` / `_from_checkpoint`) and a shared `_rebuild_embedding` chooses, in order, a caller-supplied module, the recipe, or the flagged TorchScript archive behind `trust_executable`. `load_equine_model` only dispatches. Spec: `docs/superpowers/specs/2026-09-28-d191-persistence-design.md`.

**Tech Stack:** Python 3.10+, PyTorch (`torch.load(weights_only=True)`, `torch.jit` for the bridge only), beartype (both model classes are class-decorated, so every new signature is type-checked at runtime), pytest + pytest-xdist, ruff, codespell.

**Branch and workflow:** work on `feat/persistence-as-code` on the fork (`martinez-hub/equine`), cut from `fix/safe-model-loading`. Open the PR against the fork's `main`. Do not push to `mit-ll-responsible-ai/equine`. Do not edit `README.md`.

**Environment:** a venv with the package installed in editable mode and the `tests` extra:
```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[tests]" pytest-xdist
```
Run tests with `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 4`. Lint with `uvx ruff check src tests && uvx ruff format --check src tests && uvx codespell src docs`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/equine/registry.py` (new) | Registry: decorator, function registration, recipe recording and validation, `build_from_recipe`, `registered_architectures`, `embedding_recipe`. No knowledge of EQUINE models. |
| `src/equine/architectures.py` (new) | Builders that ship with EQUINE: `MLP` registered as `equine.mlp`. |
| `src/equine/__init__.py` | Export the registry API and `MLP`; import `architectures` for its registration side effect. |
| `src/equine/equine_protonet.py` | `_to_checkpoint`, `_from_checkpoint`, `_rebuild_embedding`, `save(allow_executable)`, `load(device, allow_unsafe_legacy_format, trust_executable, embedding_model)`. |
| `src/equine/equine_gp.py` | Same four methods. `load` keeps its current parameters and gains the two new flags; the `device` parameter arrives in PR-2b (Phase 2). |
| `src/equine/load_equine_model.py` | Forward `trust_executable` and `embedding_model`. |
| `tests/conftest.py` | `BasicEmbeddingModel` gets the decorator (the one-line change that existing tests need). |
| `tests/test_registry.py` (new) | Registry behaviour. |
| `tests/test_persistence.py` (new) | Recipe round trips, transition bridge, migration, override, unknown recipe in a subprocess; parametrized over both model classes. |
| `tests/test_safe_loading.py` | Adjust the layout helper and the legacy re-save test for the recipe layout. |
| `CHANGELOG.md` (new) | Release note with the migration command and the removal schedule. |

---

### Task 1: Registry module

> **Post-review note (2026-09-28):** the code below was implemented, then a code-quality review required these additions, applied in a follow-up commit on the branch: only instances of the exact registered class record a recipe (an unregistered subclass records nothing); plain-value checks use exact types; recorded kwargs are deep-copied; positional-only parameters and classes without their own `__init__` are refused at registration; redefinition of the same module+qualname replaces the entry with a warning; registering under a second name raises; a `threading.Lock` guards registration; `build_from_recipe` validates the recipe shape and wraps constructor `TypeError`s; TypeVar-typed decorator; test isolation via an autouse fixture restoring `_REGISTRY`. The committed `src/equine/registry.py` is the reference; the block below is the pre-review version.

**Files:**
- Create: `src/equine/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Registry of embedding architectures: recipes recorded at construction, rebuilt by name."""

import pytest
import torch

from equine import registry


def _fresh_name(suffix: str) -> str:
    return f"equine.tests.registry.{suffix}"


def test_decorated_class_records_recipe_with_defaults_applied() -> None:
    @registry.embedding_architecture(_fresh_name("defaults"))
    class Net(torch.nn.Module):
        def __init__(self, width: int, depth: int = 2, act: str = "relu") -> None:
            super().__init__()
            self.lin = torch.nn.Linear(width, width)

    net = Net(8)
    assert registry.embedding_recipe(net) == {
        "builder": _fresh_name("defaults"),
        "kwargs": {"width": 8, "depth": 2, "act": "relu"},
    }
    assert _fresh_name("defaults") in registry.registered_architectures()


def test_build_from_recipe_calls_registered_constructor() -> None:
    @registry.embedding_architecture(_fresh_name("build"))
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(width, 1)

    built = registry.build_from_recipe({"builder": _fresh_name("build"), "kwargs": {"width": 5}})
    assert isinstance(built, Net)
    assert built.lin.in_features == 5


def test_non_plain_constructor_argument_is_rejected_at_construction() -> None:
    @registry.embedding_architecture(_fresh_name("nonplain"))
    class Net(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

    with pytest.raises(TypeError, match="'weight'"):
        Net(torch.zeros(2))


def test_var_positional_constructor_is_rejected_at_registration() -> None:
    class Net(torch.nn.Module):
        def __init__(self, *sizes: int) -> None:
            super().__init__()

    with pytest.raises(TypeError, match=r"\*args"):
        registry.register_embedding_architecture(_fresh_name("varargs"), Net)


def test_var_keyword_arguments_are_flattened_into_recipe() -> None:
    @registry.embedding_architecture(_fresh_name("varkw"))
    class Net(torch.nn.Module):
        def __init__(self, width: int, **extra: int) -> None:
            super().__init__()

    assert registry.embedding_recipe(Net(3, depth=4))["kwargs"] == {"width": 3, "depth": 4}


def test_duplicate_name_for_different_class_is_rejected_and_same_class_is_noop() -> None:
    name = _fresh_name("dup")

    @registry.embedding_architecture(name)
    class A(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class B(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    registry.register_embedding_architecture(name, A)  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        registry.register_embedding_architecture(name, B)


def test_unknown_recipe_error_lists_registered_names_and_override() -> None:
    with pytest.raises(ValueError) as err:
        registry.build_from_recipe({"builder": "nobody.home", "kwargs": {}})
    message = str(err.value)
    assert "'nobody.home'" in message
    assert "not registered" in message
    assert "embedding_model=" in message
    assert "embedding_architecture" in message


def test_unregistered_module_has_no_recipe() -> None:
    assert registry.embedding_recipe(torch.nn.Linear(2, 2)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name 'registry' from 'equine'` (collection error).

- [ ] **Step 3: Implement the registry**

```python
# src/equine/registry.py
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Registry of embedding-model architectures.

An EQUINE model file describes its embedding model as a *recipe*: the name of
a registered architecture plus the plain-valued constructor arguments it was
built with. ``load()`` rebuilds the module by calling only a registered
constructor, so a model file never names an import path and never carries
executable content unless the saver explicitly opts in.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, Optional

import torch

_REGISTRY: dict[str, type[torch.nn.Module]] = {}
_RECIPE_ATTR = "_equine_recipe"
_PLAIN_SCALARS = (int, float, str, bool, type(None))


def _check_plain(value: Any, path: str) -> None:
    """Accept only values that survive ``torch.load(weights_only=True)`` as-is."""
    if isinstance(value, _PLAIN_SCALARS):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_plain(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(
                    f"Recipe argument {path!r} has a non-plain dict key {key!r}."
                )
            _check_plain(item, f"{path}[{key!r}]")
        return
    raise TypeError(
        f"Constructor argument {path!r} of a registered embedding architecture must "
        f"be a plain value (int, float, str, bool, None, or lists/dicts of those) so "
        f"it can be stored in a model file; got {type(value).__name__}."
    )


def register_embedding_architecture(name: str, cls: type[torch.nn.Module]) -> None:
    """
    Register ``cls`` under ``name`` and make its instances carry a recipe.

    The class's ``__init__`` is wrapped so that every instance records the
    arguments it was constructed with (defaults applied). Those arguments must
    be plain values. Registering the same class under the same name again is a
    no-op; a different class under a taken name raises ``ValueError``.

    Parameters
    ----------
    name : str
        Stable identifier stored in model files. Namespace it by project
        (``"myproject.encoder"``). Never an import path.
    cls : type[torch.nn.Module]
        The architecture class.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Embedding architecture name {name!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}; cannot register "
            f"{cls.__module__}.{cls.__qualname__} under it."
        )
    if getattr(cls, "_equine_registered_as", None) == name:
        return

    original_init = cls.__init__
    signature = inspect.signature(original_init)
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"{cls.__qualname__}.__init__ accepts *args; registered architectures "
                "need named constructor arguments so they can be recorded in a recipe."
            )

    @functools.wraps(original_init)
    def recording_init(self: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        recipe_kwargs: dict[str, Any] = {}
        for key, value in bound.arguments.items():
            if key == "self":
                continue
            if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                recipe_kwargs.update(value)
            else:
                recipe_kwargs[key] = value
        for key, value in recipe_kwargs.items():
            _check_plain(value, key)
        original_init(self, *args, **kwargs)
        setattr(self, _RECIPE_ATTR, {"builder": name, "kwargs": recipe_kwargs})

    cls.__init__ = recording_init  # type: ignore[method-assign]
    cls._equine_registered_as = name  # type: ignore[attr-defined]
    _REGISTRY[name] = cls


def embedding_architecture(
    name: str,
) -> Callable[[type[torch.nn.Module]], type[torch.nn.Module]]:
    """
    Class-decorator form of :func:`register_embedding_architecture`.

    Example
    -------
    >>> @equine.embedding_architecture("myproject.encoder")
    ... class Encoder(torch.nn.Module):
    ...     def __init__(self, in_features: int, out_features: int) -> None: ...
    """

    def decorator(cls: type[torch.nn.Module]) -> type[torch.nn.Module]:
        register_embedding_architecture(name, cls)
        return cls

    return decorator


def registered_architectures() -> list[str]:
    """Names currently registered in this process, sorted."""
    return sorted(_REGISTRY)


def embedding_recipe(module: torch.nn.Module) -> Optional[dict[str, Any]]:
    """The recipe recorded on ``module``, or ``None`` if it is not a registered architecture."""
    recipe = getattr(module, _RECIPE_ATTR, None)
    return dict(recipe) if recipe is not None else None


def build_from_recipe(recipe: dict[str, Any]) -> torch.nn.Module:
    """
    Instantiate a registered architecture from a recipe read out of a model file.

    Raises
    ------
    ValueError
        If the recipe names an architecture not registered in this process.
    """
    name = recipe.get("builder")
    cls = _REGISTRY.get(name)
    if cls is None:
        known = ", ".join(registered_architectures()) or "(none)"
        raise ValueError(
            f"This model file was saved with embedding architecture {name!r}, which is "
            f"not registered in this process. Registered: {known}. Import the module that "
            f"registers it (@equine.embedding_architecture({name!r})), or pass the "
            f"architecture yourself with load(..., embedding_model=YourModule(...))."
        )
    return cls(**recipe.get("kwargs", {}))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q -p no:cacheprovider`
Expected: `8 passed`.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check --fix src/equine/registry.py tests/test_registry.py && uvx ruff format src/equine/registry.py tests/test_registry.py
git add src/equine/registry.py tests/test_registry.py
git commit -m "feat(registry): embedding architectures registered by name with recorded recipes"
```

---

### Task 2: Shipped `MLP` builder and public exports

**Files:**
- Create: `src/equine/architectures.py`
- Modify: `src/equine/__init__.py`
- Test: `tests/test_registry.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
import equine as eq


def test_shipped_mlp_is_registered_and_records_recipe() -> None:
    mlp = eq.MLP(6, [16, 8], 3)
    assert eq.embedding_recipe(mlp) == {
        "builder": "equine.mlp",
        "kwargs": {
            "in_features": 6,
            "hidden_sizes": [16, 8],
            "out_features": 3,
            "activation": "relu",
        },
    }
    assert "equine.mlp" in eq.registered_architectures()
    assert mlp(torch.rand(4, 6)).shape == (4, 3)


def test_shipped_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        eq.MLP(6, [16], 3, activation="swish")


def test_public_api_exports() -> None:
    for name in (
        "MLP",
        "embedding_architecture",
        "register_embedding_architecture",
        "registered_architectures",
        "embedding_recipe",
    ):
        assert name in eq.__all__, name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q -p no:cacheprovider -k "mlp or exports"`
Expected: `AttributeError: module 'equine' has no attribute 'MLP'` (3 failures).

- [ ] **Step 3: Implement the builder and the exports**

```python
# src/equine/architectures.py
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Embedding architectures that ship with EQUINE.

They are registered on import, so a model file that names one loads anywhere
EQUINE is installed without the user defining a class. Their constructor
signatures are part of the model-file format: change them only additively,
with defaults.
"""

from __future__ import annotations

import torch

from .registry import embedding_architecture

_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


@embedding_architecture("equine.mlp")
class MLP(torch.nn.Module):
    """
    A plain multilayer perceptron for tabular features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_sizes : list[int]
        Width of each hidden layer, in order. An empty list gives a linear map.
    out_features : int
        Embedding dimension (``emb_out_dim`` for the EQUINE model).
    activation : str
        One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"sigmoid"``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int],
        out_features: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}."
            )
        layers: list[torch.nn.Module] = []
        width = in_features
        for hidden in hidden_sizes:
            layers += [torch.nn.Linear(width, hidden), _ACTIVATIONS[activation]()]
            width = hidden
        layers.append(torch.nn.Linear(width, out_features))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

In `src/equine/__init__.py`, replace the import block and `__all__` so they read:

```python
from typing import TYPE_CHECKING

from . import architectures as _architectures  # noqa: F401  (registers "equine.*")
from .architectures import MLP
from .equine import Equine, EquineOutput
from .equine_gp import EquineGP
from .equine_protonet import CovType, EquineProtonet
from .load_equine_model import load_equine_model
from .registry import (
    embedding_architecture,
    embedding_recipe,
    register_embedding_architecture,
    registered_architectures,
)
from .utils import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    generate_episode,
    generate_model_metrics,
    generate_model_summary,
    generate_support,
    generate_train_summary,
    mahalanobis_distance_nosq,
)
```

and add to `__all__`, after `"CovType"`:

```python
    "MLP",
    "embedding_architecture",
    "embedding_recipe",
    "register_embedding_architecture",
    "registered_architectures",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q -p no:cacheprovider`
Expected: `11 passed`.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check --fix src tests && uvx ruff format src tests
git add src/equine/architectures.py src/equine/__init__.py tests/test_registry.py
git commit -m "feat: ship equine.mlp builder and export the registry API"
```

---

### Task 3: Register the test embedding model (the one-decorator change)

**Files:**
- Modify: `tests/conftest.py:15`

- [ ] **Step 1: Add the decorator**

Change the class header in `tests/conftest.py` from:

```python
class BasicEmbeddingModel(torch.nn.Module):
```

to:

```python
@eq.embedding_architecture("equine.tests.basic")
class BasicEmbeddingModel(torch.nn.Module):
```

(`eq` is already imported at the top of `conftest.py`.)

Also move the autouse registry-isolation fixture from `tests/test_registry.py` into `tests/conftest.py` so every test module gets it (Task 4's `test_persistence.py` registers a class inside a parametrized test and would otherwise trigger the redefinition warning on the second parameter):

```python
@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    from equine import registry

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
```

Remove the copy from `tests/test_registry.py` (add `import pytest` to conftest if missing).

- [ ] **Step 2: Run the whole suite to confirm nothing changes yet**

Run: `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 4`
Expected: all tests pass (the save path still writes TorchScript until Task 4). `BasicEmbeddingModel(tensor_dim, num_classes)` takes two ints, so its recipe records `{"tensor_dim": ..., "num_classes": ...}`.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: register the test embedding model; share the registry-isolation fixture"
```

---

### Task 4: `EquineProtonet` — recipe save/load with the transition bridge

> **Boundary correction (2026-09-28, found during implementation):** Task 4's tests call `load_equine_model(..., trust_executable=..., embedding_model=...)`, and without forwarding the trust flag the generic loader would regress on legacy files between Tasks 4 and 6. So Task 4 also makes `load_equine_model` accept `trust_executable` and `embedding_model` and forward them (with `trust_executable or allow_unsafe_legacy_format`) to `EquineProtonet._from_checkpoint`, leaving the GP branch untouched. The two GP-dependent cases in `tests/test_safe_loading.py` (`test_gp_save_format_is_weights_only_safe` and the gp parameter of the legacy re-save test) carry `xfail(strict=True, reason="EquineGP recipe persistence lands in Task 5")` until Task 5 removes them. The mismatch test spies on `equine.utils.build_from_recipe` (asserting the first build happens on the meta device) rather than on `torch.empty`, because torch caches its tensor-creation functions on first device-context entry and a patched `torch.empty` makes the meta device stop applying.

**Files:**
- Modify: `src/equine/equine_protonet.py` (imports; `save`; `load`; `_from_checkpoint`; add `_to_checkpoint`, `_rebuild_embedding`)
- Create: `tests/test_persistence.py`
- Modify: `tests/test_safe_loading.py` (layout helper; legacy re-save test)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_persistence.py
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Embedding models persist as a recipe + state_dict; TorchScript only behind
allow_executable / trust_executable for one transition release (D-191).
"""

import subprocess
import sys

import pytest
import torch
from conftest import BasicEmbeddingModel

import equine as eq


class UnregisteredNet(torch.nn.Module):
    """An embedding with no recipe (also what torch.jit.script yields)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(6, 3)

    def forward(self, x):
        return self.lin(x)


def _dataset(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    return torch.utils.data.TensorDataset(X, Y), X


def train_protonet(embedding, **kwargs):
    dataset, X = _dataset()
    model = eq.EquineProtonet(embedding, 3, **kwargs)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def train_gp(embedding, **kwargs):
    dataset, X = _dataset()
    model = eq.EquineGP(embedding, 3, 3, num_random_features=16, **kwargs)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
    )
    return model, X


TRAINERS = [
    pytest.param(train_protonet, eq.EquineProtonet, id="protonet"),
    pytest.param(train_gp, eq.EquineGP, id="gp"),
]


def assert_same(a, b, X):
    oa, ob = a.predict(X[:8]), b.predict(X[:8])
    assert torch.allclose(oa.classes, ob.classes, atol=1e-6)
    assert torch.allclose(oa.ood_scores, ob.ood_scores, atol=1e-6)


# --- data-only files -----------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_registered_embedding_saves_data_only_and_round_trips(tmp_path, train, cls) -> None:
    model, X = train(BasicEmbeddingModel(6, 3))
    model.label_names = ["a", "b", "c"]
    path = str(tmp_path / "m.eq")
    model.save(path)

    ckpt = torch.load(path, weights_only=True)  # must not raise
    assert ckpt["equine_format_version"] == 2
    assert ckpt["contains_executable"] is False
    assert "embed_jit_save" not in ckpt
    assert ckpt["embedding_recipe"] == {
        "builder": "equine.tests.basic",
        "kwargs": {"tensor_dim": 6, "num_classes": 3},
    }
    assert set(ckpt["embedding_state_dict"]) == set(model.embedding_model.state_dict())

    for reloaded in (cls.load(path), eq.load_equine_model(path)):
        assert isinstance(reloaded, cls)
        assert isinstance(reloaded.embedding_model, BasicEmbeddingModel)
        assert reloaded.label_names == ["a", "b", "c"]
        assert all(isinstance(k, int) for k in reloaded.get_support().keys())
        assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_caller_supplied_architecture_overrides_recipe(tmp_path, train, cls) -> None:
    model, X = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    fresh = eq.MLP(6, [16], 3)
    assert_same(model, cls.load(path, embedding_model=fresh), X)
    assert_same(model, eq.load_equine_model(path, embedding_model=eq.MLP(6, [16], 3)), X)


# --- transition bridge -------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_unregistered_embedding_is_refused_without_opt_in(tmp_path, train, cls) -> None:
    model, _ = train(UnregisteredNet())
    with pytest.raises(ValueError) as err:
        model.save(str(tmp_path / "m.eq"))
    message = str(err.value)
    assert "UnregisteredNet" in message
    assert "embedding_architecture" in message
    assert "allow_executable=True" in message
    assert "trust_executable=True" in message


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_executable_opt_in_is_flagged_and_needs_trust(tmp_path, train, cls) -> None:
    model, X = train(torch.jit.script(UnregisteredNet()))
    path = str(tmp_path / "m.eq")
    model.save(path, allow_executable=True)

    ckpt = torch.load(path, weights_only=True)  # still safe to *read*
    assert ckpt["contains_executable"] is True
    assert ckpt["embed_jit_save"].dtype == torch.uint8

    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="trust_executable=True"):
            loader(path)
        assert_same(model, loader(path, trust_executable=True), X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_archive_without_flag_still_needs_trust(tmp_path, train, cls) -> None:
    """A hand-edited file cannot bypass the check by dropping the flag."""
    model, _ = train(torch.jit.script(UnregisteredNet()))
    path = str(tmp_path / "m.eq")
    model.save(path, allow_executable=True)
    ckpt = torch.load(path, weights_only=True)
    del ckpt["contains_executable"]
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="trust_executable=True"):
        cls.load(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_migration_from_executable_file_to_recipe(tmp_path, train, cls) -> None:
    model, X = train(torch.jit.script(BasicEmbeddingModel(6, 3)))
    flagged = str(tmp_path / "flagged.eq")
    model.save(flagged, allow_executable=True)

    migrated = cls.load(flagged, trust_executable=True, embedding_model=BasicEmbeddingModel(6, 3))
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)  # no flag needed any more

    ckpt = torch.load(data_only, weights_only=True)
    assert ckpt["contains_executable"] is False
    assert ckpt["embedding_recipe"]["builder"] == "equine.tests.basic"
    assert_same(model, eq.load_equine_model(data_only), X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_migration_without_trust_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(torch.jit.script(BasicEmbeddingModel(6, 3)))
    flagged = str(tmp_path / "flagged.eq")
    model.save(flagged, allow_executable=True)
    with pytest.raises(ValueError, match="trust_executable=True"):
        cls.load(flagged, embedding_model=BasicEmbeddingModel(6, 3))


# --- recipe / weights consistency ----------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_recipe_that_disagrees_with_weights_is_refused_before_allocation(
    tmp_path, train, cls, monkeypatch
) -> None:
    """A tampered recipe describing a huge network must fail on the meta device,
    before any real tensor is allocated."""
    model, _ = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["embedding_recipe"]["kwargs"]["hidden_sizes"] = [10**6]  # ~6 GB if built
    torch.save(ckpt, path)

    allocated: list[int] = []
    real_empty = torch.empty

    def spy(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        if out.device.type != "meta":
            allocated.append(out.numel())
        return out

    monkeypatch.setattr(torch, "empty", spy)
    with pytest.raises(ValueError, match="does not match the weights"):
        cls.load(path)
    assert not any(n >= 10**6 for n in allocated), "a large tensor was allocated"


# --- unknown recipe -----------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_unknown_recipe_is_actionable_in_a_fresh_interpreter(tmp_path, train, cls) -> None:
    @eq.embedding_architecture("equine.tests.only_here")
    class OnlyHere(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)

        def forward(self, x):
            return self.lin(x)

    model, _ = train(OnlyHere())
    path = str(tmp_path / "m.eq")
    model.save(path)
    code = (
        "import sys, equine as eq\n"
        "try:\n"
        f"    eq.load_equine_model({path!r})\n"
        "except ValueError as e:\n"
        "    print(e); sys.exit(3)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert "'equine.tests.only_here'" in proc.stdout
    assert "not registered" in proc.stdout
    assert "embedding_model=" in proc.stdout
```

Note for the GP tests in this task: they fail until Task 5 adds GP parity; run only the Protonet cases in this task with `-k protonet`.

Then change `tests/test_safe_loading.py`:

Replace the body of `_assert_weights_only_safe_layout` with:

```python
def _assert_weights_only_safe_layout(path: str) -> dict:
    checkpoint = torch.load(path, weights_only=True)  # must not raise
    assert checkpoint["equine_format_version"] == 2
    if checkpoint["contains_executable"]:
        # transition layout: TorchScript archive as a uint8 tensor (raw bytes are
        # only accepted by the weights-only unpickler from torch 2.5 on)
        archive = checkpoint["embed_jit_save"]
        assert isinstance(archive, torch.Tensor) and archive.dtype == torch.uint8
    else:
        assert isinstance(checkpoint["embedding_recipe"]["builder"], str)
        assert "embedding_state_dict" in checkpoint
        assert "embed_jit_save" not in checkpoint
    return checkpoint
```

and replace the tail of `test_legacy_file_loads_with_opt_in_and_resaves_safely`, from `safe_path = str(tmp_path / "resaved.eq")` to the end, with:

```python
    # A legacy file carries a TorchScript embedding. Re-saving keeps it
    # executable (flagged, trust required); migrating through a registered
    # class yields a data-only file.
    flagged = str(tmp_path / "resaved.eq")
    reloaded.save(flagged, allow_executable=True)
    assert _assert_weights_only_safe_layout(flagged)["contains_executable"] is True
    _assert_same_predictions(model, eq.load_equine_model(flagged, trust_executable=True), X)

    migrated = eq.load_equine_model(
        legacy_path,
        allow_unsafe_legacy_format=True,
        embedding_model=BasicEmbeddingModel(6, 3),
    )
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)
    assert _assert_weights_only_safe_layout(data_only)["contains_executable"] is False
    _assert_same_predictions(model, eq.load_equine_model(data_only), X)
```

(`BasicEmbeddingModel` is already imported in that file.)

- [ ] **Step 2: Run the Protonet tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_persistence.py -q -p no:cacheprovider -k protonet`
Expected: failures such as `TypeError: save() got an unexpected keyword argument 'allow_executable'` and `KeyError: 'contains_executable'`.

- [ ] **Step 3: Implement in `src/equine/equine_protonet.py`**

Add to the imports, after `from .equine import Equine, EquineOutput`:

```python
from .registry import build_from_recipe, embedding_recipe
```

Replace the whole `save` method with:

```python
    def save(self, path: str, allow_executable: bool = False) -> None:
        """
        Save all model parameters to a file.

        The embedding model is stored as a *recipe* (registered architecture
        name plus constructor arguments) and a ``state_dict``, so the file holds
        only data. If the embedding model is not a registered architecture (for
        example a ``torch.jit.ScriptModule``), pass ``allow_executable=True`` to
        embed a TorchScript copy instead; such a file is flagged and can only be
        opened with ``load(..., trust_executable=True)``. This opt-in exists for
        one release to migrate existing files and will then be removed.

        Parameters
        ----------
        path : str
            Filename to write the model.
        allow_executable : bool, optional
            Permit embedding executable TorchScript when no recipe is available.

        Raises
        ------
        ValueError
            If the embedding model has no recipe and ``allow_executable`` is False.
        """
        torch.save(self._to_checkpoint(allow_executable), path)

    def _to_checkpoint(self, allow_executable: bool = False) -> dict[str, Any]:
        """Build the checkpoint dictionary that ``save`` writes and ``_from_checkpoint`` reads."""
        model_settings = {
            "cov_type": self.cov_type.value,
            "emb_out_dim": self.emb_out_dim,
            "use_temperature": self.use_temperature,
            "init_temperature": self.temperature.item(),
            "relative_mahal": self.relative_mahal,
            "device": self.device,
        }
        # Everything stored here must be readable by torch.load(weights_only=True)
        # on every supported torch version: tensors, containers and plain
        # scalars/strings only (issue #168).
        save_data: dict[str, Any] = {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            **_embedding_checkpoint(self.model.embedding_model, allow_executable),
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "model_head_save": self.model.model_head.state_dict(),
            "outlier_kde": {
                int(label): _kde_to_state(kde)
                for label, kde in self.outlier_score_kde.items()
            },
            "settings": model_settings,
            "support": {int(label): x for label, x in self.model.support.items()},
            "train_summary": self.train_summary,
        }
        return save_data
```

Replace the `load` classmethod and `_from_checkpoint` with:

```python
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        allow_unsafe_legacy_format: bool = False,
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
    ) -> Equine:
        """
        Load a previously saved EquineProtonet model.

        The file is read with ``torch.load(weights_only=True)`` and the
        embedding model is rebuilt from its recipe through the architecture
        registry, so a default file contains nothing executable.

        Parameters
        ----------
        path : str
            The filename of the saved model.
        device : Optional[str]
            The device to load the model onto.
        allow_unsafe_legacy_format : bool, optional
            Permit loading a file written in the legacy pickle format. This
            uses unrestricted unpickling and can execute code embedded in the
            file, so only enable it for files you trust. Implies
            ``trust_executable``. Defaults to False.
        trust_executable : bool, optional
            Permit running the TorchScript module embedded in a file saved with
            ``allow_executable=True``. Defaults to False.
        embedding_model : Optional[torch.nn.Module]
            Use this module as the embedding architecture instead of rebuilding
            it from the file's recipe; the file's weights are loaded into it.
            With a legacy or executable file (and trust), the archive's weights
            are copied into it, which migrates the file to the recipe format on
            the next ``save``.

        Returns
        -------
        EquineProtonet
            The reconstituted EquineProtonet object.

        Raises
        ------
        ValueError
            If the file cannot be loaded safely, names an unregistered
            architecture, or contains executable content without trust.
        """
        model_save = load_checkpoint(
            path,
            map_location=device,
            allow_unsafe_legacy_format=allow_unsafe_legacy_format,
            _stacklevel=4,  # skip the beartype wrapper around this classmethod
        )
        return cls._from_checkpoint(
            model_save,
            device,
            trust_executable=trust_executable or allow_unsafe_legacy_format,
            embedding_model=embedding_model,
        )

    @classmethod
    def _from_checkpoint(
        cls,
        model_save: dict[str, Any],
        device: Optional[str] = None,
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
    ) -> Equine:
        """Rebuild an EquineProtonet from an already-loaded checkpoint dictionary."""
        support = OrderedDict(
            (int(label), x) for label, x in model_save.get("support").items()
        )
        embedding = _rebuild_embedding(model_save, device, trust_executable, embedding_model)

        settings = dict(model_save.get("settings"))
        if isinstance(settings.get("cov_type"), str):
            settings["cov_type"] = CovType(settings["cov_type"])
        if device is not None:
            settings["device"] = device

        eq_model = cls(embedding, **settings)
        eq_model.model.model_head.load_state_dict(model_save.get("model_head_save"))
        eq_model.eval()
        eq_model.model.update_support(support)
        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.outlier_score_kde = OrderedDict(
            (int(label), _kde_from_state(kde) if isinstance(kde, dict) else kde)
            for label, kde in model_save.get("outlier_kde").items()
        )
        eq_model.train_summary = model_save.get("train_summary")
        return eq_model
```

The two helpers `_embedding_checkpoint` and `_rebuild_embedding` are shared by both model classes, so they live in `src/equine/utils.py`. Add them at the end of `utils.py` (after `load_jit_archive`), and import them in `equine_protonet.py` from `.utils` alongside the existing names:

```python
def _embedding_checkpoint(
    embedding: torch.nn.Module, allow_executable: bool
) -> dict[str, Any]:
    """
    The checkpoint entries describing an embedding model.

    A registered architecture is stored as its recipe and ``state_dict``. An
    unregistered one is refused unless ``allow_executable`` is set, in which
    case a flagged TorchScript archive is stored instead (transition only).
    """
    from .registry import embedding_recipe  # local import: registry has no deps on utils

    recipe = embedding_recipe(embedding)
    if recipe is not None:
        return {
            "embedding_recipe": recipe,
            "embedding_state_dict": embedding.state_dict(),
            "contains_executable": False,
        }
    if allow_executable:
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(prepare_jit_module(embedding)), buffer)
        return {
            "embed_jit_save": jit_archive_to_tensor(buffer),
            "contains_executable": True,
        }
    raise ValueError(
        f"Cannot save: the embedding model ({type(embedding).__name__}) is not a "
        "registered architecture, so it has no recipe to store. Register it with "
        "@equine.embedding_architecture('yourproject.name') and construct it with "
        "plain-valued arguments, or pass save(path, allow_executable=True) to embed a "
        "TorchScript copy (executable content; the file will then require "
        "load(..., trust_executable=True))."
    )


def _rebuild_embedding(
    model_save: dict[str, Any],
    device: Optional[str],
    trust_executable: bool,
    embedding_model: Optional[torch.nn.Module],
) -> torch.nn.Module:
    """
    Reconstitute an embedding model from a checkpoint.

    Order: a caller-supplied module (weights loaded into it), then the recipe
    (rebuilt through the registry), then a TorchScript archive behind
    ``trust_executable``. Any archive requires trust, whether or not the
    ``contains_executable`` flag is present, so a hand-edited file cannot
    bypass the check.
    """
    from .registry import build_from_recipe

    state_dict = model_save.get("embedding_state_dict")
    archive = model_save.get("embed_jit_save")
    executable_error = ValueError(
        "This model file embeds an executable TorchScript module (saved with "
        "allow_executable=True or by an older EQUINE version). Opening it runs that "
        "code. If you trust the file, load it with trust_executable=True; to migrate it "
        "to the data-only format, also pass embedding_model=YourRegisteredClass(...) and "
        "call save() on the result."
    )
    if embedding_model is not None:
        if state_dict is None and archive is not None:
            if not trust_executable:
                raise executable_error
            state_dict = load_jit_archive(archive, device).state_dict()
        if state_dict is not None:
            embedding_model.load_state_dict(state_dict)
        return embedding_model
    if "embedding_recipe" in model_save:
        recipe = model_save["embedding_recipe"]
        # Memory guard: a recipe from an untrusted file could describe an
        # enormous network. Build it once on the meta device (no allocation),
        # check that its parameters match the file's state_dict in name and
        # shape, and only then build for real. The file's own size therefore
        # bounds what loading can allocate. (Meta-device construction needs
        # torch >= 2.0; PR-0 raised the floor to 2.6.)
        with torch.device("meta"):
            skeleton = build_from_recipe(recipe)
        expected = {k: tuple(v.shape) for k, v in skeleton.state_dict().items()}
        actual = {k: tuple(v.shape) for k, v in (state_dict or {}).items()}
        if expected != actual:
            raise ValueError(
                f"The embedding recipe {recipe.get('builder')!r} does not match the "
                "weights stored in the file (parameter names or shapes differ); the "
                "file is inconsistent or was tampered with."
            )
        embedding = build_from_recipe(recipe)
        embedding.load_state_dict(state_dict)
        return embedding
    if archive is not None:
        if not trust_executable:
            raise executable_error
        return load_jit_archive(archive, device)
    raise ValueError("Model file contains no embedding model (no recipe and no archive).")
```

In `equine_protonet.py`, import them: extend the `from .utils import (...)` block with `_embedding_checkpoint,` and `_rebuild_embedding,` (keep the block alphabetized as ruff's isort wants: underscore names sort first). Remove the now-unused imports `io`, `jit_archive_to_tensor`, `load_jit_archive`, `prepare_jit_module` from `equine_protonet.py` if ruff reports them unused; `embedding_recipe`/`build_from_recipe` are no longer needed in the Protonet module either (they are used from `utils`), so drop that import line if unused.

- [ ] **Step 4: Run the Protonet tests and the safe-loading tests**

Run: `.venv/bin/python -m pytest tests/test_persistence.py tests/test_safe_loading.py -q -p no:cacheprovider -k "protonet or not gp"`
Expected: all Protonet cases pass; `test_safe_loading.py` passes. GP cases in `test_persistence.py` are deselected.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check --fix src tests && uvx ruff format src tests
git add src/equine/equine_protonet.py src/equine/utils.py tests/test_persistence.py tests/test_safe_loading.py
git commit -m "feat(protonet): save embedding as recipe + state_dict; TorchScript only behind allow_executable/trust_executable"
```

---

### Task 5: `EquineGP` parity

> Also in this task: clone support tensors at save time in `EquineGP._to_checkpoint` (`{int(label): x.detach().clone() ...}`, as Task 4 did for Protonet, because `generate_support` returns views into the training data); check GP's device assumptions now that `load_checkpoint` loads on CPU and moves to `map_location` only when given; add a test that a flagged Protonet file relabelled as `modelType: "EquineGP"` is refused without trust through `load_equine_model` (at Task 4's commit the GP branch still runs TorchScript untrusted); remove the two `xfail` marks added in Task 4 to `tests/test_safe_loading.py`; extend `load_equine_model`'s GP branch to forward `trust_executable` and `embedding_model` to `EquineGP._from_checkpoint` (the GP half of what was Task 6); and give the abstract `Equine.save` in `src/equine/equine.py` the `allow_executable: bool = False` parameter so the `# type: ignore[call-arg]` in `tests/test_safe_loading.py` can be removed.

**Files:**
- Modify: `src/equine/equine_gp.py` (imports; `save`; `load`; `_from_checkpoint`; add `_to_checkpoint`)
- Test: `tests/test_persistence.py` (GP cases already written in Task 4)

- [ ] **Step 1: Run the GP tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_persistence.py -q -p no:cacheprovider -k gp`
Expected: failures such as `TypeError: save() got an unexpected keyword argument 'allow_executable'` and `TypeError: load() got an unexpected keyword argument 'trust_executable'`.

- [ ] **Step 2: Implement in `src/equine/equine_gp.py`**

Extend the `from .utils import (...)` block with `_embedding_checkpoint,` and `_rebuild_embedding,`. Replace the whole `save` method with:

```python
    def save(self, path: str, allow_executable: bool = False) -> None:
        """
        Save all model parameters to a file.

        The embedding model (the feature extractor) is stored as a *recipe*
        (registered architecture name plus constructor arguments) and a
        ``state_dict``, so the file holds only data. If it is not a registered
        architecture, pass ``allow_executable=True`` to embed a TorchScript copy
        instead; such a file is flagged and can only be opened with
        ``load(..., trust_executable=True)``. This opt-in exists for one release
        to migrate existing files and will then be removed.

        Parameters
        ----------
        path : str
            Filename to write the model.
        allow_executable : bool, optional
            Permit embedding executable TorchScript when no recipe is available.

        Raises
        ------
        ValueError
            If the embedding model has no recipe and ``allow_executable`` is False.
        """
        torch.save(self._to_checkpoint(allow_executable), path)

    def _to_checkpoint(self, allow_executable: bool = False) -> dict[str, Any]:
        """Build the checkpoint dictionary that ``save`` writes and ``_from_checkpoint`` reads."""
        model_settings = {
            "emb_out_dim": self.num_deep_features,
            "num_classes": self.num_outputs,
            "num_random_features": self.num_random_features,
            "init_temperature": self.temperature.item(),
            "device": self.device_type,
        }
        laplace_sd = {
            key: value
            for key, value in self.model.state_dict().items()
            if "feature_extractor" not in key
        }
        # Everything stored here must be readable by torch.load(weights_only=True)
        # on every supported torch version: tensors, containers and plain
        # scalars/strings only (issue #168).
        save_data: dict[str, Any] = {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            **_embedding_checkpoint(self.embedding_model, allow_executable),
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "laplace_model_save": laplace_sd,
            "num_data": self.model.num_data,
            "settings": model_settings,
            "support": {int(label): x for label, x in self.support.items()},
            "train_batch_size": self.model.train_batch_size,
            "train_summary": self.train_summary,
        }
        return save_data
```

Replace `load` and `_from_checkpoint` with:

```python
    @classmethod
    def load(
        cls,
        path: str,
        allow_unsafe_legacy_format: bool = False,
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
    ) -> Equine:
        """
        Load a previously saved EquineGP model.

        The file is read with ``torch.load(weights_only=True)`` and the
        embedding model is rebuilt from its recipe through the architecture
        registry, so a default file contains nothing executable.

        Parameters
        ----------
        path : str
            Input filename.
        allow_unsafe_legacy_format : bool, optional
            Permit loading a file written in the legacy pickle format. This
            uses unrestricted unpickling and can execute code embedded in the
            file, so only enable it for files you trust. Implies
            ``trust_executable``. Defaults to False.
        trust_executable : bool, optional
            Permit running the TorchScript module embedded in a file saved with
            ``allow_executable=True``. Defaults to False.
        embedding_model : Optional[torch.nn.Module]
            Use this module as the embedding architecture instead of rebuilding
            it from the file's recipe; the file's weights are loaded into it.
            With a legacy or executable file (and trust), the archive's weights
            are copied into it, which migrates the file to the recipe format on
            the next ``save``.

        Returns
        -------
        EquineGP
            The reconstituted EquineGP object.

        Raises
        ------
        ValueError
            If the file cannot be loaded safely, names an unregistered
            architecture, or contains executable content without trust.
        """
        model_save = load_checkpoint(
            path,
            allow_unsafe_legacy_format=allow_unsafe_legacy_format,
            _stacklevel=4,  # skip the beartype wrapper around this classmethod
        )
        return cls._from_checkpoint(
            model_save,
            trust_executable=trust_executable or allow_unsafe_legacy_format,
            embedding_model=embedding_model,
        )

    @classmethod
    def _from_checkpoint(
        cls,
        model_save: dict[str, Any],
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
    ) -> Equine:
        """Rebuild an EquineGP from an already-loaded checkpoint dictionary."""
        # No device parameter yet: EquineGP.load gains it in PR-2b (Phase 2).
        embedding = _rebuild_embedding(model_save, None, trust_executable, embedding_model)
        settings = dict(model_save.get("settings"))
        eq_model = cls(embedding, **settings)

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.train_summary = model_save.get("train_summary")
        eq_model.model.load_state_dict(model_save.get("laplace_model_save"), strict=False)
        eq_model.model.seen_data = model_save.get("laplace_model_save").get("seen_data")
        eq_model.model.set_training_params(
            model_save.get("num_data"), model_save.get("train_batch_size")
        )
        eq_model.eval()

        support = OrderedDict(
            (int(label), x) for label, x in model_save.get("support").items()
        )
        if len(support) > 0:
            eq_model.support = support
            eq_model.prototypes = eq_model.compute_prototypes()
        return eq_model
```

Remove the imports `io`, `jit_archive_to_tensor`, `load_jit_archive`, `prepare_jit_module` from `equine_gp.py` if ruff reports them unused.

- [ ] **Step 3: Run the whole persistence test file**

Run: `.venv/bin/python -m pytest tests/test_persistence.py tests/test_safe_loading.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 4: Lint and commit**

```bash
uvx ruff check --fix src tests && uvx ruff format src tests
git add src/equine/equine_gp.py
git commit -m "feat(gp): recipe persistence with the same save/load flags as Protonet"
```

---

### Task 6: `load_equine_model` forwards every argument

> **Mostly absorbed:** the Protonet half moved into Task 4 and the GP half into Task 5. What remains here is the generic-loader test below and a docstring pass on `load_equine_model`; skip any step already done.

**Files:**
- Modify: `src/equine/load_equine_model.py`
- Test: `tests/test_persistence.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_persistence.py`:

```python
@pytest.mark.parametrize("train, cls", TRAINERS)
def test_generic_loader_forwards_trust_flags_and_override(tmp_path, train, cls) -> None:
    model, X = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    reloaded = eq.load_equine_model(path, embedding_model=BasicEmbeddingModel(6, 3))
    assert isinstance(reloaded, cls)
    assert_same(model, reloaded, X)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_persistence.py -q -p no:cacheprovider -k generic_loader`
Expected: `TypeError: load_equine_model() got an unexpected keyword argument 'embedding_model'`.

- [ ] **Step 3: Implement**

Replace the whole file `src/equine/load_equine_model.py` with:

```python
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import torch

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet
from .utils import load_checkpoint


def load_equine_model(
    model_path: str,
    allow_unsafe_legacy_format: bool = False,
    trust_executable: bool = False,
    embedding_model: Optional[torch.nn.Module] = None,
) -> Equine:
    """
    Load an EQUINE model from a file, whichever model class saved it.

    Parameters
    ----------
    model_path : str
        The path to the model file.
    allow_unsafe_legacy_format : bool, optional
        Permit loading a file written in the legacy pickle format (unrestricted
        unpickling; only for files you trust). Implies ``trust_executable``.
    trust_executable : bool, optional
        Permit running the TorchScript module embedded in a file saved with
        ``allow_executable=True``.
    embedding_model : Optional[torch.nn.Module]
        Use this module as the embedding architecture instead of rebuilding it
        from the file's recipe (the file's weights are loaded into it).

    Returns
    -------
    Equine
        The loaded EQUINE model.

    Raises
    ------
    ValueError
        If the model type is unknown, the file cannot be loaded safely, its
        recipe names an unregistered architecture, or it contains executable
        content without ``trust_executable``.
    """
    model_save = load_checkpoint(
        model_path, allow_unsafe_legacy_format=allow_unsafe_legacy_format
    )
    model_type = model_save["train_summary"]["modelType"]
    kwargs = dict(
        trust_executable=trust_executable or allow_unsafe_legacy_format,
        embedding_model=embedding_model,
    )
    if model_type == "EquineProtonet":
        return EquineProtonet._from_checkpoint(model_save, **kwargs)
    if model_type == "EquineGP":
        return EquineGP._from_checkpoint(model_save, **kwargs)
    raise ValueError(f"Unknown model type '{model_type}'")
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 4`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check --fix src tests && uvx ruff format src tests
git add src/equine/load_equine_model.py tests/test_persistence.py
git commit -m "feat: load_equine_model forwards trust_executable and embedding_model"
```

---

### Task 7: Changelog and docstring pass

**Files:**
- Create: `CHANGELOG.md`
- Modify: `src/equine/utils.py:EQUINE_FORMAT_VERSION` comment

- [ ] **Step 1: Write the changelog**

```markdown
# Changelog

## Unreleased

### Changed
- **Model files no longer embed executable code by default.** `save()` stores the
  embedding model as a recipe (the name of a registered architecture plus its
  constructor arguments) and a `state_dict`. Register your embedding class once:

  ```python
  @equine.embedding_architecture("myproject.encoder")
  class Encoder(torch.nn.Module):
      def __init__(self, in_features: int, out_features: int): ...
  ```

  Constructor arguments must be plain values (numbers, strings, booleans, lists
  and dicts of those). A shipped architecture, `equine.MLP`, needs no class.
- Model files are read with `torch.load(weights_only=True)` (#168). Files
  written by earlier releases need a one-time migration:

  ```python
  model = equine.load_equine_model("old.eq", allow_unsafe_legacy_format=True,
                                   embedding_model=Encoder(6, 3))
  model.save("new.eq")   # data-only from here on
  ```

### Deprecated (removed in the next release)
- `save(path, allow_executable=True)` embeds a TorchScript copy of an
  unregistered embedding model and flags the file; such files need
  `load(..., trust_executable=True)`. Both flags, the TorchScript archive
  support and the `torch < 2.10` ceiling are removed in the next release.
  Migrate flagged files with the snippet above (using `trust_executable=True`).

### Fixed
- Opening an untrusted `.eq` file could execute arbitrary code (#168).
```

- [ ] **Step 1b: Fix the stale Notes paragraph in `load_equine_model`'s docstring** (it still says the embedded TorchScript module is executable code; for recipe files there is none. Say: recipe files contain no executable content; files saved with `allow_executable=True` embed TorchScript and need `trust_executable=True`.)

- [ ] **Step 2: Update the format-version comment in `src/equine/utils.py`**

Replace the comment above `EQUINE_FORMAT_VERSION = 2` with:

```python
# Version of the on-disk layout written by ``EquineProtonet.save`` and
# ``EquineGP.save``. Version 2 contains only tensors, containers and plain
# Python scalars/strings, so it can be read with ``torch.load(weights_only=True)``
# on every supported torch version. The embedding model is a recipe plus a
# state_dict (see ``equine.registry``); a TorchScript archive appears only in
# files saved with ``allow_executable=True`` during the transition release.
# Files without this key are "legacy" files that pickled arbitrary Python
# objects and need unrestricted unpickling.
```

- [ ] **Step 3: Spell-check and commit**

Run: `uvx codespell src docs CHANGELOG.md`
Expected: no output.

```bash
git add CHANGELOG.md src/equine/utils.py
git commit -m "docs: changelog for recipe persistence and the transition bridge"
```

---

### Task 8: Verification and PR on the fork

**Files:** none new.

- [ ] **Step 1: Full verification**

```bash
uvx ruff check src tests && uvx ruff format --check src tests && uvx codespell src docs CHANGELOG.md
NUMBA_DISABLE_JIT=1 .venv/bin/python -m pytest --cov-report=term-missing:skip-covered --cov-config=pyproject.toml --cov-fail-under=97 --cov=equine tests -q -p no:cacheprovider -n 4
```
Expected: `All checks passed!`, no spelling output, `Required test coverage of 97% reached`, all tests pass.

- [ ] **Step 2: Confirm the stray-file check**

Run: `git status --porcelain`
Expected: empty (no `.eq` files left in the repo root by the tests).

- [ ] **Step 3: Push and open the stack on the fork**

PR-0b is the second layer of Stack 0. Open PR-0 first (bottom, targets `main`), then PR-0b targeting PR-0's branch; GitHub offers **Create stack** when a PR's base is another open PR's branch.

```bash
gh pr create -R martinez-hub/equine --draft --base main --head fix/safe-model-loading \
  --title "PR-0: load model files without unrestricted unpickling (#168)" --body-file /dev/stdin <<'EOF'
See commit message. Bottom layer of Stack 0; PR-0b (persistence as architecture-as-code) stacks on top.
EOF
git push -u fork feat/persistence-as-code
gh pr create -R martinez-hub/equine --draft --base fix/safe-model-loading --head feat/persistence-as-code \
  --title "PR-0b: persist embedding models as recipes (architecture-as-code)" \
  --body-file docs/superpowers/specs/2026-09-28-d191-persistence-design.md
```
Expected: two draft PR URLs on `martinez-hub/equine`, PR-0b based on PR-0's branch. Do not open anything on `mit-ll-responsible-ai/equine`. After PR-0b merges on the fork, delete `spike/d191-registry`.

In the PR body's top, add a "Suggested README paragraph" section (README itself is not edited):

> Model files store the embedding network as a recipe and weights, never as code. Decorate your embedding class with `@equine.embedding_architecture("yourproject.name")` so it can be rebuilt on load, or use the shipped `equine.MLP`.

- [ ] **Step 4: Run the adversarial review on the branch before marking ready**

Use `/adversarial-workflow feat/persistence-as-code` (base `fix/safe-model-loading`) and address confirmed findings in follow-up commits on the same branch.

---

## Self-review

**Spec coverage.** §1 scope: Tasks 1–7. §2 format: Task 4/5 `_to_checkpoint` keys match the table; `contains_executable` always written. §3 registry: Task 1 (`*args` refusal, `**kwargs` flattening, plain check, idempotent re-registration, duplicate error, unknown-recipe message); shipped MLP: Task 2. §4 save/load: Tasks 4–6, the same new flags on both classes; the `device` parameter on EquineGP.load and load_equine_model is deferred to PR-2b per the roadmap. §5 bridge and migration: `_embedding_checkpoint` / `_rebuild_embedding` in Task 4, tested in Task 4/5 for both classes and in `test_safe_loading.py` for legacy input; removal schedule stated in Task 7's changelog. §6 errors: messages asserted in Tasks 1 and 4. §7 tests: all listed cases present; cross-version fixture is deferred to PR-1c as the spec allows. §8 docs: Task 7; README untouched. §9 web app: not part of this repo. §10 spike: deleted after merge (Task 8 note).

**Placeholders.** None: every code step shows full code; commands have expected output.

**Type consistency.** `_embedding_checkpoint(embedding, allow_executable) -> dict`, `_rebuild_embedding(model_save, device, trust_executable, embedding_model) -> nn.Module`, `_to_checkpoint(allow_executable=False) -> dict`, `_from_checkpoint(model_save, device=None, trust_executable=False, embedding_model=None)`, and `load(path, [device=None,] allow_unsafe_legacy_format=False, trust_executable=False, embedding_model=None)` (device on Protonet only, pre-existing) are used with the same names and order in Tasks 4, 5 and 6; EquineGP's `_from_checkpoint` omits `device` and passes `None` to `_rebuild_embedding`.
