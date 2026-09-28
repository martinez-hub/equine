# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Registry of embedding-model architectures (D-191 spike).

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
    arguments it was constructed with. Those arguments must be plain values.

    Parameters
    ----------
    name : str
        Stable, human-readable identifier stored in model files. Namespacing by
        project is recommended (``"myproject.encoder"``).
    cls : type[torch.nn.Module]
        The architecture class.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Embedding architecture name {name!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}."
        )
    if getattr(cls, "_equine_registered_as", None) == name:
        return  # idempotent re-registration (e.g. module reloaded)

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
        _check_plain(recipe_kwargs, f"{name}(...)")
        original_init(self, *args, **kwargs)
        setattr(self, _RECIPE_ATTR, {"builder": name, "kwargs": recipe_kwargs})

    cls.__init__ = recording_init  # type: ignore[method-assign]
    cls._equine_registered_as = name  # type: ignore[attr-defined]
    _REGISTRY[name] = cls


def embedding_architecture(
    name: str,
) -> Callable[[type[torch.nn.Module]], type[torch.nn.Module]]:
    """
    Class decorator form of :func:`register_embedding_architecture`.

    Example
    -------
    >>> @eq.embedding_architecture("myproject.encoder")
    ... class Encoder(torch.nn.Module):
    ...     def __init__(self, in_features: int, out_features: int) -> None: ...
    """

    def decorator(cls: type[torch.nn.Module]) -> type[torch.nn.Module]:
        register_embedding_architecture(name, cls)
        return cls

    return decorator


def registered_architectures() -> list[str]:
    """Names currently registered, sorted."""
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
        If the recipe names an architecture that is not registered in this
        process. The message lists what is registered and how to proceed.
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
