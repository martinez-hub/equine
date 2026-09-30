# D-191: Embedding-Model Persistence as Architecture-as-Code

Design for PR-0b of the remediation roadmap. Resolves issues #191 and the remainder of #189, and is the base that the #168 fix (PR-0) is rebuilt on.

**Decision (2026-09-28, maintainer):** an EQUINE model file describes its embedding model as a *recipe*, the name of a registered architecture plus plain-valued constructor arguments, together with a `state_dict`. Loading rebuilds the module by calling only a registered constructor. TorchScript persistence survives for exactly one release, as a flagged opt-in used to migrate existing files, and is then removed from the library.

**Why:** TorchScript is formally deprecated (PyTorch 2.9/2.10; 2.14 warns that `torch.jit.save`/`load` may break on Python 3.14). Its successors do not give a durable "embed a compiled program" story: `torch.export` files are documented as unstable across versions and its loader still unpickles; ONNX freezes the model and adds a runtime. A recipe plus weights depends on no PyTorch serialization format at all, is the layout every library on the Hugging Face Hub uses, and lets the loader promise that a default file contains nothing executable.

## 1. Scope

In scope:
- `equine.registry`: the registry, the decorator, recipe recording, `build_from_recipe`.
- `equine.architectures`: one shipped builder, `equine.mlp`.
- `EquineProtonet` and `EquineGP`: `save`, `load`, `_to_checkpoint`, `_from_checkpoint`, with the same new flags on both classes (Protonet's existing `device` parameter is untouched; GP gets one in PR-2b).
- `load_equine_model`: dispatch and argument forwarding.
- Format version 2 (the same bump as PR-0; there is no separate bump).
- The transition branch (`allow_executable` / `trust_executable`) and the migration path.
- Tests, docstrings, a changelog entry, and the one-decorator change to `tests/conftest.py`.

Out of scope (own issues, later phases):
- Hugging Face Hub export (`save_pretrained`, `from_pretrained`, `push_to_hub`). The spike proved it fits; it is a Phase 4+ enhancement.
- The web app's architecture catalogue and the retirement of its `.jit` upload (web-app issue; see section 9).
- ~~Example notebooks beyond adding the decorator to their embedding class.~~ Revised 2026-09-28: every shipped notebook calls `save()` on an unregistered embedding and would raise, so the notebooks are updated in this PR to registered architectures (code cells only, not re-executed).
- Any change to `README.md`.

## 2. File format, version 2

Every value is readable with `torch.load(weights_only=True)` on every torch version `pyproject.toml` declares: tensors, `dict`/`OrderedDict`/`list`, `int`, `float`, `str`, `bool`, `None`. No `bytes` (rejected by the restricted unpickler before torch 2.5), no Enum instances, no scipy objects, no `io.BytesIO`.

| Key | Type | Notes |
|---|---|---|
| `equine_format_version` | `int` | `2` |
| `contains_executable` | `bool` | `False` for a recipe file; `True` for a transition file |
| `embedding_recipe` | `dict` | `{"builder": str, "kwargs": dict}`; present when `contains_executable` is `False` |
| `embedding_state_dict` | `dict[str, Tensor]` | present with `embedding_recipe` |
| `embed_jit_save` | `uint8 Tensor` | TorchScript archive; present only when `contains_executable` is `True` (transition release only) |
| `settings` | `dict` | constructor arguments of the EQUINE class; Protonet stores `cov_type` as its string value |
| `model_head_save` (Protonet) / `laplace_model_save` (GP) | state dict | head weights; GP's excludes the feature extractor's keys |
| `support` | `dict[int, Tensor]` | plain int keys |
| `outlier_kde` (Protonet) | `dict[int, {"dataset": Tensor, "bw_factor": float}]` | rebuilt as `scipy.stats.gaussian_kde(dataset, bw_method=bw_factor)` |
| `num_data`, `train_batch_size` (GP) | `int` | as today |
| `feature_names`, `label_names` | `list[str] \| None` | as today |
| `train_summary` | `dict` | as today; `modelType` is what `load_equine_model` dispatches on |

Files without `equine_format_version` are legacy (v1) files: pickled Python objects, opened only through `allow_unsafe_legacy_format=True`.

**Resource contract (reading and rebuilding the file).** Reading a recipe file and rebuilding its embedding allocates no more than a small constant factor over the bytes actually present in the file's records. Concretely: `load_checkpoint` accepts only PyTorch zip archives on the safe path (non-zip files need the legacy opt-in) and loads them memory-mapped, so every tensor's storage is sized from the record that backs it rather than from the pickle's claim; it then copies tensors out of the mapping, refuses any tensor whose storage is smaller than its shape (expanded or overlapping views, meta and sparse tensors), and refuses support and KDE tensors that alias one another (with `allow_unsafe_legacy_format=True`, which marks the file as trusted, aliased support tensors are copied apart instead, so files from earlier releases can be migrated); the embedding is first built on the meta device and compared name-by-name and shape-by-shape with the file's weights, and the file's distinct weight storages must hold at least one byte per module element, before it is built for real. Registered constructors must therefore be buildable on the meta device, allocate only through standard factory functions without an explicit `device=`, and keep every tensor they own in the `state_dict` (no non-persistent buffers sized from arguments). `save` performs the same meta build and comparison, so a module that cannot be rebuilt from its recipe is refused at save time with an accurate message rather than at load time. `load_checkpoint` also refuses compressed or oversized zip archives before `torch.load` runs (PyTorch writes its archives uncompressed, so genuine files pass), and requires the file's distinct weight storages to hold at least one byte per element of the module they describe, so distinct weights cannot all alias one small buffer.

Known limits of this contract, accepted for this release: the work done after loading (embedding the support set, fitting nothing) scales with support rows times hidden width and can exceed the file size for honest inputs; weight dtypes are not compared (a recipe rebuilds in the default dtype and the file's tensors are cast on load); a constructor that sizes a non-persistent buffer or plain tensor attribute from its arguments is not checked at load time (documented requirement on registered classes); errors raised by `torch.jit.load` on a malformed archive opened under trust are not normalised. The size of an embedding's `forward` output must be determined by its weights or bounded in its constructor; a width set by a constructor argument alone (e.g. `repeat`, or one-hot over `k`) is attacker-controlled when the recipe comes from a file, and is documented in the decorator's trust boundary rather than checked.

Later additions (label order, covariance choices, OOD method, from Phase 3) are added with defaults on read and bump the version to 3; this spec does not define them.

## 3. Registry and recipes (`equine.registry`)

Public API, all exported from `equine`:

```python
@equine.embedding_architecture("myproject.encoder")      # class decorator
class Encoder(torch.nn.Module): ...

equine.register_embedding_architecture(name: str, cls: type[nn.Module]) -> None   # function form
equine.registered_architectures() -> list[str]
equine.embedding_recipe(module: nn.Module) -> dict | None
```

Behaviour:
- Registration wraps `cls.__init__` so that every instance records `{"builder": name, "kwargs": <bound arguments>}` on itself. Arguments are bound with `inspect.signature` and defaults applied, so a recipe is complete even when the caller relied on defaults. `**kwargs` parameters are flattened into the recipe; `*args` parameters are refused at registration time because they cannot be named.
- Every recipe value must be plain, checked by exact type (subclasses such as `numpy.float64`, enums or namedtuples are refused): `int`, `float`, `str`, `bool`, `None`, or lists/tuples/dicts of those with `str`/`int` keys. A non-plain value raises `TypeError` at construction time naming the argument and the allowed types. Recorded arguments are deep-copied so later mutation cannot change the recipe. Only instances of the exact registered class record a recipe; an unregistered subclass has none and is refused at save time.
- Names are user-chosen strings, namespaced by convention (`"project.model"`); they are never import paths, so nothing in a file can name a module to import (the Keras `safe_mode` lesson).
- Registering the same class under the same name again is a no-op. Redefining a class with the same module and qualified name (re-running a notebook cell, `importlib.reload`) replaces the entry with a `UserWarning`. A genuinely different class under a taken name raises `ValueError`, as does registering one class under a second name.
- `build_from_recipe(recipe)` (internal) validates the recipe's shape (a dict with a `str` builder and a dict of kwargs), looks the name up and calls `cls(**kwargs)`. An unknown name raises `ValueError` listing the registered names, the decorator to add, and the `embedding_model=` override; a constructor `TypeError` (signature changed since the file was saved) is re-raised as `ValueError` naming the builder.
- Trust boundary: a model file can call any registered constructor with any plain arguments. Registered constructors must therefore be safe for arbitrary plain input: no paths that get opened or unpickled, no downloads, and awareness that large sizes allocate memory. The decorator's docstring says so, and it must sit above `@beartype` when both are used.
- The registry is process-global and populated by import: shipped builders on `import equine`, user builders when their module runs.

### Shipped builder (`equine.architectures`)

`equine.MLP(in_features: int, hidden_sizes: list[int], out_features: int, activation: str = "relu")`, registered as `"equine.mlp"`. Activations: `relu`, `gelu`, `tanh`, `sigmoid`. Its constructor signature is part of the file format: changes are additive with defaults, never renames or removals. No further builders in this PR.

## 4. Save and load

Both model classes expose the same surface:

```python
model.save(path: str, *, allow_executable: bool = False) -> None
Cls.load(path: str, [device: str | None = None,]   # Protonet today; EquineGP gains it in PR-2b
         *, allow_unsafe_legacy_format: bool = False,
         trust_executable: bool = False,
         embedding_model: nn.Module | None = None) -> Cls
equine.load_equine_model(path, *, allow_unsafe_legacy_format=False,
                         trust_executable=False, embedding_model=None)
```

Amended 2026-09-28: the flags are keyword-only, so a positional `True` cannot silently mean `allow_executable`, `allow_unsafe_legacy_format` or `trust_executable`.

- `save` builds the dictionary in section 2 via `_to_checkpoint(allow_executable)` and calls `torch.save`. If the embedding has a recipe, the file is data-only. If it has none (an unregistered class, or a `torch.jit.ScriptModule`), `save` raises `ValueError` unless `allow_executable=True` (section 5).
- `load` reads through `utils.load_checkpoint` (weights_only first; see PR-0) and hands the dictionary to `_from_checkpoint`, which rebuilds the embedding, constructs the EQUINE class with `settings`, loads head weights and support, and restores KDEs.
- Rebuilding the embedding, in order: if `embedding_model` was passed, use it and load the file's `embedding_state_dict` into it when present; else if the file has `embedding_recipe`, `build_from_recipe` then `load_state_dict`; else the transition branch (section 5).
- `device` on `EquineProtonet.load` keeps its existing behaviour (`map_location` for the checkpoint and the archive, overriding `settings["device"]`). Adding `device` to `EquineGP.load` and `load_equine_model` is Phase 2 work (PR-2b) and is not part of this PR. (Done in PR-2b, 2026-09-29: both accept `device` positionally after the path, like `EquineProtonet.load`.)
- `EquineGP` gets `_to_checkpoint` / `_from_checkpoint` mirroring Protonet's, so each class defines its format in exactly one place, and `save`, `load` and `load_equine_model` contain no format knowledge.

## 5. Transition branch and its removal

**This release (N):**
- `save(path, allow_executable=True)` embeds a TorchScript archive (`embed_jit_save` as a `uint8` tensor) for an embedding with no recipe, and sets `contains_executable: True`.
- `load` of a file with `contains_executable: True`, or of any file that has `embed_jit_save` at all, raises `ValueError` unless `trust_executable=True`, checked once at the top of the embedding rebuild before anything else is inspected. `allow_unsafe_legacy_format=True` implies `trust_executable=True`, because a legacy file is already trusted by definition.
- **Migration:** `load(path, trust_executable=True or allow_unsafe_legacy_format=True, embedding_model=Registered(...))` copies the archive's `state_dict` into the supplied registered module. The next `save()` writes a data-only file. This is the documented way to convert every pre-existing file.
- The Python 3.14 workaround (`prepare_jit_module`) and `load_jit_archive` stay, used only by this branch.

**Next release (N+1):**
- Remove `allow_executable`, `trust_executable`, `embed_jit_save` handling, `load_jit_archive`, `jit_archive_to_tensor`, `prepare_jit_module`, and the torch `< 2.10` ceiling whose reason was TorchScript. `torch.jit` is no longer imported anywhere in `src/equine`.
- A file with `contains_executable: True` raises `ValueError`: "saved with executable content by EQUINE <N>; migrate it with EQUINE <N> (`load(..., trust_executable=True, embedding_model=...)`, then `save()`)".
- `allow_unsafe_legacy_format` stays for one more release for v1 files, then is removed on the same terms.

The changelog for release N states this schedule so users know they have one release to migrate.

## 6. Errors

All refusals are `ValueError` and name the remedy; the constructor-argument check is `TypeError`.

| Situation | Message must contain |
|---|---|
| `save` with an unregistered embedding and no opt-in | the class name, `@equine.embedding_architecture(...)`, `allow_executable=True`, and that the latter produces executable content requiring `trust_executable=True` |
| `load` with an unknown recipe name | the name, the list of registered names, the decorator, `embedding_model=` |
| `load` of a flagged file without trust | `trust_executable=True` and the migration route |
| non-plain constructor argument | the argument name and the allowed types |
| duplicate registration of a different class | both class paths |
| `load` of a v1 file without the legacy opt-in | unchanged from PR-0 |
| tensor with expanded/overlapping storage anywhere in the file | refused; message names the condition |
| `embedding_model=` whose parameters do not match the file's weights | count of mismatches and up to three truncated key names |
| `save` of a module that cannot be rebuilt from its recipe | says so and points at the registry docstring's constructor requirements |
| `settings` key the model class's constructor does not accept | count of unknown keys and up to three truncated key names (a 0.1.5 GP file's `use_temperature` is dropped with a `UserWarning` instead) |

## 7. Testing

New `tests/test_registry.py` and additions to `tests/test_safe_loading.py`; `tests/conftest.py`'s `BasicEmbeddingModel` gains the decorator, which is the only change existing tests need.

- Registry: shipped MLP records its recipe with defaults applied; non-plain argument rejected at construction; `*args` constructor rejected at registration; duplicate name for a different class rejected; re-registration idempotent; `registered_architectures` lists both shipped and user names.
- Round trip, both classes: file loads with `torch.load(weights_only=True)`; `contains_executable` is `False`; no `embed_jit_save` key; rebuilt embedding has the registered type; support keys are ints; label and feature names preserved; predictions and OOD scores equal to `1e-6`. Device tests are Phase 2 (PR-2a).
- Override: `embedding_model=` with a fresh registered instance reproduces predictions.
- Unknown recipe in a fresh interpreter (subprocess) yields the actionable error.
- Transition: unregistered embedding refused without opt-in; opt-in writes a flagged file readable with `weights_only=True`; flagged file refused without trust through `load` and `load_equine_model`; loads with trust; migration via `embedding_model=` yields a data-only file with equal predictions, for both classes and for both legacy and flagged inputs.
- Legacy: existing legacy tests keep passing with the trust implication.
- Cross-version fixture from PR-1c keeps loading (when PR-1c has landed; otherwise a v2 file written by this branch's own code is committed as the fixture).

Coverage stays at or above the 97 % gate; ruff and codespell clean.

## 8. Documentation

Docstrings on every public function and on both `save`/`load`. A `CHANGELOG.md` entry (created by PR-0 if not present) describing: the new format, the decorator, the migration command, the transition branch and its removal in N+1. No edits to `README.md`; a suggested README paragraph is listed in the PR body for the maintainers to apply if they wish.

## 9. Web app, stage 1 (separate repo, same release)

Not part of this PR, but the PR is not released until this is ready in equine-webapp:
- bump the web app's `torch<2.6` pin to `torch>=2.6,<2.10` (required by the equine release: CVE-2025-32434 bypass of `weights_only` below 2.6);
- training: `model.save(path, allow_executable=True)`;
- the cached loader: `lambda p: eq.load_equine_model(p, trust_executable=True)`;
- `weights_only=True` on the summary view's `torch.load` and on uploaded `.pt` datasets;
- a startup migration of the models folder: legacy files re-saved with `allow_executable=True`.
Stage 2 (architecture catalogue, `.jit` upload retired, files data-only) must land before equine release N+1 removes the transition branch.

## 10. Relationship to the spike

Branch `spike/d191-registry` on the fork implemented sections 3 to 6 for `EquineProtonet` and the deferred Hub export. PR-0b re-implements from this spec with `EquineGP` parity and without the Hub code; the spike is deleted once PR-0b is merged on the fork.
