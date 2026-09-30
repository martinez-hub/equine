# Stack 2: Device and Mode Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Both model classes train, predict, update their support, save and load correctly on CPU, MPS and CUDA; prediction runs one embedding pass without autograd; train/eval mode cannot corrupt state. CPU numerics do not change: the Stack 1 golden and fixture tests pass untouched through every layer.

**Architecture:** Four stacked PRs (roadmap PR-2a to PR-2d) on top of Stack 1 (`stack1/pr-1e-ci-tooling`, tip 54d9181). PR-2a adds device-parametrized tests with strict expected failures for exactly what fails today; PR-2b fixes device placement and adds `device` to the GP and generic loaders; PR-2c collapses the double embedding pass and disables autograd at inference; PR-2d fixes mode handling. Each fixing layer flips the corresponding expected failures.

**Tech Stack:** PyTorch 2.6–2.9 (MPS has no float64 and no `cholesky_inverse` kernel; `cholesky`, `cholesky_ex`, `linalg.inv`, `linalg.solve` exist), beartype on both classes, icontract, pytest with hypothesis and xdist. CI is Ubuntu-only with no accelerator, so every accelerator test is skipped there; the MPS proof happens on the maintainer's M1.

**Facts established on 2026-09-29 (torch 2.9.1, M1, MPS available):**

| Path on MPS today | Result |
|---|---|
| Protonet train, predict, update_support, save/load, `use_temperature=False` | works |
| Protonet train with `use_temperature=True` | works |
| Protonet predict with `use_temperature=True` | `RuntimeError: ... mps:0 and cpu` at `dists / self.temperature` (#170) |
| `temperature` buffer placement, both classes | stays on CPU (registered after `Equine.__init__` moved the module) (#170) |
| Protonet raw `Protonet.support[label]` tensors after training on MPS | stay on CPU (stored as given); embeddings, prototypes, covariances land on MPS |
| GP train (no `vis_support`) | works; `precision` on MPS, but `_Laplace.seen_data` is reassigned on CPU by `reset_precision_matrix` |
| GP update_support, train with `vis_support=True` | `RuntimeError: Tensor for argument input is on cpu but expected on mps` in `compute_embeddings` (#177) |
| GP predict, save+load then predict (forward already moves its input) | `NotImplementedError: aten::cholesky_inverse.out ... not implemented for MPS` (#173) |
| Predict with float64 input, model on CPU, both classes | `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float` (no cast anywhere today) |
| Protonet predict with float64 input, model on MPS | `TypeError: Cannot convert a MPS Tensor to float64` |
| Protonet `update_support` on an untrained (training-mode) model, CPU | `AttributeError: 'Protonet' object has no attribute 'global_mean'` (#179) |
| GP `predict` after an explicit `.train()`, CPU | `AssertionError: Did not reset precision matrix at start of epoch` on the third call (#209) |

**Branch and workflow:** worktree `/Users/josuemartinez/Documents/PersonalProjects/Equine/equine-stack2`, venv `.venv` (`sitecustomize.py` puts `src` on the path). Tests: `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 2` (381 passed, 2 xfailed at the base). Lint: `uvx ruff check --fix src tests && uvx ruff format src tests`; `uvx codespell src`. Accelerator tests: run without `-n` when debugging MPS (`-p no:xdist`), they are also fine under `-n 2`. Linux check for CPU paths: the Docker recipe in the Stack 1 plan.

| Layer | Branch | Base |
|---|---|---|
| PR-2a | `stack2/pr-2a-device-scaffold` | `stack1/pr-1e-ci-tooling` |
| PR-2b | `stack2/pr-2b-device-plumbing` | PR-2a |
| PR-2c | `stack2/pr-2c-single-pass` | PR-2b |
| PR-2d | `stack2/pr-2d-mode-handling` | PR-2c |

Rules: do not edit `README.md`; keep `docs/superpowers/` out of the site (already excluded); no change to the saved-file format (`equine_format_version` stays 2; settings keys unchanged: both classes already persist `"device"` as a string); golden literals and fixtures are never touched in this stack; every `src/` change in a layer must be covered by a test that fails before it.

---

## File structure

| File | Responsibility in this stack |
|---|---|
| `tests/conftest.py` | `available_devices()`, `ACCELERATOR` constant, `assert_on_device(model, device)` helper, `CountingEmbedding` wrapper |
| `tests/test_devices.py` (new, PR-2a) | device-parametrized train/predict/update_support/save/load tests for both classes; buffer-placement tests; the strict xfails that PR-2b flips |
| `tests/test_inference.py` (new, PR-2c) | one embedding pass per predict, no autograd on outputs, `update_support` embeds once per class |
| `tests/test_modes.py` (new, PR-2d) | untrained `update_support`, predict after `.train()` leaves state untouched, wrapper/inner mode consistency |
| `src/equine/equine.py` | unchanged in substance (base `__init__` already stores `device: str` and moves the module) |
| `src/equine/equine_protonet.py` | temperature placement, assigned `.to()` calls, support moved, dtype/device of allocated tensors, `_forward_with_embeddings`, `no_grad` predict, unconditional global moments, `eval()` in predict/update_support |
| `src/equine/equine_gp.py` | `device` to `super().__init__`, `device: str` + deprecated `device_type`, `compute_embeddings` moves/casts input, `seen_data` device, CPU fallback for `cholesky_inverse`, `load(device=)`, `_forward_with_features`, `no_grad` predict, wrapper-level `train()/eval()` |
| `src/equine/load_equine_model.py` | `device` parameter forwarded to `_from_checkpoint` |
| `src/equine/utils.py` | `_rebuild_embedding` moves a recipe-built embedding to `device` (today it is built on CPU and moved by the constructor; keep behaviour, add nothing unless a test needs it) |

---

## Layer PR-2a — Device test scaffold (#188)

Branch: `stack2/pr-2a-device-scaffold` (already created from the Stack 1 tip). Tests only; no `src/` changes.

### Task 1: device helpers in conftest

**Files:** Modify `tests/conftest.py`.

- [ ] **Step 1: Add the helpers** (below `assert_valid_prediction`):

```python
def available_devices() -> list[str]:
    """CPU plus whichever accelerator this machine has. CI (Ubuntu) has none."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


ACCELERATOR = next((d for d in available_devices() if d != "cpu"), None)


def stored_tensors(model) -> dict[str, torch.Tensor]:
    """Every tensor a trained equine model keeps: parameters, buffers, support,
    prototypes, covariances. Used to assert device placement after train/load."""
    found: dict[str, torch.Tensor] = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        found[name] = t
    inner = model.model
    for attr in ("prototypes", "covariance", "global_mean", "global_covariance"):
        t = getattr(inner, attr, None)
        if torch.is_tensor(t) and t.numel() > 0:
            found[f"model.{attr}"] = t
    for attr in ("support", "support_embeddings"):
        d = getattr(inner, attr, None) or getattr(model, attr, None) or {}
        for label, t in d.items():
            if torch.is_tensor(t):
                found[f"{attr}[{label}]"] = t
    return found


def assert_on_device(model, device: str) -> None:
    wrong = {
        name: str(t.device)
        for name, t in stored_tensors(model).items()
        if t.device.type != torch.device(device).type
    }
    assert wrong == {}, f"tensors not on {device}: {wrong}"
```

Read `Protonet` / `EquineGP` to confirm which of those attributes exist on each inner module (`support` lives on `EquineGP` itself and on `Protonet` for the Protonet; adjust the lookups so nothing raises for a missing attribute).

- [ ] **Step 2: Register the marker semantics.** `accelerator` is already registered in `pytest_configure`; add a second line registering `device: parametrized over available_devices()` (informational).

- [ ] **Step 3: Commit** `test: device helpers (available_devices, assert_on_device) (#188)`.

### Task 2: device-parametrized behaviour tests with strict xfails

**Files:** Create `tests/test_devices.py`.

- [ ] **Step 1: Write the module.** Use `golden_data.separable_dataset()` (float32, 120×6×3) and `BasicEmbeddingModel(FEATURES, CLASSES)`; short training (Protonet 5 episodes; GP 2 epochs, batch 32, lr 0.05). Parametrize with `@pytest.mark.parametrize("device", available_devices())` and mark the function `@pytest.mark.accelerator` only where the CPU case is already covered elsewhere (do not skip CPU: CPU cases must run in CI).

Tests (each parametrized over device; `MPS = pytest.mark.xfail(strict=True, raises=(RuntimeError, NotImplementedError, TypeError), reason=...)` applied via `pytest.param(..., marks=...)` ONLY for the accelerator case and ONLY where the fact table above says it fails today; the CPU case is never marked):

```python
def _protonet(device, use_temperature=False):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES,
                              use_temperature=use_temperature, device=device)
    model.train_model(dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30)
    return model, x, y

def _gp(device):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES,
                        num_random_features=16, device=device)
    model.train_model(dataset, torch.nn.CrossEntropyLoss(),
                      torch.optim.SGD(model.parameters(), lr=0.05), num_epochs=2, batch_size=32)
    return model, x, y
```

1. `test_protonet_trains_and_predicts[device]`: train, `assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)`; input on CPU. Passes everywhere today.
2. `test_protonet_with_temperature_predicts[device]`: same with `use_temperature=True`. Accelerator case xfail `reason="#170: temperature buffer stays on CPU"`.
3. `test_protonet_update_support[device]`: `model.update_support(x, y.float(), 0.5)` then predict. Passes today.
4. `test_protonet_save_load_round_trip[device]`: save to `tmp_path`, `eq.load_equine_model(path)` (lands on the saved device), `EquineProtonet.load(path, device)`; predictions equal (`atol=1e-5, rtol=0`) to the original on the same input. Passes today.
5. `test_stored_tensors_on_device[protonet|gp][device]`: `assert_on_device(model, device)` after training. Accelerator case xfail for BOTH classes `reason="#170: temperature registered after the module was moved"`. Measured: Protonet leaves `temperature` and the raw `support[label]` tensors on CPU; GP leaves `temperature` and `_Laplace.seen_data` on CPU (training itself works). PR-2b Tasks 3 and 4 fix all four.
6. `test_gp_trains[device]`: train without `vis_support`, then `assert model.model.precision.device.type == torch.device(device).type`. Passes today.
7. `test_gp_predicts[device]`: predict on CPU input. Accelerator xfail `reason="#177/#173: compute_embeddings does not move its input; cholesky_inverse has no MPS kernel"`.
8. `test_gp_update_support[device]`: `update_support(x, y.long(), 10)`. Accelerator xfail (#177).
9. `test_gp_vis_support_training[device]`: `train_model(..., vis_support=True, support_size=10)`. Accelerator xfail (#177).
10. `test_gp_save_load_round_trip[device]`: save, `load_equine_model`, predict equal. Accelerator xfail (#177). Additionally `test_gp_load_onto_device[device]` marked `pytest.mark.skip(reason="#188: EquineGP.load(device=) is added in PR-2b")` for now.
11. `test_predict_accepts_float64_input[protonet|gp][device]`: `model.predict(x[:5].double())`. Fails today on EVERY device (CPU: dtype mismatch in the first Linear; MPS: no float64), so both cases are `xfail(strict=True, raises=(RuntimeError, TypeError), reason="#188: inputs are cast to the embedding dtype in PR-2b")`. Decision taken 2026-09-29: PR-2b casts inputs to the embedding's parameter dtype on every device (Task 5, Step 3); both marks come off together there.
12. `test_device_attribute_is_a_string[protonet|gp]` (CPU only): `isinstance(model.device, str)`. GP case xfail `strict=True, raises=AssertionError, reason="#216: EquineGP.device is a torch.device"`.

- [ ] **Step 2: Run on this machine** (`.venv/bin/python -m pytest tests/test_devices.py -q -p no:cacheprovider -rxXs`): every CPU case passes; the marked MPS cases show `xfailed`; nothing `xpassed`. If a case you marked passes, remove the mark (the fact table is what you verify against, not the roadmap). If a case you did not mark fails, report it with the traceback rather than marking it.

- [ ] **Step 3: Run the whole suite** with `-n 2`; then run `tests/test_devices.py` with `CUDA_VISIBLE_DEVICES= ` and a monkeypatched `torch.backends.mps.is_available` returning False (or simply confirm by reading: with no accelerator, `available_devices()` is `["cpu"]`, so the parametrization has one case and no xfail marks are applied). The Linux container run in Task 11 is the real check.

- [ ] **Step 4: Commit** `test: device-parametrized train/predict/update/save/load tests with strict xfails for what fails on MPS today (#188)`.

Layer exit: suite green on the M1 with the new xfails; push `git push -u fork stack2/pr-2a-device-scaffold`.

---

## Layer PR-2b — Device plumbing (#170, #177, #206, #216, GP `load(device)`)

Branch: `git checkout -b stack2/pr-2b-device-plumbing` from PR-2a's tip.

### Task 3: temperature placement and `device: str` on both classes (#170, #216)

**Files:** Modify `src/equine/equine_protonet.py` (`EquineProtonet.__init__` ~486-548, `train_model` ~614-617, `predict` ~826), `src/equine/equine_gp.py` (`EquineGP.__init__` ~450-511, `calibrate_model` ~768, `forward` ~790, `_to_checkpoint` ~866), `tests/test_devices.py` (flip xfails), `tests/test_equine_gp.py` (deprecation test).

- [ ] **Step 1: Failing tests.** Flip (remove the xfail marks of) `test_stored_tensors_on_device[*]` for the Protonet case and `test_protonet_with_temperature_predicts`, and `test_device_attribute_is_a_string[gp]`. Add to `tests/test_equine_gp.py`:

```python
def test_device_type_is_deprecated_alias() -> None:
    model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16)
    with pytest.warns(DeprecationWarning, match="device_type"):
        assert model.device_type == "cpu"
    assert model.device == "cpu"
```

Run: they fail (`temperature` on CPU; `device` is a `torch.device`; no warning).

- [ ] **Step 2: Protonet.** In `EquineProtonet.__init__`, after `self.register_buffer("temperature", ...)` and after `self.model = Protonet(...)`, add `self.to(self.device)` as the LAST statement of `__init__`. In `train_model` (the `if self.use_temperature:` re-creation), build the tensor on the right device: `torch.full((1,), self.init_temperature, dtype=self.temperature.dtype, device=self.temperature.device)`. In `predict`, `dists = dists / self.temperature` stays (the buffer is now on the device). In `calibrate_temperature`, `self.temperature.to(self.device)` becomes unnecessary but harmless; leave it.

- [ ] **Step 3: GP.** In `EquineGP.__init__`: pass `device=device` to `super().__init__(...)`; delete `self.device_type = device` and `self.device: torch.device = torch.device(self.device_type)`; keep `self.device` as the base class set it (a `str`); after registering `temperature` and building `self.model`, end `__init__` with `self.to(self.device)` (this also moves `self.model`, so `self.model.to(self.device)` can go). Add:

```python
@property
def device_type(self) -> str:
    """Deprecated alias of ``device``; removed in the next release."""
    warnings.warn(
        "EquineGP.device_type is deprecated; use EquineGP.device (a str)",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.device
```

In `_to_checkpoint`, `"device": self.device_type` → `"device": self.device` (the persisted value was already the string, so files do not change). `forward`'s `self.temperature.to(self.device)` may stay. Everywhere `self.device` was used as a `torch.device` (e.g. `torch.device(...)` comparisons, `.type`), it is now a string; grep the file and adjust (`torch.device(self.device)` where a device object is needed).

- [ ] **Step 4: Run** `tests/test_devices.py tests/test_equine_gp.py tests/test_golden.py tests/test_fixtures.py tests/test_persistence.py tests/test_safe_loading.py` then the whole suite. The golden and fixture tests must pass unchanged. `test_safe_loading.py:149` writes `model.device_type` into a legacy file: change that helper to `model.device` (it is a test of the legacy reader, not of the alias).

- [ ] **Step 5: Commit** `fix: temperature buffer moves with the module; EquineGP.device is a str with a deprecated device_type alias (#170, #216)`.

### Task 4: inputs and stored tensors follow the model device (#177, #206, #173 device half)

**Files:** Modify `src/equine/equine_protonet.py` (`train_model` ~628-631, `Protonet.update_support` ~440, `compute_covariance_by_type` ~229-236 UNIT, `compute_shared_covariance` ~326-338, `regularize_covariance` ~274), `src/equine/equine_gp.py` (`compute_embeddings` ~672, `update_support` ~649, `predict` ~811-814, `_Laplace.reset_precision_matrix` ~266, `_from_checkpoint` ~1056), `tests/test_devices.py`.

- [ ] **Step 1: Failing tests.** Flip `test_gp_update_support`, `test_gp_vis_support_training`, `test_gp_save_load_round_trip`, `test_stored_tensors_on_device[gp]` (GP predict stays xfail until Task 5 because of `cholesky_inverse`; if `update_support` on MPS also reaches the Laplace eval forward, keep its mark until Task 5 and say so). Add `test_protonet_train_moves_data_once` (CPU): wrap `train_x.to` is not observable; instead assert after training on the accelerator that `model.model.support[label].device.type == device` (covered by `assert_on_device`).

- [ ] **Step 2: Protonet.** Assign the four moves: `train_x = train_x.to(self.device)` etc. In `Protonet.update_support`, store `self.support = OrderedDict((label, t.to(self.device)) for label, t in support.items())`. In `compute_covariance_by_type` UNIT branch: `torch.ones(self.emb_out_dim, device=self.device)`; in `compute_shared_covariance`: `torch.zeros(..., device=self.device)`. Keep numerics identical (same ops, same order) so CPU goldens do not move.

- [ ] **Step 3: GP.** `compute_embeddings(self, x)`: first line `x = x.to(self.device)`. `update_support`: `self.support = OrderedDict((label, t.to(self.device)) ...)` before embedding. `predict`: `equiprobable = torch.ones(self.num_outputs, device=logits.device) / self.num_outputs`, and pass the moved `X` (`X = X.to(self.device)` at the top; Task 5 removes the second pass entirely). `_Laplace.reset_precision_matrix`: `self.seen_data = torch.tensor(0, device=self.precision.device)`; in `_from_checkpoint`, move the restored `seen_data` to `eq_model.model.precision.device`.

- [ ] **Step 4: Run** targeted files, goldens/fixtures, whole suite. **Commit** `fix: inputs, support and allocated tensors follow the model device on both classes (#177, #206)`.

### Task 5: `cholesky_inverse` on devices without the kernel; MPS dtype policy

**Files:** Modify `src/equine/equine_gp.py` (`_Laplace.forward` eval branch ~359-368), `src/equine/equine_protonet.py` (`Protonet.compute_embeddings` ~154), `src/equine/equine_gp.py` (`compute_embeddings`), `tests/test_devices.py`.

- [ ] **Step 1: Failing tests.** Flip `test_gp_predicts` and `test_predict_accepts_float64_input[*][mps]`.

- [ ] **Step 2: Cholesky fallback.** Replace `torch.cholesky_inverse(u, out=self.covariance)` with a helper in `equine_gp.py`:

```python
def _cholesky_inverse(u: torch.Tensor) -> torch.Tensor:
    """cholesky_inverse with a CPU round trip where the kernel is missing (MPS)."""
    if u.device.type == "mps":
        return torch.cholesky_inverse(u.cpu()).to(u.device)
    return torch.cholesky_inverse(u)
```

and `self.covariance.copy_(_cholesky_inverse(u))` (keeps the buffer identity, so state_dict and `out=` semantics are preserved). On CPU and CUDA the op and its numerics are unchanged. Do NOT switch to `linalg.inv` or `cholesky_solve`: that would move CPU numbers.

- [ ] **Step 3: Dtype policy at the model boundary.** In `Protonet.compute_embeddings` and `EquineGP.compute_embeddings`, cast to the embedding's parameter dtype: `param = next(self.embedding_model.parameters(), None)`; `X = X.to(device=self.device, dtype=param.dtype if param is not None else X.dtype)`. Hmm: on CPU this changes behaviour for float64 inputs (today a float64 input into a float32 Linear raises `RuntimeError: mat1 and mat2 must have the same dtype`; check what test 11's CPU case showed in PR-2a). Policy: cast on every device (consistent), document it in both classes' `predict` docstrings ("inputs are cast to the embedding model's parameter dtype"), and note it in the PR body as a behaviour change that only affects inputs that previously raised. Float32 inputs are unaffected, so goldens do not move.

- [ ] **Step 4: Run** `tests/test_devices.py` (no xfails left except `test_gp_load_onto_device` skip), goldens, fixtures, whole suite. **Commit** `fix: GP covariance on devices without cholesky_inverse; inputs cast to the embedding dtype at the model boundary`.

### Task 6: `device` on `EquineGP.load` and `load_equine_model`

**Files:** Modify `src/equine/equine_gp.py` (`load` ~899, `_from_checkpoint` ~965-1004, `_require_device` call ~1032), `src/equine/load_equine_model.py` (~17-89), `tests/test_devices.py`, `tests/test_persistence.py`.

- [ ] **Step 1: Failing tests.** Un-skip `test_gp_load_onto_device[device]`: `EquineGP.load(path, device)` and `eq.load_equine_model(path, device=device)`; assert `model.device == device`, `assert_on_device`, predictions equal to the original. Add in `tests/test_persistence.py`: `load_equine_model(path, device="cpu")` for a GP saved on CPU works; `device="meta"` refused with the existing message; a saved-on-MPS file (build one on the M1 in the test, skip without accelerator) loads onto CPU with `device="cpu"` and predicts within `1e-4` of the MPS model.

- [ ] **Step 2: Implement.** `EquineGP.load(cls, path, device: Optional[str] = None, *, ...)` mirroring the Protonet: `load_checkpoint(path, map_location=device, ...)`; `_from_checkpoint(..., device=None, ...)`: `_rebuild_embedding(model_save, device, ...)`, then the same override block as the Protonet (`if device is not None: settings["device"] = device else _require_device(settings, hint=...)`); after `load_state_dict`, `eq_model.to(eq_model.device)` is redundant (constructor moved it) but move `seen_data`. `load_equine_model(model_path, device: Optional[str] = None, *, ...)`: pass `map_location=device` to `load_checkpoint` and `device=device` to both `_from_checkpoint`s (the Protonet's already accepts it). Keep the parameter order of the Protonet's `load` (positional `device` second) for consistency; update the docstrings.

- [ ] **Step 3: Run** everything; **Commit** `feat: EquineGP.load and load_equine_model accept device= like EquineProtonet.load (#188)`.

Layer exit: `tests/test_devices.py` fully green on the M1 (no xfails, no skips except accelerator-only tests on machines without one); golden/fixture untouched; whole suite green; push `stack2/pr-2b-device-plumbing`.

---

## Layer PR-2c — Single forward pass and no autograd at inference (#173, #182, #212)

Branch: `git checkout -b stack2/pr-2c-single-pass` from PR-2b's tip.

### Task 7: counting tests

**Files:** Modify `tests/conftest.py`; create `tests/test_inference.py`.

- [ ] **Step 1: Helper** in conftest:

```python
class CountingEmbedding(torch.nn.Module):
    """BasicEmbeddingModel that counts forward calls (not registered: tests build it directly)."""
    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.inner = BasicEmbeddingModel(tensor_dim, num_classes)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return self.inner(x)
```

- [ ] **Step 2: Tests** (CPU, seeded, short training as in `test_devices.py`, using `CountingEmbedding`):
  - `test_predict_embeds_once[protonet|gp]`: reset `calls = 0`, `predict(x[:5])`, `assert emb.calls == 1`. Today: 2 for both.
  - `test_predict_outputs_carry_no_autograd[protonet|gp]`: `out.classes.grad_fn is None and out.embeddings.grad_fn is None and not out.ood_scores.requires_grad`. Today: fails.
  - `test_predict_outputs_are_plain_tensors_usable_in_autograd[protonet|gp]`: `(out.embeddings.clone().requires_grad_() * 2).sum().backward()` works (guards against `inference_mode`).
  - `test_gp_update_support_embeds_once_per_class`: after `update_support(x, y.long(), 10)` calls == number of classes (today 2×).
  - `test_protonet_train_model_embeds_calibration_set_once`: harder to count exactly; assert `calls` after `train_model` equals the count of a hand-derived formula OR skip this and instead assert the calibration path calls `_forward_with_embeddings` once by counting calls during `update_support(x, y.float(), 0.5)` on the Protonet: today 2 passes for the calib set + 1 per class for support; after: 1 + classes.
  All marked `xfail(strict=True, raises=AssertionError, reason="#173/#182/#212")` in this commit, flipped in Task 8.

- [ ] **Step 3: Commit** `test: predict must embed once and return autograd-free tensors; update_support embeds once per class (#173, #182, #212)`.

### Task 8: implement the single pass

**Files:** Modify `src/equine/equine_gp.py` (`_Laplace.forward` ~340-379, `EquineGP.forward` ~788, `predict` ~809-814, `compute_prototypes` ~691, `update_support`, `calibrate_model` ~767), `src/equine/equine_protonet.py` (`Protonet.forward` ~417-429, `train_model` ~660-661, `calibrate_temperature` ~709-711, `predict` ~821-829, `update_support` ~881-882), tests flipped.

- [ ] **Step 1: GP.** Add `_Laplace._forward_with_features(self, x) -> tuple[Tensor, Tensor]` returning `(logits, features)` where `features` is the post-`rff` embedding that `compute_embeddings` returns today (read `compute_embeddings` to match exactly: feature_extractor → jl → normalize → rff). `_Laplace.forward(x)` becomes `return self._forward_with_features(x)[0]` (unchanged output). `EquineGP._forward_with_embeddings(X)`: `X = X.to(self.device)`; `logits, emb = self.model._forward_with_features(X)`; `return logits / self.temperature, emb`. `forward` returns the logits only. `predict`: under `with torch.no_grad():`, one call to `_forward_with_embeddings`, then softmax/entropy/ood as today, `equiprobable` on `logits.device`. `compute_prototypes`: use `self.support_embeddings` when populated (fall back to recomputing when empty, e.g. the load path, which can instead populate `support_embeddings` first). `calibrate_model`: `logits / self.temperature` fine (device fixed in 2b). Roadmap mentions returning `pred_var_diag` for PR-3e/PR-5b: return it as a third element ONLY if it is free (it is the `torch.diag(pred_cov)` already computed in eval); if you add it, `forward` still returns logits only.

- [ ] **Step 2: Protonet.** `Protonet._forward_with_embeddings(X) -> (classes, distances, embeddings)`; `forward` returns `(classes, distances)` via it. `EquineProtonet.predict`: `with torch.no_grad(): preds, dists, X_embed = self.model._forward_with_embeddings(X)`; rest unchanged. `train_model` lines 660-661 and `update_support` 881-882: one call. `calibrate_temperature`: keeps `no_grad` around the forward and needs no embeddings.

- [ ] **Step 3: Flip the Task 7 xfails; run** `tests/test_inference.py`, goldens, fixtures, `tests/test_devices.py`, whole suite. Numerics: the embeddings and distances are the same tensors computed once, so goldens must not move; if any golden literal moves, stop and report (it means the two passes were not equivalent, e.g. one used `training`-mode dropout or a different normalisation).

- [ ] **Step 4: Commit** `perf: one embedding pass per predict/update_support and no autograd at inference (#173, #182, #212)`.

Layer exit: suite green, goldens untouched; push `stack2/pr-2c-single-pass`.

---

## Layer PR-2d — Mode handling (#209, #179)

Branch: `git checkout -b stack2/pr-2d-mode-handling` from PR-2c's tip.

### Task 9: mode tests

**Files:** Create `tests/test_modes.py`.

- [ ] **Step 1: Tests** (CPU, seeded, `xfail(strict=True, raises=(AssertionError, AttributeError), reason="#209/#179")` in this commit):
  - `test_protonet_update_support_on_untrained_model`: fresh `EquineProtonet`, `update_support(x, y.float(), 0.5)`, then `assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)`. Today: `AttributeError: global_mean`.
  - `test_protonet_update_support_in_train_mode_computes_global_moments`: trained model, `model.train()`, `update_support(...)`, `assert model.model.global_mean is not None` and predict works.
  - `test_gp_predict_after_train_mode_does_not_touch_precision`: trained GP, `model.train()`, snapshot `precision.clone()` and `seen_data.clone()`, call `predict` three times, assert both unchanged and no exception. Today: `AssertionError: Did not reset precision matrix`.
  - `test_gp_train_model_leaves_wrapper_and_inner_in_eval`: after `train_model`, `model.training is False and model.model.training is False`. Today: wrapper `True`.
  - `test_predict_leaves_model_in_eval[protonet|gp]`: after `predict`, `model.training is False`.

- [ ] **Step 2: Commit** `test: mode handling: untrained update_support, predict in train mode, wrapper/inner consistency (#209, #179)`.

### Task 10: implement

**Files:** Modify `src/equine/equine_gp.py` (`train_model` ~575, 589; `predict`), `src/equine/equine_protonet.py` (`Protonet.update_support` ~452-460; `EquineProtonet.update_support`; `predict`).

- [ ] **Step 1: GP.** `train_model`: `self.train()` at the start of the loop body in place of `self.model.train()`, and `self.eval()` in place of `self.model.eval()` (the wrapper's call propagates to the inner module). `predict`: first statement `self.eval()`; document in the docstring that predict switches the model to eval mode.

- [ ] **Step 2: Protonet.** `Protonet.update_support`: compute `self.compute_global_moments()` unconditionally; keep the covariance choice (`PRED_COV_TYPE` in eval, `self.cov_type` in training) exactly as today so trained-model numerics do not move. `EquineProtonet.update_support`: `self.eval()` as its first statement. `EquineProtonet.predict`: `self.eval()` first.

- [ ] **Step 3: Flip the Task 9 xfails; run** `tests/test_modes.py`, goldens, fixtures, whole suite. **Commit** `fix: update_support works on an untrained model; predict runs in eval mode; GP toggles mode on the wrapper (#209, #179)`.

Layer exit: suite green; goldens untouched; push `stack2/pr-2d-mode-handling`.

---

### Task 11: verification and opening Stack 2 on the fork

- [ ] **Step 1:** On the top branch: `uvx ruff check src tests && uvx ruff format --check src tests && uvx codespell src`; full suite with coverage `--cov-fail-under=97`; `tests/test_devices.py tests/test_inference.py tests/test_modes.py -p no:xdist -rxXs` on the M1 (all MPS cases pass, none xfailed/xpassed); Linux container run of the whole suite (CPU cases; accelerator tests skipped); `git status --porcelain` clean.
- [ ] **Step 2:** Open four drafts on `martinez-hub/equine`, bottom first, bases chained from `stack1/pr-1e-ci-tooling`, bodies stating the behaviour changes (predict switches to eval mode; inputs cast to the embedding dtype; `device_type` deprecated; `load(device=)` added) and that goldens/fixtures are untouched. Do not open anything upstream.
- [ ] **Step 3:** Run the adversarial workflow over the stack (base `stack1/pr-1e-ci-tooling` tip, head PR-2d tip, `code` profile, thorough) and fix confirmed findings in their layers.

---

## Self-review

**Spec coverage (roadmap Phase 2):** PR-2a Tasks 1–2 (scaffold, parametrization, xfail only what fails today, GP `load(device)` skip). PR-2b Tasks 3–6 (temperature buffer then `.to`; GP passes `device` to super; `compute_embeddings`/`update_support` moves; four assigned `.to`; `Protonet.update_support` moves support; GP `load(device)` + `load_equine_model(device)`; `device` is `str` with deprecated `device_type`; MPS `cholesky_inverse` on CPU; MPS dtype policy = cast at the boundary, code fix preferred over the env var). The roadmap's optional "map the Protonet checkpoint to CPU and move only the module" is deliberately not done: the TorchScript path is transitional and removed next release. PR-2c Tasks 7–8 (`_forward_with_features`, `_forward_with_embeddings`, four Protonet callers, `no_grad` not `inference_mode`, GP `update_support` reuses embeddings). PR-2d Tasks 9–10 (wrapper `train()/eval()`, `predict` calls `eval()`, unconditional global moments, untrained `update_support`). Exit: MPS tests pass on the M1; goldens and fixtures untouched.

**Deviations from the roadmap, from the 2026-09-29 facts:** #170 manifests as a crash in Protonet `predict` (not in training) plus wrong buffer placement on both classes (`temperature`; Protonet raw support; GP `seen_data`); the GP has two independent MPS failures (input move in `update_support`/`vis_support`, missing `cholesky_inverse` kernel in `predict`) fixed in Tasks 4 and 5; float64 inputs fail on every device today, and PR-2b casts them to the embedding dtype (Task 5, Step 3).

**Placeholders:** none; every step names the functions and lines (approximate, from the 2026-09-29 survey; re-read before editing).

**Type consistency:** `available_devices()`, `ACCELERATOR`, `assert_on_device`, `stored_tensors`, `CountingEmbedding` defined in Tasks 1 and 7 and used by name afterwards; `_forward_with_features` (GP inner), `_forward_with_embeddings` (both wrappers / Protonet inner) named consistently across Tasks 7–8; `_cholesky_inverse` defined in Task 5.
