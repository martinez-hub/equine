# Stack 1: Test and CI Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make regressions visible before the behaviour-changing phases: tests that leave no files behind and assert real values, a deterministic and wider hypothesis strategy, golden-value and cross-version fixtures, an honest coverage gate, and CI/tooling that runs what the manifest declares.

**Architecture:** Five stacked PRs (roadmap PR-1a to PR-1e), each its own branch on the fork built on the previous, all on top of PR 2 (`feat/persistence-as-code`) because Stack 0 rewrote the test fixtures. No `src/` behaviour changes anywhere in this stack. Two tests are landed as `xfail(strict=True)` against known bugs (#174, #171) and flipped in Phase 3.

**Tech Stack:** pytest, hypothesis, pytest-cov, pytest-xdist, ruff, codespell, tox, GitHub Actions. Both model classes are `@beartype`-decorated, so argument types in tests must match annotations exactly.

**Branch and workflow:** worktree `/Users/josuemartinez/Documents/PersonalProjects/Equine/equine-stack1`, venv `.venv` (run `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 2`; ~20 s; if import fails prefix `PYTHONPATH=src`). Lint `uvx ruff check --fix src tests && uvx ruff format src tests`; `uvx codespell src tests`. Layers and branches:

| Layer | Branch | Base |
|---|---|---|
| PR-1a | `stack1/pr-1a-test-hygiene` | `feat/persistence-as-code` |
| PR-1b | `stack1/pr-1b-hypothesis-data` | PR-1a |
| PR-1c | `stack1/pr-1c-golden-fixtures` | PR-1b |
| PR-1d | `stack1/pr-1d-coverage` | PR-1c |
| PR-1e | `stack1/pr-1e-ci-tooling` | PR-1d |

Start each layer with `git checkout -b <branch>` from the previous layer's tip. Do not push to upstream. Do not edit `README.md`. `pyproject.toml` may be edited only in the sections named by a task.

---

## File structure

| File | Responsibility in this stack |
|---|---|
| `tests/conftest.py` | fixtures and strategies: temp-dir save/load helper, seeded wider `random_dataset` returning training kwargs |
| `tests/test_equine_protonet.py`, `tests/test_equine_gp.py` | existing behaviour tests; gain real assertions and use the strategy's kwargs |
| `tests/test_golden.py` (new) | seeded, non-hypothesis golden values for both models; OOD ordering; validation-metrics xfail |
| `tests/test_fixtures.py` (new) + `tests/fixtures/*.eq` + `tests/fixtures/expected.json` | cross-version fixture files and pinned outputs |
| `tests/test_utils.py` | known-value metric tests; metrics-arg-order xfail; raise-path tests |
| `tests/test_equine.py`, `tests/test_equine_integration.py` | raise-path tests for coverage honesty |
| `pyproject.toml` | coverage exclusions (PR-1d); tox `extras`, `deps`, pyright env description/deps (PR-1e) |
| `.github/workflows/Tests.yml`, `.github/workflows/gh-pages.yml`, `.pre-commit-config.yaml` | CI drift fixes (PR-1e) |

---

## Layer PR-1a — Test hygiene (#224, #220, #221)

Branch: `stack1/pr-1a-test-hygiene` (already created from `feat/persistence-as-code`).

### Task 1: save/load helper writes to a temporary directory

**Files:** Modify `tests/conftest.py` (`use_save_load_model_tests`), `tests/test_equine_protonet.py`, `tests/test_equine_gp.py` (callers).

- [ ] **Step 1: Write the failing test** (append to `tests/test_equine_integration.py`):

```python
def test_suite_leaves_no_model_files_in_the_repo_root() -> None:
    """Regression guard for #224: tests must not write .eq files into the cwd."""
    import glob
    import os

    stray = [f for f in glob.glob("*.eq") if os.path.isfile(f)]
    assert stray == [], f"tests wrote model files into the working directory: {stray}"
```

This passes today only if run first; the point is the helper change below, after which it passes in any order.

- [ ] **Step 2: Replace the helper in `tests/conftest.py`:**

```python
def use_save_load_model_tests(model, X, tmp_filename: str = "tmp.eq"):
    """Save, reload through load_equine_model, and assert predictions are unchanged.

    Writes into a temporary directory that is removed on return. Not a pytest
    fixture on purpose: hypothesis' function_scoped_fixture health check rejects
    ``tmp_path`` inside ``@given`` tests.
    """
    old_output = model.predict(X[1:10])
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, tmp_filename)
        model.save(path)
        new_model = eq.load_equine_model(path)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Class predictions changed on reload"
    assert (
        torch.nn.functional.mse_loss(old_output.ood_scores, new_output.ood_scores)
        <= 1e-7
    ), "OOD predictions changed on reload"
    return new_model
```

Add `import tempfile` to the imports. The helper now returns only `new_model`.

- [ ] **Step 3: Update every caller.** In `tests/test_equine_protonet.py` and `tests/test_equine_gp.py`, each `new_model, tmp_filename = use_save_load_model_tests(...)` (or `_, tmp_filename = ...`) becomes `new_model = use_save_load_model_tests(...)`, and the trailing `if os.path.exists(tmp_filename): os.remove(tmp_filename)  # Cleanup` blocks are deleted. Remove `import os` from those files if it becomes unused (ruff will say).

- [ ] **Step 4: Find any other test that writes to the cwd:** `grep -rn "\.save(\"\|\.save('\|torch.save(" tests/` and `grep -rn '"untrained' tests/`. Any path that is not under `tmp_path`, `tmp_path_factory` or a `TemporaryDirectory` is changed to use one (non-hypothesis tests may use the `tmp_path` fixture).

- [ ] **Step 5: Run the suite twice, then check the tree:**

Run: `.venv/bin/python -m pytest tests -q -p no:cacheprovider -n 2 && .venv/bin/python -m pytest tests -q -p no:cacheprovider -n 2 && git status --porcelain`
Expected: all pass both times; `git status --porcelain` prints nothing (no stray `.eq` files).

- [ ] **Step 6: Commit**

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add tests
git commit -m "test: save/load helper writes to a temporary directory (#224)"
```

### Task 2: real assertions on the assertion-free tests (#220)

**Files:** Modify `tests/test_equine_protonet.py`, `tests/test_equine_gp.py`.

- [ ] **Step 1: Add a shared assertion helper to `tests/conftest.py`:**

```python
def assert_valid_prediction(out, num_rows: int, num_classes: int) -> None:
    """Shape and value checks on an EquineOutput from predict()."""
    assert out.classes.shape == (num_rows, num_classes)
    assert out.ood_scores.shape == (num_rows,)
    assert torch.isfinite(out.classes).all()
    assert torch.isfinite(out.ood_scores).all()
    assert torch.all(out.classes >= 0) and torch.all(out.classes <= 1)
    assert torch.allclose(out.classes.sum(dim=1), torch.ones(num_rows), atol=1e-5)
    assert torch.all(out.ood_scores >= 0) and torch.all(out.ood_scores <= 1)
```

Range and sum-to-one apply to `predict().classes` only. Do NOT add value assertions on `forward()` output (PR-4a changes `forward` to return logits); `forward` gets shape checks only.

- [ ] **Step 2: Use it in every test that calls `predict` without checking values.** In `tests/test_equine_gp.py`: `test_equine_gp_train_from_scratch`, `..._with_temperature`, `..._with_scheduler`, `..._with_validation` end with `model.predict(X[1:10])  # Contracts should fire asserts on errors`; replace with `assert_valid_prediction(model.predict(X[1:10]), 9, num_classes)`. In `tests/test_equine_protonet.py`: after each `eq_out = model.predict(X)` add `assert_valid_prediction(eq_out, len(X), num_classes)`; `test_compute_embeddings` asserts the embedding shape `(rows, num_classes)` (shape only). Import the helper from `conftest` like the other helpers.

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/test_equine_protonet.py tests/test_equine_gp.py -q -p no:cacheprovider`
Expected: all pass.

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add tests
git commit -m "test: assert prediction shapes and probability ranges instead of only calling predict (#220)"
```

### Task 3: temperature tests assert the temperature moved (#221)

**Files:** Modify `tests/test_equine_protonet.py::test_train_episodes_with_temperature`, `tests/test_equine_gp.py::test_equine_gp_train_from_scratch_with_temperature` and `test_equine_gp_save_load_with_temperature`.

- [ ] **Step 1: Add assertions.** Protonet: record `before = model.temperature.item()` immediately after construction (it equals `init_temperature`, default 1.0); after `train_model(...)` (which calibrates when `use_temperature=True`) and after the explicit `calibrate_temperature(...)` call, assert `model.temperature.item() != before` and `model.temperature.item() > 0`. GP: after `calibrate_model(...)` / the temperature training path, assert the same. If a test reaches this assertion with the temperature unchanged because calibration runs zero steps for that data shape, increase `num_calibration_epochs` in that test rather than weakening the assertion.

- [ ] **Step 2: Run, lint, commit**

```bash
.venv/bin/python -m pytest tests/test_equine_protonet.py tests/test_equine_gp.py -q -p no:cacheprovider -k temperature
uvx ruff check --fix tests && uvx ruff format tests
git add tests
git commit -m "test: temperature calibration tests assert the temperature changed (#221)"
```

Layer exit: full suite green twice, clean tree. Push: `git push -u fork stack1/pr-1a-test-hygiene`.

---

## Layer PR-1b — Deterministic, wider hypothesis data (#197, #198)

Branch: `git checkout -b stack1/pr-1b-hypothesis-data` from PR-1a's tip.

### Task 4: seeded, wider `random_dataset` that returns its training arguments

**Files:** Modify `tests/conftest.py` (`random_dataset`, `use_basic_embedding_model`), every test using `random_dataset` in `tests/test_equine_protonet.py`, `tests/test_equine_gp.py`, `tests/test_equine_integration.py`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_equine_integration.py`):

```python
from hypothesis import given, settings

from conftest import random_dataset


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=20)
def test_random_dataset_shape_and_training_args(random_dataset) -> None:
    dataset, num_classes, train_kwargs = random_dataset
    X, Y = dataset.tensors
    rows = X.shape[0]
    assert 120 <= rows <= 200
    assert 2 <= num_classes <= 5
    labels, counts = torch.unique(Y, return_counts=True)
    assert len(labels) == num_classes
    assert counts.min() >= 30
    assert train_kwargs["way"] == min(3, num_classes)
    per_class_train = int(rows * (1 - train_kwargs["calib_frac"]) // num_classes)
    assert 1 <= train_kwargs["support_size"] < per_class_train
    assert train_kwargs["episode_size"] >= train_kwargs["way"]
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest tests/test_equine_integration.py -q -p no:cacheprovider -k random_dataset_shape`
Expected: FAIL (`random_dataset` returns `(dataset, num_classes, way)` today and rows are always 100).

- [ ] **Step 3: Replace the strategy in `tests/conftest.py`:**

```python
@st.composite
def random_dataset(draw):
    """A labelled dataset plus the training arguments that fit its shape.

    Returns ``(dataset, num_classes, train_kwargs)``. ``train_kwargs`` is passed
    to ``EquineProtonet.train_model`` (GP tests take what they need from it):
    the defaults (way=3, support_size=25, episode_size=100) only fit a 100-row,
    3-class dataset, and otherwise generate_episode raises. Torch is seeded from
    a drawn integer so hypothesis can replay and shrink failing examples.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    torch.manual_seed(seed)
    num_classes = draw(st.integers(min_value=2, max_value=5))
    # >= 30 rows per class and 120 <= rows <= 200 for every num_classes in 2..5
    rows_per_class = draw(
        st.integers(
            min_value=max(30, math.ceil(120 / num_classes)),
            max_value=200 // num_classes,
        )
    )
    rows = rows_per_class * num_classes
    cols = draw(st.integers(min_value=1, max_value=64))
    dataset_x = torch.rand(rows, cols)
    dataset_y = torch.arange(num_classes).repeat_interleave(rows_per_class).float()
    dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)  # type: ignore

    calib_frac = 0.2
    per_class_train = int(rows * (1 - calib_frac) // num_classes)
    way = min(3, num_classes)
    support_size = min(10, per_class_train - 5)
    episode_size = min(50, way * (per_class_train - support_size))
    train_kwargs = {
        "calib_frac": calib_frac,
        "way": way,
        "support_size": support_size,
        "episode_size": episode_size,
    }
    return dataset, num_classes, train_kwargs


def use_basic_embedding_model(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)
    return dataset, num_classes, X, embedding_model, train_kwargs
```

`cols` is capped at 64 (was 1000) so the wider strategy does not slow the suite; nothing under test depends on width beyond the embedding's input layer. Add `import math` to conftest. `stratified_train_test_split` holds out `round(count * test_size)` rows per class, so with balanced classes each class keeps at least `floor(0.8 * rows_per_class)` training rows; `per_class_train` is therefore a lower bound and `support_size = min(10, per_class_train - 5)` never trips `generate_episode`'s "Not enough support examples" check.

- [ ] **Step 4: Update every consumer.** Grep: `grep -n "random_dataset\|use_basic_embedding_model\|train_model(" tests/test_equine_protonet.py tests/test_equine_gp.py tests/test_equine_integration.py`. Rules:
  - Unpack `dataset, num_classes, X, embedding_model, train_kwargs = use_basic_embedding_model(random_dataset)` (was four values); where `random_dataset` is unpacked directly as `dataset, num_classes, way = random_dataset`, use `dataset, num_classes, train_kwargs = random_dataset` and `way = train_kwargs["way"]`.
  - Every `EquineProtonet.train_model(dataset, num_episodes=N, ...)` call passes `**train_kwargs` (drop any explicit `calib_frac`/`support_size`/`way`/`episode_size` argument that would conflict, or override deliberately after `**train_kwargs`).
  - GP `train_model(..., vis_support=True, support_size=...)` uses `support_size=train_kwargs["support_size"]`.
  - Tests that assert `len(model.model.support) == num_classes` still hold. Tests that index `support[0]`-style still hold because labels start at 0.
  - `test_train_episodes_with_temperature` and any test calling `calibrate_temperature(calib_x, calib_y, ...)` keep working (calibration set is 20 % of rows, at least 24 rows).
  - `tests/test_utils.py` has its own `support_dataset` strategy; leave it.

- [ ] **Step 5: Run the affected files with many examples once to shake out shape bugs**

Run: `.venv/bin/python -m pytest tests/test_equine_protonet.py tests/test_equine_gp.py tests/test_equine_integration.py -q -p no:cacheprovider --hypothesis-seed=0 -p no:randomly 2>&1 | tail -5`, then the whole suite.
Expected: all pass. Any failure is a shape/argument bug in THIS task (fix the kwargs derivation or the test), never an `xfail`. If a failure exposes a genuine `src/` bug (e.g. the single-example-class crash, #184), reduce the strategy's minimum so it does not hit it and note the issue number in a comment; Phase 3 fixes those.

- [ ] **Step 6: Commit**

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add tests
git commit -m "test: seeded, wider random_dataset that returns training arguments matching its shape (#197, #198)"
```

Layer exit: suite green; push `git push -u fork stack1/pr-1b-hypothesis-data`.

---

## Layer PR-1c — Golden values and cross-version fixtures (#199, #187, #186)

Branch: `git checkout -b stack1/pr-1c-golden-fixtures` from PR-1b's tip.

### Task 5: seeded golden test for both models

**Files:** Create `tests/test_golden.py`, `tests/golden_data.py` (the fixture builder used by both this test and the fixture files).

- [ ] **Step 1: Create `tests/golden_data.py`:**

```python
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Deterministic data and models for golden-value and cross-version tests.

Everything here is seeded so that the same code produces the same numbers on
CPU across runs. Golden literals in tests/test_golden.py are produced by these
functions inside golden_dtype() (float64). The fixture files in tests/fixtures/
are float32 and predate golden_dtype(); they are never regenerated.
"""

import contextlib

import torch

from conftest import BasicEmbeddingModel

import equine as eq

ROWS, FEATURES, CLASSES = 120, 6, 3
SEED = 0


@contextlib.contextmanager
def golden_dtype():
    """Run the golden path in float64: cross-platform deviation ~1e-16 instead of up to 5e-4."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def separable_dataset(seed: int = SEED):
    """Uniform noise plus a class-dependent shift of 3 on the first `CLASSES` features."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES).repeat_interleave(ROWS // CLASSES)
    x = torch.rand(ROWS, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return torch.utils.data.TensorDataset(x, y.float()), x, y


def query_batch(seed: int = SEED + 1) -> torch.Tensor:
    """Fixed in-distribution queries: one per class, same construction as the training data."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES)
    x = torch.rand(CLASSES, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return x


def far_ood_batch() -> torch.Tensor:
    """Queries shifted by +6 on every feature: far from every class."""
    return query_batch() + 6.0


def trained_protonet(seed: int = SEED):
    with golden_dtype():
        torch.manual_seed(seed)
        dataset, _, _ = separable_dataset(seed)
        model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
        model.train_model(
            dataset, num_episodes=20, calib_frac=0.2, support_size=10, way=3, episode_size=30
        )
    return model


def trained_gp(seed: int = SEED):
    with golden_dtype():
        torch.manual_seed(seed)
        dataset, _, _ = separable_dataset(seed)
        model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16)
        model.train_model(
            dataset,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=40,  # 5 epochs at lr 0.01 left the GP undertrained (class probs ~1/3)
            batch_size=32,
            vis_support=True,
            support_size=10,
        )
    return model
```

The builders run inside `golden_dtype()` (float64). The committed fixtures in `tests/fixtures/` are float32 and predate it: they were written before the builders switched dtype and are never regenerated.

- [ ] **Step 2: Generate the literals once.** Run (from the worktree):

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "tests")
import torch, json
from golden_data import golden_dtype, trained_protonet, trained_gp, query_batch
torch.set_printoptions(precision=8)
for name, make in (("protonet", trained_protonet), ("gp", trained_gp)):
    with golden_dtype():
        a = make().predict(query_batch()); b = make().predict(query_batch())
    assert torch.equal(a.classes, b.classes) and torch.equal(a.ood_scores, b.ood_scores), f"{name} not deterministic"
    print(name, "classes", a.classes.tolist()); print(name, "ood", a.ood_scores.tolist())
EOF
```

Expected: two identical runs per model (the assert proves determinism), then eight lists. If a model is NOT deterministic (assert fails), find the unseeded source (e.g. a `torch.randperm` without generator, or `scipy` randomness) and seed it in `golden_data.py` via `torch.manual_seed` before each training call; do not change `src/`.

- [ ] **Step 3: Create `tests/test_golden.py`** with the printed lists pasted in:

```python
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Golden-value tests: seeded models must keep producing these exact numbers.

A PR that changes model output must update the literals here AND explain the
change in its description. Phase 3 PRs are expected to do so; any other PR
that moves them has changed behaviour it did not mean to change.
"""

import pytest
import torch

from golden_data import far_ood_batch, golden_dtype, query_batch, trained_gp, trained_protonet

import equine as eq

PROTONET_CLASSES = [[...], [...], [...]]  # paste from the generator
PROTONET_OOD = [...]
GP_CLASSES = [[...], [...], [...]]
GP_OOD = [...]
GOLDEN_ATOL = 1e-6  # see comment in the test; golden path runs in float64


@pytest.mark.parametrize(
    "make, expected_classes, expected_ood",
    [
        pytest.param(trained_protonet, PROTONET_CLASSES, PROTONET_OOD, id="protonet"),
        pytest.param(trained_gp, GP_CLASSES, GP_OOD, id="gp"),
    ],
)
def test_predictions_match_golden_values(make, expected_classes, expected_ood) -> None:
    with golden_dtype():
        out = make().predict(query_batch())
    # Golden models are built and queried in float64 (golden_data.golden_dtype()): in float32,
    # training drifts across CPU BLAS backends (macOS arm64 vs Linux x86_64: Protonet OOD 5.0e-4,
    # GP classes 5.0e-5); in float64 the cross-platform deviation is ~1e-16, so GOLDEN_ATOL = 1e-6
    # holds. Fixtures stay float32; load-then-predict agrees within 1e-5 across platforms.
    assert out.classes.argmax(dim=1).tolist() == [0, 1, 2]
    assert torch.allclose(out.classes, torch.tensor(expected_classes), atol=GOLDEN_ATOL)
    assert torch.allclose(out.ood_scores, torch.tensor(expected_ood), atol=GOLDEN_ATOL)


@pytest.mark.parametrize("make", [trained_protonet, trained_gp], ids=["protonet", "gp"])
def test_far_out_of_distribution_queries_score_higher(make) -> None:
    """#187: OOD scores must separate far-OOD queries from in-distribution ones."""
    model = make()
    in_dist = model.predict(query_batch()).ood_scores
    far = model.predict(far_ood_batch()).ood_scores
    assert torch.all(in_dist >= 0) and torch.all(in_dist <= 1)
    assert torch.all(far >= 0) and torch.all(far <= 1)
    assert far.mean() > in_dist.mean()


def test_protonet_ood_scores_lie_in_unit_interval_on_training_data() -> None:
    model = trained_protonet()
    from golden_data import separable_dataset

    _, x, _ = separable_dataset()
    ood = model.predict(x).ood_scores
    assert torch.all(ood >= 0) and torch.all(ood <= 1)
```

- [ ] **Step 4: Run twice; commit**

Run: `.venv/bin/python -m pytest tests/test_golden.py -q -p no:cacheprovider && .venv/bin/python -m pytest tests/test_golden.py -q -p no:cacheprovider -n 2`
Expected: all pass both times (also under xdist). If `test_far_out_of_distribution_queries_score_higher` fails for a model, that is a real finding: report it as DONE_WITH_CONCERNS with the numbers rather than weakening the assertion (the GP's entropy-based score is the known weak case, #175; if it fails, mark only the GP case `xfail(strict=True, reason="#175: entropy OOD score does not separate far-OOD queries")`).

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add tests/golden_data.py tests/test_golden.py
git commit -m "test: seeded golden values and OOD ordering for both models (#187)"
```

### Task 6: cross-version fixture files

**Files:** Create `tests/fixtures/protonet_v2.eq`, `tests/fixtures/gp_v2.eq`, `tests/fixtures/expected.json` (predictions plus the SHA-256 of each `.eq`, asserted before loading so a regenerated file cannot pass silently), `tests/test_fixtures.py`; modify `.gitattributes` (create if missing) to mark `*.eq binary`.

- [ ] **Step 1: Generate the fixtures with the CURRENT code:**

```bash
mkdir -p tests/fixtures
.venv/bin/python - <<'EOF'
import hashlib, sys, json; sys.path.insert(0, "tests")
import torch
from golden_data import trained_protonet, trained_gp, query_batch
expected = {}
for name, make in (("protonet", trained_protonet), ("gp", trained_gp)):
    model = make(); path = f"tests/fixtures/{name}_v2.eq"; model.save(path)
    out = model.predict(query_batch())
    expected[name] = {"classes": out.classes.tolist(), "ood_scores": out.ood_scores.tolist()}
    expected[name]["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump(expected, open("tests/fixtures/expected.json", "w"), indent=1)
print({k: (len(v["classes"]), len(v["ood_scores"])) for k, v in expected.items()})
EOF
printf '*.eq binary\n' >> .gitattributes
ls -la tests/fixtures
```

Expected: two `.eq` files (tens of KB each) and `expected.json`.

- [ ] **Step 2: Create `tests/test_fixtures.py`:**

```python
# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Cross-version compatibility: files saved by an earlier code version keep
loading and keep producing the same predictions.

tests/fixtures/*_v2.eq were written by the code at the commit that added them
(format version 2). Do NOT regenerate them when outputs change; a later PR
that cannot keep this test passing has broken compatibility and must gate its
change behind a persisted setting with a legacy default (roadmap rule 3).
"""

import hashlib
import json
import os

import pytest
import torch

from golden_data import query_batch

import equine as eq

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
EXPECTED = json.load(open(os.path.join(FIXTURES, "expected.json")))


@pytest.mark.parametrize("name, cls", [("protonet", eq.EquineProtonet), ("gp", eq.EquineGP)])
def test_v2_fixture_loads_and_predicts_the_same(name, cls) -> None:
    path = os.path.join(FIXTURES, f"{name}_v2.eq")
    with open(path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED[name]["sha256"]
    model = eq.load_equine_model(path)
    assert isinstance(model, cls)
    out = model.predict(query_batch())
    # 1e-5: the GP load path recomputes the covariance (cholesky); measured x86 drift 4.8e-7.
    assert torch.allclose(out.classes, torch.tensor(EXPECTED[name]["classes"]), atol=1e-5)
    assert torch.allclose(out.ood_scores, torch.tensor(EXPECTED[name]["ood_scores"]), atol=1e-5)
```

- [ ] **Step 3: Run; commit (fixtures included)**

Run: `.venv/bin/python -m pytest tests/test_fixtures.py -q -p no:cacheprovider`
Expected: 2 passed.

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add .gitattributes tests/fixtures tests/test_fixtures.py
git commit -m "test: commit v2 model fixtures with pinned predictions as the cross-version guard"
```

### Task 7: known-value metric tests and the two strict xfails (#199, #174, #186, #171)

**Files:** Modify `tests/test_utils.py`, `tests/test_equine_gp.py`.

- [ ] **Step 1: Read `brier_skill_score` and `generate_model_metrics` in `src/equine/utils.py`** to confirm the formulas (`brier_skill_score` compares against a reference forecast; write the expected value from its formula, not from running it).

- [ ] **Step 2: Append to `tests/test_utils.py`:**

```python
def _probs(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_brier_score_known_values() -> None:
    # perfect one-hot predictions -> 0
    y = torch.tensor([0, 1, 2])
    assert eq.utils.brier_score(_probs([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), y) == pytest.approx(0.0)
    # uniform predictions over K classes -> (K-1)/K
    k = 3
    uniform = _probs([[1 / k] * k] * 3)
    assert eq.utils.brier_score(uniform, y) == pytest.approx((k - 1) / k, abs=1e-6)
    # confidently wrong -> 2
    assert eq.utils.brier_score(_probs([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), y) == pytest.approx(2.0)


def test_brier_skill_score_known_values() -> None:
    y = torch.tensor([0, 1, 2])
    perfect = _probs([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # write the expected value from the formula in utils.brier_skill_score's docstring
    assert eq.utils.brier_skill_score(perfect, y) == pytest.approx(1.0)


def test_expected_calibration_error_known_values() -> None:
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # fully confident and always right -> 0
    right = torch.nn.functional.one_hot(y, 2).float()
    assert eq.utils.expected_calibration_error(right, y) == pytest.approx(0.0, abs=1e-6)
    # fully confident and always wrong -> 1
    wrong = torch.nn.functional.one_hot(1 - y, 2).float()
    assert eq.utils.expected_calibration_error(wrong, y) == pytest.approx(1.0, abs=1e-6)
    # confidence 0.6 on ten samples of which six are right -> perfectly calibrated bin -> 0
    calibrated = _probs([[0.6, 0.4]] * 10)
    y_cal = torch.tensor([0] * 6 + [1] * 4)
    assert eq.utils.expected_calibration_error(calibrated, y_cal) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="#174: generate_model_metrics passes (target, preds) to torchmetrics, which expects (preds, target)",
)
def test_generate_model_metrics_known_confusion_matrix() -> None:
    y_true = torch.tensor([0, 0, 0, 0, 1, 2])
    probs = torch.nn.functional.one_hot(torch.tensor([0, 0, 0, 1, 1, 1]), 3).float()
    out = eq.EquineOutput(classes=probs, ood_scores=torch.zeros(6), embeddings=probs)
    metrics = eq.utils.generate_model_metrics(out, y_true)
    # macro accuracy: class 0 -> 3/4, class 1 -> 1/1, class 2 -> 0/1 => mean 0.5833
    assert metrics["accuracy"] == pytest.approx(0.58333, abs=1e-4)
    # rows = true class, columns = predicted class
    assert metrics["confusionMatrix"] == [[3, 1, 0], [0, 1, 0], [0, 1, 0]]
```

Check the exact key name for the confusion matrix in `generate_model_metrics` (`confusionMatrix`) and adjust the assertion to its type (list vs tensor). The metrics test MUST fail today with the swapped-argument values; if it unexpectedly passes, the `xfail(strict=True)` will error and you must re-check the expected numbers rather than remove the mark.

- [ ] **Step 3: Validation-metrics xfail in `tests/test_equine_gp.py`.** Rewrite `test_equine_gp_train_from_scratch_with_validation` to keep the returned dict and assert per-epoch state, marked strict xfail against #171:

```python
@pytest.mark.xfail(
    strict=True,
    reason="#171: EquineGP.train_model never resets val_metrics between epochs",
)
def test_validation_metrics_are_reset_between_epochs() -> None:
    from golden_data import separable_dataset

    dataset, x, y = separable_dataset()
    val = torch.utils.data.TensorDataset(x[:64], y[:64].long())
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3)
    model = eq.EquineGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        num_epochs=3,
        batch_size=32,
        validation_dataset=val,
        val_metrics=[metric],
    )
    # two validation batches of 32 per epoch; state must be reset each epoch
    assert metric._update_count == 2
```

Keep the existing hypothesis-driven `..._with_validation` test as is (it exercises the API), but make it assert `assert_valid_prediction(...)` as in Task 2.

- [ ] **Step 4: Run; commit**

Run: `.venv/bin/python -m pytest tests/test_utils.py tests/test_equine_gp.py -q -p no:cacheprovider`
Expected: all pass except the two `xfailed`.

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add tests
git commit -m "test: known-value metric tests; strict xfails for #174 and #171 (#199, #186)"
```

Layer exit: suite green with exactly two xfails; push `git push -u fork stack1/pr-1c-golden-fixtures`.

---

## Layer PR-1d — Coverage honesty (#195, #222)

Branch: `git checkout -b stack1/pr-1d-coverage` from PR-1c's tip.

### Task 8: remove the raise-line exclusions and cover the raises

**Files:** Modify `pyproject.toml` (`[tool.coverage.report] exclude_lines` only), `tests/test_utils.py`, `tests/test_equine.py`, `tests/test_equine_integration.py`, `tests/test_equine_protonet.py`, `tests/test_equine_gp.py`.

- [ ] **Step 1: Remove the two exclusions** `'raise NotImplementedError'` and `'raise ValueError'` from `exclude_lines` in `pyproject.toml`. Leave the others.

- [ ] **Step 2: Measure what is now uncovered:**

Run: `NUMBA_DISABLE_JIT=1 .venv/bin/python -m pytest --cov-report=term-missing:skip-covered --cov-config=pyproject.toml --cov=equine tests -q -p no:cacheprovider -n 2 2>&1 | grep -E "^(src/|TOTAL)"`
Expected: a list of uncovered lines per file; note every `raise` line among them.

- [ ] **Step 3: Add tests for each uncovered raise** (append to the matching test file; each is a plain, non-hypothesis test):

```python
def test_get_prototypes_on_base_class_raises() -> None:
    with pytest.raises(NotImplementedError):
        eq.Equine.get_prototypes(None)  # type: ignore[arg-type]


def test_generate_support_refuses_support_size_larger_than_a_class() -> None:
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Not enough support examples"):
        eq.generate_support(x, y, support_size=4, selected_labels=[0, 1])


def test_generate_episode_refuses_support_size_larger_than_a_class() -> None:
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Not enough support examples"):
        eq.generate_episode(x, y, support_size=4, way=2, episode_size=2)


def test_load_equine_model_unknown_model_type(tmp_path) -> None:
    path = tmp_path / "weird.eq"
    torch.save({"equine_format_version": 2, "train_summary": {"modelType": "Nope"}}, path)
    with pytest.raises(ValueError, match="Unknown model type"):
        eq.load_equine_model(str(path))
```

For `validate_feature_label_names` bad-length paths and `update_support(label_names=...)`, read the functions (`src/equine/equine.py`, `equine_protonet.py`, `equine_gp.py`) and write one test per `raise` with `pytest.raises(ValueError, match=<a distinctive fragment of the message>)`. Adjust the `generate_*` signatures to the real ones in `utils.py` (check parameter names before writing).

- [ ] **Step 4: Re-measure and set the gate honestly**

Run the coverage command again with `--cov-fail-under=97`.
Expected: passes. If total coverage is below 97 % after the exclusions are removed and the new tests added, lower `--cov-fail-under` in the tox `coverage` env command to the true value rounded down to a whole percent, and add a `# TODO(#195): raise back to 97 in PR-3b` comment next to it. Never re-add an exclusion.

- [ ] **Step 5: Commit**

```bash
uvx ruff check --fix tests && uvx ruff format tests
git add pyproject.toml tests
git commit -m "test: stop excluding raise lines from coverage; cover the error paths (#195, #222)"
```

Layer exit: coverage gate honest and passing; push `git push -u fork stack1/pr-1d-coverage`.

---

## Layer PR-1e — CI and tooling drift (#223, #225, #226, #228, #229, #233)

Branch: `git checkout -b stack1/pr-1e-ci-tooling` from PR-1d's tip.

### Task 9: tox installs the `tests` extra; pre-commit and tox agree on ruff; codespell in pre-commit

**Files:** Modify `pyproject.toml` (tox `[testenv]`, `[testenv:coverage]`, `[testenv:pyright]`, `[testenv:format]`, `[testenv:enforce-format]` blocks only), `.pre-commit-config.yaml`.

- [ ] **Step 1: tox `[testenv]`:** add `extras = tests` and reduce `deps` to what the extra lacks: `pytest-xdist`, `tzdata`. Remove `pytest`, `pytest-cov`, `hypothesis`, `numpy`, `torch` from `deps` (they come from the package's dependencies and the `tests` extra; the hypothesis pin in `pyproject.toml` is now what CI runs). `[testenv:coverage]`: `extras = tests`, `deps = {[testenv]deps}` plus `coverage[toml]`; drop its duplicated numpy/torch lines. `[testenv:pyright]`: replace the stale `numpy<2.0.0 ; darwin`/`numpy`/`torch>=2.0.0` deps with just `pyright` (the package's own dependencies install torch >= 2.6), and change the description to say it scans `src/` (`tests/` is not scanned; PR-5c decides whether to add it). Pin ruff in `[testenv:format]` and `[testenv:enforce-format]` to the same version the pre-commit hook uses (next step).

- [ ] **Step 2: `.pre-commit-config.yaml`:** set the `ruff-pre-commit` `rev` to the current ruff release (0.16.9 on 2026-09-29; check `uvx ruff --version` and that the mirror has the tag: `git ls-remote --tags https://github.com/astral-sh/ruff-pre-commit`) and pin tox's `ruff==<version>` to match; add the codespell hook:

```yaml
-   repo: https://github.com/codespell-project/codespell
    rev: v2.4.3
    hooks:
    -   id: codespell
        args: [src/, docs/]
        pass_filenames: false
        additional_dependencies: [tomli]
```

(Use the latest codespell tag. `pass_filenames: false` makes the hook check the same paths tox does regardless of what is staged. `tomli` is required for codespell to read `[tool.codespell]` from `pyproject.toml` on Python < 3.11; without it the hook reports the ~20 `docs/` hits that the config skips, diverging from tox.) Also add `files: ^(src|tests)/` to both ruff hooks: ruff 0.16 formats fenced code in Markdown and notebooks by default, and tox only checks `src/` and `tests/`. Add `extras =` (empty) to `[testenv:pyright]` so it does not inherit the `tests` extra from `[testenv]`.

- [ ] **Step 3: Verify tox still resolves** (tox is not installed in the venv; use uvx): `uvx --with tox-uv tox -e enforce-format` and `uvx --with tox-uv tox -e py312 -- -q -n 2` (this creates `.tox/`; add `.tox/` to `.gitignore` if not already ignored). Expected: both pass. If tox cannot run in this environment, say so in the report and verify the config with `uvx --with tox-uv tox config -e py312 | grep -E "extras|deps"`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml .gitignore
git commit -m "ci: tox installs the tests extra; pin ruff consistently; codespell in pre-commit (#223, #225, #226)"
```

### Task 10: workflow fixes

**Files:** Modify `.github/workflows/gh-pages.yml`, `.github/workflows/Tests.yml`.

- [ ] **Step 1: `gh-pages.yml` line ~31:** replace `run: echo "::set-output name=dir::$(pip cache dir)"` with `run: echo "dir=$(pip cache dir)" >> "$GITHUB_OUTPUT"`. The consumer `${{ steps.pip-cache.outputs.dir }}` is unchanged.

- [ ] **Step 2: `Tests.yml`:** delete the `pull-requests: write` line under `permissions:` (leave `contents: read`). Update the commented-out `run-pyright` block's `pip install tox tox-gh-actions` to `tox tox-uv tox-gh-actions` so it matches the live jobs when PR-5c uncomments it.

- [ ] **Step 3: Validate YAML** `python3 -c "import yaml,sys;[yaml.safe_load(open(f)) for f in sys.argv[1:]];print('yaml ok')" .github/workflows/*.yml` (install pyyaml with `uvx --from pyyaml python ...` if needed).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows
git commit -m "ci: use GITHUB_OUTPUT in the pages workflow; drop unused pull-requests write permission (#228, #229)"
```

Layer exit: push `git push -u fork stack1/pr-1e-ci-tooling`.

---

### Task 11: verification and opening Stack 1 on the fork

- [ ] **Step 1: On the top branch, full verification:**

```bash
uvx ruff check src tests && uvx ruff format --check src tests && uvx codespell src tests
NUMBA_DISABLE_JIT=1 .venv/bin/python -m pytest --cov-report=term-missing:skip-covered --cov-config=pyproject.toml --cov-fail-under=97 --cov=equine tests -q -p no:cacheprovider -n 2
git status --porcelain
```
Expected: clean lint; the gate passes (or the honestly lowered gate from Task 8); exactly two `xfailed`; clean tree.

- [ ] **Step 2: Open the five draft PRs on the fork, bottom first, each based on the previous branch:**

```bash
gh pr create -R martinez-hub/equine --draft --base feat/persistence-as-code --head stack1/pr-1a-test-hygiene --title "PR-1a: test hygiene (#224, #220, #221)" --body "Stack 1, layer 1, stacked on PR #2. No src changes."
gh pr create -R martinez-hub/equine --draft --base stack1/pr-1a-test-hygiene --head stack1/pr-1b-hypothesis-data --title "PR-1b: seeded, wider hypothesis data (#197, #198)" --body "Stack 1, layer 2. No src changes."
gh pr create -R martinez-hub/equine --draft --base stack1/pr-1b-hypothesis-data --head stack1/pr-1c-golden-fixtures --title "PR-1c: golden values and cross-version fixtures (#199, #187, #186)" --body "Stack 1, layer 3. Adds strict xfails for #174 and #171, flipped in Phase 3. No src changes."
gh pr create -R martinez-hub/equine --draft --base stack1/pr-1c-golden-fixtures --head stack1/pr-1d-coverage --title "PR-1d: coverage honesty (#195, #222)" --body "Stack 1, layer 4. Removes raise-line exclusions; covers error paths. No src changes."
gh pr create -R martinez-hub/equine --draft --base stack1/pr-1d-coverage --head stack1/pr-1e-ci-tooling --title "PR-1e: CI and tooling drift (#223, #225, #226, #228, #229, #233)" --body "Stack 1, layer 5. tox installs the tests extra; ruff pinned consistently; codespell hook; GITHUB_OUTPUT; permissions."
```

Do not open anything on `mit-ll-responsible-ai/equine`.

---

## Self-review

**Spec coverage (roadmap Phase 1):** PR-1a: Tasks 1–3 (temp dir helper, assertions on `predict` only, temperature). PR-1b: Task 4 (seeded, 120–200 rows, 2–5 classes, ≥30 per class, kwargs derived; no xfails). PR-1c: Tasks 5–7 (seeded golden test, committed fixtures with pinned outputs, metrics xfail #174, Brier/BSS/ECE known values, OOD ordering on the separable fixture, validation-metrics xfail #171). PR-1d: Task 8 (exclusions removed, listed raises covered, honest gate). PR-1e: Tasks 9–10 (extras, ruff pin, codespell, GITHUB_OUTPUT, permissions, pyright description); #227 deferred to PR-6a as the roadmap says. Task 11 opens the stack.

**Placeholders:** the golden literals and `.eq` fixtures are generated by the given commands, not hand-written; the plan says exactly how. `brier_skill_score`'s expected value is left as 1.0 for a perfect forecast with an instruction to confirm the formula.

**Type consistency:** `random_dataset` returns `(dataset, num_classes, train_kwargs)` and `use_basic_embedding_model` returns five values in Task 4; Tasks 5–7 use `golden_data` helpers with the names defined in Task 5; `assert_valid_prediction(out, num_rows, num_classes)` is defined in Task 2 and used in Tasks 2 and 7.
