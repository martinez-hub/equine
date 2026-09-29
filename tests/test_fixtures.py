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

import json
import os

import pytest
import torch
from golden_data import query_batch

import equine as eq

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(FIXTURES, "expected.json")) as f:
    EXPECTED = json.load(f)


@pytest.mark.parametrize(
    "name, cls", [("protonet", eq.EquineProtonet), ("gp", eq.EquineGP)]
)
def test_v2_fixture_loads_and_predicts_the_same(name, cls) -> None:
    path = os.path.join(FIXTURES, f"{name}_v2.eq")
    model = eq.load_equine_model(path)
    assert isinstance(model, cls)
    out = model.predict(query_batch())
    assert torch.allclose(
        out.classes, torch.tensor(EXPECTED[name]["classes"]), atol=1e-6
    )
    assert torch.allclose(
        out.ood_scores, torch.tensor(EXPECTED[name]["ood_scores"]), atol=1e-6
    )
