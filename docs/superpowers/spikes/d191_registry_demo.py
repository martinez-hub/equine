# D-191 spike demo: what changes for a user when the embedding model is stored
# as a recipe instead of a TorchScript archive. Run with the repo installed:
#     python docs/superpowers/spikes/d191_registry_demo.py
import os
import tempfile
import textwrap

import torch

import equine as eq

torch.manual_seed(0)
X = torch.rand(120, 6)
Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
dataset = torch.utils.data.TensorDataset(X, Y)
tmp = tempfile.mkdtemp()


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def describe(path):
    ckpt = torch.load(path, weights_only=True)  # the only call a loader needs
    print(f"file: {os.path.getsize(path):,} bytes | weights_only=True load: OK")
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            desc = f"tensor {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
        elif (
            isinstance(v, dict)
            and v
            and all(isinstance(t, torch.Tensor) for t in v.values())
        ):
            desc = f"dict of {len(v)} tensors"
        else:
            desc = repr(v)[:70]
        print(f"  {k:24s} {desc}")


section("BEFORE (main today): user code")
print(
    textwrap.dedent("""
    class EmbeddingModel(torch.nn.Module):          # defined in the notebook
        def __init__(self): ...
    model = eq.EquineProtonet(EmbeddingModel(), emb_out_dim=3)
    model.train_model(dataset, ...)
    model.save("model.eq")        # embeds a TorchScript copy of EmbeddingModel
    eq.load_equine_model("model.eq")   # torch.jit.load runs the embedded code
""")
)

section("AFTER (spike): user code, one decorator added")
print(
    textwrap.dedent("""
    @eq.embedding_architecture("mynotebook.embedding")   # <-- the one change
    class EmbeddingModel(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int): ...

    model = eq.EquineProtonet(EmbeddingModel(6, 3), emb_out_dim=3)
    model.train_model(dataset, ...)
    model.save("model.eq")        # stores {"builder": "mynotebook.embedding",
                                  #         "kwargs": {"in_features": 6, "out_features": 3}}
                                  # plus a state_dict. No executable content.
    eq.load_equine_model("model.eq")   # rebuilds EmbeddingModel(6, 3), loads weights
""")
)


@eq.embedding_architecture("mynotebook.embedding")
class EmbeddingModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.net(x)


model = eq.EquineProtonet(EmbeddingModel(6, 3), 3)
model.train_model(
    dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
)
p1 = os.path.join(tmp, "notebook_model.eq")
model.save(p1)
section("AFTER: what the file contains")
describe(p1)
re1 = eq.load_equine_model(p1)
print(
    "reloaded embedding type:",
    type(re1.embedding_model).__name__,
    "| predictions equal:",
    torch.allclose(model.predict(X[:8]).classes, re1.predict(X[:8]).classes),
)

section("Shipped architecture: no class to define at all")
print(
    textwrap.dedent("""
    model = eq.EquineProtonet(eq.MLP(6, [32], 3), emb_out_dim=3)
    # recipe: {"builder": "equine.mlp", "kwargs": {"in_features": 6, "hidden_sizes": [32],
    #                                             "out_features": 3, "activation": "relu"}}
""")
)
m2 = eq.EquineProtonet(eq.MLP(6, [32], 3), 3)
m2.train_model(
    dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
)
p2 = os.path.join(tmp, "mlp_model.eq")
m2.save(p2)
print("recipe stored:", torch.load(p2, weights_only=True)["embedding_recipe"])

section("Web app path today: a TorchScript module as the embedding model")
scripted = torch.jit.script(EmbeddingModel(6, 3))  # what torch.jit.load(upload) yields
m3 = eq.EquineProtonet(scripted, 3)
m3.train_model(
    dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
)
p3 = os.path.join(tmp, "webapp_model.eq")
try:
    m3.save(p3)
except ValueError as e:
    print("save() without opt-in ->", str(e)[:160], "...")
m3.save(p3, allow_executable=True)
print("\nsave(path, allow_executable=True) -> file written; contents:")
describe(p3)
try:
    eq.load_equine_model(p3)
except ValueError as e:
    print("\nload without trust ->", str(e)[:150], "...")
re3 = eq.load_equine_model(p3, trust_executable=True)
print("load(trust_executable=True) -> OK, type:", type(re3.embedding_model).__name__)

section("Opening a recipe file where the class is unknown")
print("(simulated: recipe names 'someone_else.encoder', not registered here)")
try:
    eq.registry.build_from_recipe({"builder": "someone_else.encoder", "kwargs": {}})
except ValueError as e:
    print(str(e))

section("Summary")
print(
    textwrap.dedent(f"""
    registered here: {eq.registered_architectures()}
    - default file: data only, readable with torch.load(weights_only=True), no trust flag needed
    - unregistered class: save() refuses unless allow_executable=True; file flagged;
      load needs trust_executable=True (the web app's current path, made explicit)
    - unknown recipe on load: actionable error, or pass embedding_model=YourClass(...)
    - not done in this spike: EquineGP, GP/Protonet symmetry, notebooks, docs
""")
)
