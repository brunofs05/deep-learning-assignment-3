"""Confere contagens, aninhamento dos orçamentos e ausência de vazamento entre splits."""
from pa3.data import SPLITS, load_classes, load_split
from pa3.paths import data_root

root = data_root()
print(f"data_root = {root}")
classes = load_classes()
assert len(classes) == 21, len(classes)

expected = {"budget1": 21, "budget5": 105, "budget20": 420, "val": 202, "test_public": 811}
dfs = {name: load_split(name) for name in SPLITS}
train_all = load_split("train_all")
print(f"{'train_all':12s} {len(train_all):5d}")
for name, df in dfs.items():
    print(f"{name:12s} {len(df):5d}  (esperado {expected[name]})")
    assert len(df) == expected[name], name
    missing = [p for p in df["path"] if not __import__("os").path.exists(p)]
    assert not missing, f"{name}: {len(missing)} imagens faltando"
assert len(train_all) == 6505

ids = {k: set(v["image_id"]) for k, v in dfs.items()}
assert ids["budget1"] <= ids["budget5"] <= ids["budget20"], "orçamentos não aninhados"
pool = set(train_all["image_id"])
assert ids["budget20"] <= pool, "orçamento fora do train"
for a in ("val", "test_public"):
    assert not (ids[a] & pool), f"{a} vaza para o train"
assert not (ids["val"] & ids["test_public"]), "val ∩ test_public"

for b in ("budget1", "budget5", "budget20"):
    counts = dfs[b]["label"].value_counts()
    assert counts.nunique() == 1 and len(counts) == 21, f"{b} desbalanceado: {counts.to_dict()}"
print("\nval por classe:\n", dfs["val"]["label"].value_counts().to_string())
print("\ntest_public por classe:\n", dfs["test_public"]["label"].value_counts().to_string())
print("\nOK: contagens, aninhamento e disjunção conferem.")
