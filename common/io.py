import json
from dataclasses import dataclass

import numpy as np

from common.paths import ensure_parent


def to_per_head(df, dims):
    """Long-format activation DataFrame -> dataset[layer][head] = {data, label}."""
    labels = df["label"].to_numpy()
    dataset = [[None] * dims.num_heads for _ in range(dims.num_layers)]
    for l in range(dims.num_layers):
        col = f"model.layers.{l}.self_attn.o_proj"
        mat = np.stack(df[col].to_numpy()).reshape(len(df), dims.num_heads, dims.head_dim)
        for h in range(dims.num_heads):
            dataset[l][h] = {"data": np.ascontiguousarray(mat[:, h, :]), "label": labels}
    return dataset


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path, obj):
    path = ensure_parent(path)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


@dataclass
class Split:
    """`development` = train + val; probes fit on train and rank on val."""
    train: list
    val: list
    development: list
    test: list

    def to_dict(self):
        return {"train": list(self.train), "val": list(self.val), "development": list(self.development), "test": list(self.test)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["train"], d["val"], d["development"], d["test"])

    def summary(self):
        return f"train: {len(self.train)}, val: {len(self.val)}, development: {len(self.development)}, test: {len(self.test)}"
