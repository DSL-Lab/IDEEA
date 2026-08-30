import numpy as np
import pandas as pd
import torch

from common.config import NEGATIVE, POSITIVE, resolve_dims
from common.io import load_json
from common.paths import activations_path, directions_path
from methods.base import FitMethod, HookedMethod, Steerer


def perturb_residual(direction):
    """As in ITI, the direction moves device but keeps its float64: the addition promotes
    and rounds once on store, where casting first would round the direction."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1:] += direction.to(hidden.device)
        return output
    return hook


class CAAFit(FitMethod):
    name = "caa"

    def path_segments(self):
        return ("caa",)

    def fit(self, task, model_name):
        dims = resolve_dims(model_name)
        df = pd.read_pickle(activations_path(task.name, model_name, "residual", task.character))
        dev_df = task.dev_rows(df)
        labels = dev_df["label"].to_numpy()
        n_pos = int((labels == POSITIVE).sum())
        n_neg = int((labels == NEGATIVE).sum())
        print(f"Dev rows: {len(dev_df)} ({n_pos} pos, {n_neg} neg)")

        directions = []
        for l in range(dims.num_layers):
            acts = np.stack(dev_df[f"layer_{l}"].to_numpy())
            direction = acts[labels == POSITIVE].mean(axis=0) - acts[labels == NEGATIVE].mean(axis=0)
            directions.append(direction.tolist())
            print(f"  layer {l}: norm={np.linalg.norm(direction):.4f}")

        return self._save_directions(task, model_name, {"direction": directions})


class CAA(HookedMethod):
    name = "caa"

    def __init__(self, caa_layer, alpha):
        if caa_layer is None:
            raise ValueError("--caa_layer is required for CAA steering")
        self.caa_layer = caa_layer
        self.alpha = alpha

    def path_segments(self):
        return ("caa",)

    def run_id(self):
        return f"layer_{self.caa_layer}_alpha_{self.alpha}"

    def install(self, model, task, model_name):
        d = load_json(directions_path(task.name, model_name, self.path_segments(), task.character))
        directions = np.array(d["direction"])
        bias = torch.tensor(directions[self.caa_layer] * self.alpha)

        target = model.get_submodule(f"model.layers.{self.caa_layer}")
        handle = target.register_forward_hook(perturb_residual(bias))
        return Steerer([handle], info={"caa_layer": self.caa_layer, "alpha": self.alpha, "direction_norm": float(np.linalg.norm(directions[self.caa_layer]))})
