import os

import matplotlib
matplotlib.use("Agg")   # headless: must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from common.config import NEGATIVE, POSITIVE, resolve_dims
from common.hooks import iter_o_proj, layer_index
from common.io import load_json, to_per_head
from common.paths import activations_path, directions_path
from methods.base import FitMethod, HookedMethod, Steerer, top_k_mask


def get_probe_acc(dataset, idx_train, idx_test):
    layers = len(dataset)
    heads = len(dataset[0])
    all_train_acc = [[None for h in range(heads)] for l in range(layers)]
    all_test_acc = [[None for h in range(heads)] for l in range(layers)]
    for l in tqdm(range(layers), desc="Training probes"):
        for h in range(heads):
            X_train = dataset[l][h]["data"][idx_train]
            X_test = dataset[l][h]["data"][idx_test]
            y_train = dataset[l][h]["label"][idx_train]
            y_test = dataset[l][h]["label"][idx_test]
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X_train, y_train)
            all_train_acc[l][h] = (probe.predict(X_train) == y_train).sum() / len(idx_train)
            all_test_acc[l][h] = (probe.predict(X_test) == y_test).sum() / len(idx_test)
    return np.array(all_train_acc), np.array(all_test_acc)


def get_normalized_mass_mean(dataset):
    layers = len(dataset)
    heads = len(dataset[0])
    normalized_direction = [[None for h in range(heads)] for l in range(layers)]
    for l in range(layers):
        for h in range(heads):
            activations = dataset[l][h]
            mean_pos = activations["data"][activations["label"] == POSITIVE].mean(axis=0)
            mean_neg = activations["data"][activations["label"] == NEGATIVE].mean(axis=0)
            direction = (mean_pos - mean_neg).astype(np.float32)
            if np.linalg.norm(direction) == 0:
                normalized_direction[l][h] = direction
            else:
                normalized_direction[l][h] = direction / np.linalg.norm(direction)
    return normalized_direction


def get_naive_std(dataset):
    layers = len(dataset)
    heads = len(dataset[0])
    naive_std = [[None for h in range(heads)] for l in range(layers)]
    for l in range(layers):
        for h in range(heads):
            naive_norms = np.linalg.norm(dataset[l][h]["data"], axis=1)
            naive_std[l][h] = np.std(naive_norms)
    return naive_std


def rank_heads(val_acc, normalized_dir):
    """Zero-direction heads are forced to accuracy 0: steering them is a no-op."""
    adjusted = np.array(val_acc, dtype=float).copy()
    for l in range(adjusted.shape[0]):
        for h in range(adjusted.shape[1]):
            if np.linalg.norm(normalized_dir[l][h]) == 0:
                adjusted[l][h] = 0
    return np.unravel_index(np.argsort(adjusted.flatten())[::-1], adjusted.shape)


def plot_probe_acc(acc, out_dir, title):
    layers = len(acc)
    plt.figure(figsize=(7, 6))
    plt.imshow(np.sort(acc, axis=1)[::-1, ::-1], cmap="YlGnBu", vmin=0.5, vmax=1)
    plt.xticks([])
    plt.yticks(ticks=np.arange(layers), labels=np.arange(layers - 1, -1, -1))
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.colorbar()
    os.makedirs(out_dir, exist_ok=True)
    name = f"probe_acc_sorted_{title}.png"
    plt.savefig(os.path.join(out_dir, name))
    plt.close()


def perturb_head_input(bias_row):
    """bias_row moves device but keeps its float64: `+=` promotes for the addition and
    rounds once on store, where casting to x.dtype first would round the direction."""
    def hook(module, inputs):
        x = inputs[0]
        x[:, -1:] += bias_row.to(x.device)
        return inputs
    return hook


class ITIFit(FitMethod):
    name = "iti"

    def path_segments(self):
        return ("iti",)

    def fit(self, task, model_name):
        dims = resolve_dims(model_name)
        df = pd.read_pickle(activations_path(task.name, model_name, "head", task.character))
        dev_df, idx_train, idx_val = task.probe_indices(df)

        assert dev_df["model.layers.0.self_attn.o_proj"][dev_df.index[0]].shape[0] == dims.head_width, "activation width does not match model dims"
        print(f"Dev rows: {len(dev_df)}  train: {len(idx_train)}  val: {len(idx_val)}")

        dataset = to_per_head(dev_df, dims)
        train_acc, val_acc = get_probe_acc(dataset, idx_train, idx_val)
        print(f"train_acc < 0.5: {(train_acc < 0.5).sum()}, val_acc < 0.5: {(val_acc < 0.5).sum()}")

        normalized_dir = get_normalized_mass_mean(dataset)
        naive_std = get_naive_std(dataset)
        sorted_idx = rank_heads(val_acc, normalized_dir)

        out_dir = directions_path(task.name, model_name, self._directions_segments(), task.character).parent
        plot_probe_acc(train_acc, out_dir, "train")
        plot_probe_acc(val_acc, out_dir, "val")

        result = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "normalized_dir": normalized_dir,
            "naive_std": naive_std,
            "sorted_idx": sorted_idx,
        }
        return self._save_directions(task, model_name, result)


class ITI(HookedMethod):
    name = "iti"

    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha

    def path_segments(self):
        return ("iti",)

    def run_id(self):
        return f"k_{self.k}_alpha_{self.alpha}"

    def install(self, model, task, model_name):
        d = load_json(directions_path(task.name, model_name, self._directions_segments(), task.character))
        normalized_dir = np.array(d["normalized_dir"])
        naive_std = np.array(d["naive_std"])
        num_layers, num_heads, head_dim = normalized_dir.shape

        mask = top_k_mask(d["sorted_idx"], self.k, (num_layers, num_heads))
        bias = normalized_dir * naive_std[:, :, np.newaxis] * self.alpha * mask[:, :, np.newaxis]
        bias = torch.tensor(bias).reshape(num_layers, num_heads * head_dim)

        handles = [module.register_forward_pre_hook(perturb_head_input(bias[layer_index(name)])) for name, module in iter_o_proj(model)]
        return Steerer(handles, info={"k": self.k, "alpha": self.alpha, "heads_steered": int(mask.sum())})
