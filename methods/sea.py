import numpy as np
import pandas as pd
import torch

from common.config import resolve_dims
from common.io import save_json
from common.paths import activations_path, directions_dir, ensure_parent
from methods.base import FitMethod, HookedMethod, Steerer


def ev_id(normalised_ev, k_ratio):
    cum = 0.0
    for i, ev in enumerate(normalised_ev):
        cum += ev
        if cum >= k_ratio:
            return i + 1
    return len(normalised_ev)


def resolve_sea_layers(spec, L, num_layers):
    if spec == "all":
        return list(range(num_layers))
    if spec == "first-L":
        return list(range(int(L)))
    if spec == "last-L":
        return list(range(num_layers - int(L), num_layers))
    if spec == "last":
        return [num_layers - 1]
    if spec == "specific":
        return [int(i) for i in str(L).split(",")]
    raise ValueError(f"Unknown apply_sea_layers: {spec}")


def sea_segments(k_ratio):
    return ("sea", f"k_ratio_{k_ratio}")


def projection_path(task, model_name, k_ratio):
    return directions_dir(task.name, model_name, sea_segments(k_ratio), task.character) / "projections.pt"


def project_residual(U_pos, U_neg_top, combine):
    """Unlike the additive methods, the whole projection runs in float32 and casts
    back once at the end, so the projections only ever move device."""
    eps = 1e-8

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        dtype = hidden.dtype
        x = hidden.to(torch.float32)                      # (bs, L, d)

        Up = U_pos.to(x.device)
        Un = U_neg_top.to(x.device)
        pos_x = torch.matmul(torch.matmul(x, Up), Up.T)   # P_pos @ x
        neg_x = x - torch.matmul(torch.matmul(x, Un), Un.T)  # P_neg @ x

        if combine == "average":
            comb = (pos_x + neg_x) / 2
        else:
            # per-sequence over its own token axis, so no statistics leak across batch
            comb = pos_x + neg_x
            norm = x.norm(dim=1, keepdim=True)            # (bs, 1, d)
            norm_comb = comb.norm(dim=1, keepdim=True)
            comb = comb * norm / (norm_comb + eps)

        comb = comb.to(dtype)
        if is_tuple:
            return (comb,) + output[1:]
        return comb
    return hook


class SEAFit(FitMethod):
    name = "sea"

    def __init__(self, k_ratio):
        self.k_ratio = k_ratio

    def path_segments(self):
        return sea_segments(self.k_ratio)

    def fit(self, task, model_name):
        dims = resolve_dims(model_name)
        df = pd.read_pickle(activations_path(task.name, model_name, "sea", task.character))
        N = len(df)
        print(f"Demonstrations: {N}  (layers={dims.num_layers}, hidden={dims.hidden_dim})")

        U_pos_list, U_neg_top_list, stats = [], [], {}
        for l in range(dims.num_layers):
            # (N, d) -> (d, N) to match the paper's convention
            pos = torch.tensor(np.stack(df[f"layer_{l}_pos"].values), dtype=torch.float64).T
            neg = torch.tensor(np.stack(df[f"layer_{l}_neg"].values), dtype=torch.float64).T
            base = torch.tensor(np.stack(df[f"layer_{l}_base"].values), dtype=torch.float64).T

            # positive: KEEP the top components (max covariance with positive)
            U_pos, s_pos, _ = torch.linalg.svd(torch.matmul(base, pos.T) / N, full_matrices=True)
            ev_pos = torch.pow(s_pos, 2)
            k_pos = ev_id((ev_pos / ev_pos.sum()).tolist(), self.k_ratio)

            # negative: DROP the top components (min covariance with negative)
            U_neg, s_neg, _ = torch.linalg.svd(torch.matmul(base, neg.T) / N, full_matrices=True)
            ev_neg = torch.pow(s_neg, 2)
            k_neg = ev_id((ev_neg / ev_neg.sum()).tolist(), self.k_ratio)

            U_pos_list.append(U_pos[:, :k_pos].float().contiguous())
            U_neg_top_list.append(U_neg[:, :k_neg].float().contiguous())
            stats[l] = {"k_pos": int(k_pos), "k_neg": int(k_neg)}
            print(f"  layer {l}: k_pos={k_pos}, k_neg={k_neg}")

        out_path = ensure_parent(projection_path(task, model_name, self.k_ratio))
        torch.save({"U_pos": U_pos_list, "U_neg_top": U_neg_top_list, "k_ratio": self.k_ratio,
                    "hidden_dim": dims.hidden_dim, "num_layers": dims.num_layers}, out_path)
        save_json(out_path.with_name("stats.json"), stats)
        print(f"Saved {out_path}")
        return stats


class SEA(HookedMethod):
    name = "sea"

    def __init__(self, k_ratio, apply_sea_layers, L, combine):
        self.k_ratio = k_ratio
        self.apply_sea_layers = apply_sea_layers
        self.L = L
        self.combine = combine

    def path_segments(self):
        return sea_segments(self.k_ratio)

    def run_id(self):
        parts = [self.apply_sea_layers]
        if self.apply_sea_layers in ("first-L", "last-L", "specific"):
            parts.append(str(self.L))
        if self.combine != "l2_norm":
            parts.append(self.combine)
        return "_".join(parts)

    def install(self, model, task, model_name):
        path = projection_path(task, model_name, self.k_ratio)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run compute_directions first.")
        proj = torch.load(path, weights_only=False)

        dims = resolve_dims(model_name)
        layers = resolve_sea_layers(self.apply_sea_layers, self.L, dims.num_layers)
        print(f"Editing layers: {layers}")

        handles = []
        for l in layers:
            target = model.get_submodule(f"model.layers.{l}")
            handles.append(target.register_forward_hook(project_residual(proj["U_pos"][l], proj["U_neg_top"][l], self.combine)))
        return Steerer(handles, info={"k_ratio": self.k_ratio, "layers": layers, "combine": self.combine})
