from itertools import permutations

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from common.config import NEGATIVE, POSITIVE, resolve_dims
from common.hooks import iter_o_proj, layer_index
from common.io import load_json, to_per_head
from common.paths import activations_path, directions_path
from methods.base import FitMethod, HookedMethod, Steerer, top_k_mask


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def solve_qap_bruteforce(directions_matrix, n_clusters):
    """Returns sigma maximizing sum_{a<b} cos_sim(v_{a,sigma(a)}, v_{b,sigma(b)}), where
    directions_matrix[i][j] = v_{i,j} with i indexing neg clusters and j pos."""
    cost = np.zeros((n_clusters, n_clusters, n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            for a in range(n_clusters):
                for b in range(n_clusters):
                    cost[i, j, a, b] = cosine_similarity(directions_matrix[i][j], directions_matrix[a][b])

    best_score = -np.inf
    best_perm = None
    for perm in permutations(range(n_clusters)):
        score = 0.0
        for a in range(n_clusters):
            for b in range(a + 1, n_clusters):
                score += cost[a, perm[a], b, perm[b]]
        if score > best_score:
            best_score = score
            best_perm = perm
    return list(best_perm), best_score


def fit_fixed_nc(dataset, n_clusters, seed):
    """matching[i] = j means neg cluster i is paired with pos cluster j."""
    layers = len(dataset)
    heads = len(dataset[0])

    cluster_dirs = [[None for _ in range(heads)] for _ in range(layers)]
    pos_centroids = [[None for _ in range(heads)] for _ in range(layers)]
    neg_centroids = [[None for _ in range(heads)] for _ in range(layers)]
    matching = [[None for _ in range(heads)] for _ in range(layers)]

    for l in tqdm(range(layers), desc="IDEEA directions"):
        for h in range(heads):
            activations = dataset[l][h]
            pos_data = activations["data"][activations["label"] == POSITIVE]
            neg_data = activations["data"][activations["label"] == NEGATIVE]

            km_pos = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            km_neg = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            km_pos.fit(pos_data)
            km_neg.fit(neg_data)

            pos_centers = km_pos.cluster_centers_  # (N, head_dim)
            neg_centers = km_neg.cluster_centers_  # (N, head_dim)

            dir_matrix = np.zeros((n_clusters, n_clusters, pos_data.shape[1]))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    d = (pos_centers[j] - neg_centers[i]).astype(np.float32)
                    norm = np.linalg.norm(d)
                    if norm > 0:
                        d = d / norm
                    dir_matrix[i][j] = d

            best_perm, _ = solve_qap_bruteforce(dir_matrix, n_clusters)

            cluster_dirs[l][h] = [dir_matrix[i][best_perm[i]].tolist() for i in range(n_clusters)]
            pos_centroids[l][h] = pos_centers.tolist()
            neg_centroids[l][h] = neg_centers.tolist()
            matching[l][h] = best_perm

    return cluster_dirs, pos_centroids, neg_centroids, matching


def select_nc(data, nc_min, nc_max, seed):
    """Silhouette-best number of clusters, preferring smaller nc on ties."""
    best_nc, best_score = nc_min, -1.0
    for nc in range(nc_min, nc_max + 1):
        if nc >= len(data):
            break
        km = KMeans(n_clusters=nc, random_state=seed, n_init=10)
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_nc, best_score = nc, score
    return best_nc


def fit_auto_nc(dataset, nc_min, nc_max, seed):
    """No matching: a variable per-head nc leaves the two sides unpaired."""
    layers = len(dataset)
    heads = len(dataset[0])

    nc_pos_map = [[None for _ in range(heads)] for _ in range(layers)]
    nc_neg_map = [[None for _ in range(heads)] for _ in range(layers)]
    pos_centroids = [[None for _ in range(heads)] for _ in range(layers)]
    neg_centroids = [[None for _ in range(heads)] for _ in range(layers)]

    for l in tqdm(range(layers), desc="IDEEA auto-nc directions"):
        for h in range(heads):
            activations = dataset[l][h]
            pos_data = activations["data"][activations["label"] == POSITIVE]
            neg_data = activations["data"][activations["label"] == NEGATIVE]

            nc_pos = select_nc(pos_data, nc_min, nc_max, seed)
            nc_neg = select_nc(neg_data, nc_min, nc_max, seed)

            km_pos = KMeans(n_clusters=nc_pos, random_state=seed, n_init=10)
            km_neg = KMeans(n_clusters=nc_neg, random_state=seed, n_init=10)
            km_pos.fit(pos_data)
            km_neg.fit(neg_data)

            nc_pos_map[l][h] = nc_pos
            nc_neg_map[l][h] = nc_neg
            pos_centroids[l][h] = km_pos.cluster_centers_.tolist()
            neg_centroids[l][h] = km_neg.cluster_centers_.tolist()

    return nc_pos_map, nc_neg_map, pos_centroids, neg_centroids


VARIANTS = ("min_perp", "nearest_cluster", "nearest_pos_neg", "auto_nc")

#: variants that need the QAP pos<->neg matching, hence a fixed nc
_MATCHED = ("min_perp", "nearest_cluster")


def _as_tensor_grid(nested, mask):
    """Nested lists -> tensors where mask==1; unsteered heads stay None."""
    num_layers, num_heads = mask.shape
    grid = [[None] * num_heads for _ in range(num_layers)]
    for l in range(num_layers):
        for h in range(num_heads):
            if mask[l][h] == 1:
                grid[l][h] = torch.tensor(np.array(nested[l][h]), dtype=torch.float32)
    return grid


def _nearest(x, centroids):
    dists = torch.sum((x.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=-1)
    return centroids[torch.argmin(dists, dim=-1)]


def _select_min_perp(x, dirs, pos_c, neg_c, inv_perm):
    """argmax_i |x . v_i| == argmin_i ||x - (x.v_i)v_i||"""
    return dirs[torch.argmax(torch.abs(dirs @ x.T), dim=0)]


def _select_nearest_cluster(x, dirs, pos_c, neg_c, inv_perm):
    """Nearest of all 2N centroids (neg first, then pos); a pos hit maps back through
    the QAP matching to its paired neg cluster's direction."""
    n = dirs.shape[0]
    all_c = torch.cat([neg_c, pos_c], dim=0)
    dists = torch.sum((x.unsqueeze(1) - all_c.unsqueeze(0)) ** 2, dim=-1)
    nearest = torch.argmin(dists, dim=-1)
    idx = nearest.clone()
    is_pos = nearest >= n
    idx[is_pos] = inv_perm[nearest[is_pos] - n]
    return dirs[idx]


def _select_nearest_pos_neg(x, dirs, pos_c, neg_c, inv_perm):
    """Nearest pos and neg centroid picked independently, so the only rule that
    tolerates a variable nc."""
    v = _nearest(x, pos_c) - _nearest(x, neg_c)
    return v / torch.norm(v, dim=-1, keepdim=True)


SELECT = {
    "min_perp": _select_min_perp,
    "nearest_cluster": _select_nearest_cluster,
    "nearest_pos_neg": _select_nearest_pos_neg,
    "auto_nc": _select_nearest_pos_neg,
}


def perturb_head_input_ideea(layer, variant, mask, alpha, head_dim, num_heads, cluster_dirs, pos_centroids, neg_centroids, inv_perm, naive_std):
    select = SELECT[variant]

    # The matched rules multiply by a std already rounded to the activation dtype while
    # nearest_pos_neg reads the float64 scalar, which gives different bfloat16 logits.
    rounds_std = variant in _MATCHED
    std_scalars = {h: torch.tensor(naive_std[layer][h]) for h in range(num_heads) if mask[layer][h]} if rounds_std else {}

    def head_tensor(grid, h, like, dtype=True):
        """Unlike ITI and CAA, centroids and directions do take the activation dtype:
        they feed matmuls needing operands of one type. Index tensors move device only."""
        if grid is None or grid[layer][h] is None:
            return None
        t = grid[layer][h]
        return t.to(like.device, like.dtype) if dtype else t.to(like.device)

    def hook(module, inputs):
        x_full = inputs[0]
        for h in range(num_heads):
            if mask[layer][h] == 0:
                continue
            s, e = h * head_dim, (h + 1) * head_dim
            # basic indexing -> a view, so `+=` writes through to inputs[0]
            x = x_full[:, -1, s:e]                       # (batch, head_dim)
            direction = select(
                x,
                head_tensor(cluster_dirs, h, x),
                head_tensor(pos_centroids, h, x),
                head_tensor(neg_centroids, h, x),
                head_tensor(inv_perm, h, x, dtype=False),
            )
            std = std_scalars[h].to(x.device, x.dtype) if rounds_std else naive_std[layer][h]
            x += alpha * direction * std
        return inputs
    return hook


class FixedClusters:
    """Shared by the three fixed-nc variants, which is why the directions path drops the
    variant while the eval path keeps it.

    `method` roots the segments and `_fit_clusters` runs the clustering, so a method that
    clusters a different activation site subclasses these two rather than restating what
    an nc means -- see methods.ideea_caa."""

    method = "ideea"

    def __init__(self, n_clusters):
        assert n_clusters > 1, "n_clusters == 1 is just ITI"
        self.n_clusters = n_clusters

    def path_segments(self, variant):
        return (self.method, variant, f"nc_{self.n_clusters}")

    def directions_segments(self):
        return (self.method, f"nc_{self.n_clusters}")

    def _fit_clusters(self, dataset, seed):
        return fit_fixed_nc(dataset, self.n_clusters, seed)

    def fit(self, dataset, seed):
        dirs, pos_c, neg_c, matching = self._fit_clusters(dataset, seed)
        return {"n_clusters": self.n_clusters, "cluster_dirs": dirs, "pos_centroids": pos_c, "neg_centroids": neg_c, "matching": matching}


class AutoClusters:
    method = "ideea"

    def __init__(self, nc_min, nc_max):
        assert nc_min >= 2, "nc_min must be >= 2 (silhouette needs >= 2 clusters)"
        assert nc_max >= nc_min, "nc_max must be >= nc_min"
        self.nc_min = nc_min
        self.nc_max = nc_max

    def path_segments(self, variant):
        return self.directions_segments()

    def directions_segments(self):
        return (self.method, "auto_nc", f"nc_{self.nc_min}_{self.nc_max}")

    def _fit_clusters(self, dataset, seed):
        return fit_auto_nc(dataset, self.nc_min, self.nc_max, seed)

    def fit(self, dataset, seed):
        nc_pos, nc_neg, pos_c, neg_c = self._fit_clusters(dataset, seed)
        # np.mean over the rectangular [layer][head] grid, or over the flat per-layer
        # list a residual-site subclass returns
        print(f"nc_pos mean={np.mean(nc_pos):.2f}  nc_neg mean={np.mean(nc_neg):.2f}")
        return {"nc_min": self.nc_min, "nc_max": self.nc_max, "nc_pos": nc_pos, "nc_neg": nc_neg, "pos_centroids": pos_c, "neg_centroids": neg_c}


class IDEEAFit(FitMethod):
    name = "ideea"

    def __init__(self, variant, clusters, seed):
        self.variant = variant
        self.clusters = clusters
        self.seed = seed

    def path_segments(self):
        return self.clusters.path_segments(self.variant)

    def _directions_segments(self):
        return self.clusters.directions_segments()

    def fit(self, task, model_name):
        iti_path = directions_path(task.name, model_name, ("iti",), task.character)
        if not iti_path.exists():
            raise FileNotFoundError(f"{iti_path} not found. IDEEA reuses ITI's head ranking and naive_std; run `compute_directions --method iti` first.")
        iti = load_json(iti_path)

        dims = resolve_dims(model_name)
        df = pd.read_pickle(activations_path(task.name, model_name, "head", task.character))
        dev_df, _, _ = task.probe_indices(df)
        print(f"Clustering on {len(dev_df)} dev rows")
        dataset = to_per_head(dev_df, dims)

        result = {"sorted_idx": iti["sorted_idx"], "naive_std": iti["naive_std"]}
        result.update(self.clusters.fit(dataset, self.seed))
        return self._save_directions(task, model_name, result)


class IDEEA(HookedMethod):
    name = "ideea"

    def __init__(self, variant, clusters, k, alpha):
        self.variant = variant
        self.clusters = clusters
        self.k = k
        self.alpha = alpha

    def path_segments(self):
        return self.clusters.path_segments(self.variant)

    def _directions_segments(self):
        return self.clusters.directions_segments()

    def run_id(self):
        return f"k_{self.k}_alpha_{self.alpha}"

    def install(self, model, task, model_name):
        dpath = directions_path(task.name, model_name, self._directions_segments(), task.character)
        if not dpath.exists():
            raise FileNotFoundError(f"{dpath} not found; run compute_directions first.")
        d = load_json(dpath)

        naive_std = np.array(d["naive_std"])
        pos_centroids_raw = d["pos_centroids"]
        num_layers = len(pos_centroids_raw)
        num_heads = len(pos_centroids_raw[0])
        head_dim = len(pos_centroids_raw[0][0][0])
        mask = top_k_mask(d["sorted_idx"], self.k, (num_layers, num_heads))

        pos_c = _as_tensor_grid(pos_centroids_raw, mask)
        neg_c = _as_tensor_grid(d["neg_centroids"], mask)

        cluster_dirs = inv_perm = None
        if self.variant in _MATCHED:
            if "cluster_dirs" not in d:
                raise ValueError(f"variant={self.variant} needs the QAP matching, but {dpath} holds auto_nc directions (unmatched clusters).")
            cluster_dirs = _as_tensor_grid(d["cluster_dirs"], mask)
            # matching[i] = j (neg i paired with pos j); invert to map pos -> neg
            inv_perm = [[None] * num_heads for _ in range(num_layers)]
            for l in range(num_layers):
                for h in range(num_heads):
                    if mask[l][h] == 1:
                        matching = d["matching"][l][h]
                        inv = np.zeros(len(matching), dtype=np.int64)
                        for i, j in enumerate(matching):
                            inv[j] = i
                        inv_perm[l][h] = torch.tensor(inv, dtype=torch.long)

        handles = [
            module.register_forward_pre_hook(perturb_head_input_ideea(layer_index(name), self.variant, mask, self.alpha, head_dim, num_heads, cluster_dirs, pos_c, neg_c, inv_perm, naive_std))
            for name, module in iter_o_proj(model)
        ]
        return Steerer(handles, info={"variant": self.variant, "k": self.k, "alpha": self.alpha, "heads_steered": int(mask.sum())})
