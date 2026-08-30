import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from common.config import NEGATIVE, POSITIVE, resolve_dims
from common.io import load_json
from common.paths import activations_path, directions_path
from methods.base import FitMethod, HookedMethod, Steerer
from methods.ideea import AutoClusters, FixedClusters, select_nc, solve_qap_bruteforce

#: variants that need the QAP pos<->neg matching, hence a fixed nc. Same rule as IDEEA's,
#: for the same reason, but owned here so neither method reaches into the other.
_MATCHED = ("min_perp", "nearest_cluster")


def to_per_layer(df, dims):
    """Long-format residual DataFrame -> dataset[layer] = {data, label}.

    The per-head counterpart is common.io.to_per_head; this one has no reshape to do,
    since a residual column already holds one layer's whole vector."""
    labels = df["label"].to_numpy()
    return [{"data": np.stack(df[f"layer_{l}"].to_numpy()), "label": labels}
            for l in range(dims.num_layers)]


def fit_fixed_nc_layers(dataset, n_clusters, seed):
    """matching[i] = j means neg cluster i is paired with pos cluster j.

    Unlike methods.ideea.fit_fixed_nc the stored directions are not normalized, so alpha
    stays on CAA's scale. The QAP cost is a cosine similarity, which is invariant to
    that, so the matching is the one IDEEA would have found."""
    cluster_dirs, pos_centroids, neg_centroids, matching = [], [], [], []

    for l in tqdm(range(len(dataset)), desc="IDEEA-CAA directions"):
        activations = dataset[l]
        pos_data = activations["data"][activations["label"] == POSITIVE]
        neg_data = activations["data"][activations["label"] == NEGATIVE]

        km_pos = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        km_neg = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        km_pos.fit(pos_data)
        km_neg.fit(neg_data)

        pos_centers = km_pos.cluster_centers_  # (N, hidden)
        neg_centers = km_neg.cluster_centers_  # (N, hidden)

        # dir_matrix[i][j] = pos j - neg i, the direction pairing those two clusters
        dir_matrix = pos_centers[np.newaxis, :, :] - neg_centers[:, np.newaxis, :]
        best_perm, _ = solve_qap_bruteforce(dir_matrix, n_clusters)

        cluster_dirs.append([dir_matrix[i][best_perm[i]].tolist() for i in range(n_clusters)])
        pos_centroids.append(pos_centers.tolist())
        neg_centroids.append(neg_centers.tolist())
        matching.append(best_perm)

    return cluster_dirs, pos_centroids, neg_centroids, matching


def fit_auto_nc_layers(dataset, nc_min, nc_max, seed):
    """No matching: a variable per-layer nc leaves the two sides unpaired."""
    nc_pos_map, nc_neg_map, pos_centroids, neg_centroids = [], [], [], []

    for l in tqdm(range(len(dataset)), desc="IDEEA-CAA auto-nc directions"):
        activations = dataset[l]
        pos_data = activations["data"][activations["label"] == POSITIVE]
        neg_data = activations["data"][activations["label"] == NEGATIVE]

        nc_pos = select_nc(pos_data, nc_min, nc_max, seed)
        nc_neg = select_nc(neg_data, nc_min, nc_max, seed)

        km_pos = KMeans(n_clusters=nc_pos, random_state=seed, n_init=10)
        km_neg = KMeans(n_clusters=nc_neg, random_state=seed, n_init=10)
        km_pos.fit(pos_data)
        km_neg.fit(neg_data)

        nc_pos_map.append(nc_pos)
        nc_neg_map.append(nc_neg)
        pos_centroids.append(km_pos.cluster_centers_.tolist())
        neg_centroids.append(km_neg.cluster_centers_.tolist())

    return nc_pos_map, nc_neg_map, pos_centroids, neg_centroids


class LayerFixedClusters(FixedClusters):
    method = "ideea_caa"

    def _fit_clusters(self, dataset, seed):
        return fit_fixed_nc_layers(dataset, self.n_clusters, seed)


class LayerAutoClusters(AutoClusters):
    method = "ideea_caa"

    def _fit_clusters(self, dataset, seed):
        return fit_auto_nc_layers(dataset, self.nc_min, self.nc_max, seed)


def _nearest(x, centroids):
    dists = torch.sum((x.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=-1)
    return centroids[torch.argmin(dists, dim=-1)]


def _select_min_perp(x, dirs, unit_dirs, pos_c, neg_c, inv_perm):
    """argmax_i |x . v_i| == argmin_i ||x - (x.v_i)v_i||, ranked over the UNIT directions:
    the residual is ||x||^2 - (x.v_i)^2/||v_i||^2, so dropping the norm would let a long
    cluster difference win on length rather than angle. The raw v_i is what gets applied."""
    return dirs[torch.argmax(torch.abs(unit_dirs @ x.T), dim=0)]


def _select_nearest_cluster(x, dirs, unit_dirs, pos_c, neg_c, inv_perm):
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


def _select_nearest_pos_neg(x, dirs, unit_dirs, pos_c, neg_c, inv_perm):
    """Nearest pos and neg centroid picked independently, so the only rule that tolerates
    a variable nc. Their difference IS the direction -- CAA's own formula, over the
    closest pair of clusters rather than the two global means, so it is not normalized."""
    return _nearest(x, pos_c) - _nearest(x, neg_c)


SELECT = {
    "min_perp": _select_min_perp,
    "nearest_cluster": _select_nearest_cluster,
    "nearest_pos_neg": _select_nearest_pos_neg,
    "auto_nc": _select_nearest_pos_neg,
}


def perturb_residual_ideea(variant, alpha, dirs, unit_dirs, pos_centroids, neg_centroids, inv_perm):
    select = SELECT[variant]

    def on_device(t, like):
        """Device only, never dtype: selection runs in the directions' float64, which is
        the reverse of IDEEA casting its centroids down to the activation dtype. Either
        gives the distances and the matmul operands of one type, but promoting keeps the
        applied direction float64, so the bias rounds once on store as in ITI and CAA."""
        return None if t is None else t.to(like.device)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # a copy, not a view: `+=` below writes back to the row this reads
        x = hidden[:, -1].to(torch.float64)                  # (batch, hidden)
        direction = select(
            x,
            on_device(dirs, x),
            on_device(unit_dirs, x),
            on_device(pos_centroids, x),
            on_device(neg_centroids, x),
            on_device(inv_perm, x),
        )
        # one direction per row, where CAA adds the same one to every row
        hidden[:, -1] += alpha * direction
        return output
    return hook


class IDEEACAAFit(FitMethod):
    name = "ideea_caa"

    def __init__(self, variant, clusters, seed):
        self.variant = variant
        self.clusters = clusters
        self.seed = seed

    def path_segments(self):
        return self.clusters.path_segments(self.variant)

    def _directions_segments(self):
        return self.clusters.directions_segments()

    def fit(self, task, model_name):
        dims = resolve_dims(model_name)
        df = pd.read_pickle(activations_path(task.name, model_name, "residual", task.character))
        dev_df = task.dev_rows(df)
        labels = dev_df["label"].to_numpy()
        n_pos = int((labels == POSITIVE).sum())
        n_neg = int((labels == NEGATIVE).sum())
        print(f"Clustering on {len(dev_df)} dev rows ({n_pos} pos, {n_neg} neg)")

        dataset = to_per_layer(dev_df, dims)
        return self._save_directions(task, model_name, self.clusters.fit(dataset, self.seed))


class IDEEACAA(HookedMethod):
    """IDEEA's four variants over CAA's activations and steering site: cluster a layer's
    residual stream instead of a head's attention output, and pick the direction per
    input at the layer --caa_layer names. The selection rules are defined on clusters,
    not on the activation site, so they carry over unchanged and the nc flags simply join
    CAA's --caa_layer / --alpha. There is no head ranking, hence no --k and no ITI."""

    name = "ideea_caa"

    def __init__(self, variant, clusters, caa_layer, alpha):
        if caa_layer is None:
            raise ValueError("--caa_layer is required for IDEEA-CAA steering")
        self.variant = variant
        self.clusters = clusters
        self.caa_layer = caa_layer
        self.alpha = alpha

    def path_segments(self):
        return self.clusters.path_segments(self.variant)

    def _directions_segments(self):
        return self.clusters.directions_segments()

    def run_id(self):
        return f"layer_{self.caa_layer}_alpha_{self.alpha}"

    def install(self, model, task, model_name):
        dpath = directions_path(task.name, model_name, self._directions_segments(), task.character)
        if not dpath.exists():
            raise FileNotFoundError(f"{dpath} not found; run compute_directions first.")
        d = load_json(dpath)

        layer = self.caa_layer
        num_layers = len(d["pos_centroids"])
        # a negative index would wrap silently onto a real layer and steer the wrong one
        if not 0 <= layer < num_layers:
            raise ValueError(f"--caa_layer {layer} is outside the {num_layers} layers in {dpath}")

        pos_c = torch.tensor(np.array(d["pos_centroids"][layer]), dtype=torch.float64)
        neg_c = torch.tensor(np.array(d["neg_centroids"][layer]), dtype=torch.float64)

        dirs = unit_dirs = inv_perm = None
        if self.variant in _MATCHED:
            if "cluster_dirs" not in d:
                raise ValueError(f"variant={self.variant} needs the QAP matching, but {dpath} holds auto_nc directions (unmatched clusters).")
            dirs = torch.tensor(np.array(d["cluster_dirs"][layer]), dtype=torch.float64)
            # clamped rather than skipped as in ITI: a zero direction would only arise
            # from two coincident centroids, and steering it is a no-op either way
            unit_dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            # matching[i] = j (neg i paired with pos j); invert to map pos -> neg
            matching = d["matching"][layer]
            inv = np.zeros(len(matching), dtype=np.int64)
            for i, j in enumerate(matching):
                inv[j] = i
            inv_perm = torch.tensor(inv, dtype=torch.long)

        info = {"variant": self.variant, "caa_layer": layer, "alpha": self.alpha,
                "nc_pos": len(pos_c), "nc_neg": len(neg_c)}
        if dirs is not None:
            # CAA's direction_norm, one per cluster: this is what says whether alpha is
            # priced on the same scale as a CAA run at the same layer
            info["direction_norms"] = dirs.norm(dim=-1).tolist()

        target = model.get_submodule(f"model.layers.{layer}")
        handle = target.register_forward_hook(perturb_residual_ideea(self.variant, self.alpha, dirs, unit_dirs, pos_c, neg_c, inv_perm))
        return Steerer([handle], info=info)
