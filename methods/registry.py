from methods.caa import CAA, CAAFit
from methods.ideea import AutoClusters, FixedClusters, IDEEA, IDEEAFit
from methods.ideea_caa import IDEEACAA, IDEEACAAFit, LayerAutoClusters, LayerFixedClusters
from methods.iti import ITI, ITIFit
from methods.sae import SAE, resolve_sae_feature
from methods.sea import SEA, SEAFit

METHOD_NAMES = ("iti", "ideea", "caa", "ideea_caa", "sae", "sea")
#: SAE steers a hosted model through a fixed audited feature, so it fits nothing
FITTABLE = ("iti", "ideea", "caa", "ideea_caa", "sea")

#: the two clustering methods. An nc means the same thing to both, so they read the same
#: flags; the pair differs only in which activation site it clusters.
_CLUSTERS = {"ideea": (FixedClusters, AutoClusters),
             "ideea_caa": (LayerFixedClusters, LayerAutoClusters)}


def _clusters(args):
    fixed, auto = _CLUSTERS[args.method]
    if args.variant == "auto_nc":
        return auto(args.nc_min, args.nc_max)
    return fixed(args.n_clusters)


def build_fit_method(args):
    if args.method == "iti":
        return ITIFit()
    if args.method == "caa":
        return CAAFit()
    if args.method == "ideea":
        return IDEEAFit(args.variant, _clusters(args), args.seed)
    if args.method == "ideea_caa":
        return IDEEACAAFit(args.variant, _clusters(args), args.seed)
    if args.method == "sea":
        return SEAFit(args.k_ratio)
    raise ValueError(f"{args.method} has no directions to fit; choose from {list(FITTABLE)}")


def build_eval_method(args, task):
    if args.method == "iti":
        return ITI(args.k, args.alpha)
    if args.method == "caa":
        return CAA(args.caa_layer, args.alpha)
    if args.method == "ideea":
        return IDEEA(args.variant, _clusters(args), args.k, args.alpha)
    if args.method == "ideea_caa":
        return IDEEACAA(args.variant, _clusters(args), args.caa_layer, args.alpha)
    if args.method == "sea":
        return SEA(args.k_ratio, args.apply_sea_layers, args.L, args.combine)
    if args.method == "sae":
        # a property of (task, model), not a flag, so the method holds it rather than args
        return SAE(resolve_sae_feature(task, args.model), args.multiplier)
    raise KeyError(f"unknown method '{args.method}'; choose from {list(METHOD_NAMES)}")
