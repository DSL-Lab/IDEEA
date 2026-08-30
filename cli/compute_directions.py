import argparse

from common.config import CHARACTERS, MODEL_IDS, deterministic
from methods.ideea import VARIANTS as IDEEA_VARIANTS
from methods.registry import FITTABLE, build_fit_method
from tasks.registry import TASK_NAMES, get_task


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--model", choices=list(MODEL_IDS), required=True)
    p.add_argument("--method", choices=FITTABLE, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--character", choices=CHARACTERS, default=None)

    ideea = p.add_argument_group("ideea / ideea_caa")
    ideea.add_argument("--variant", choices=IDEEA_VARIANTS, default="nearest_pos_neg")
    ideea.add_argument("--n_clusters", type=int, default=4)
    ideea.add_argument("--nc_min", type=int, default=2)
    ideea.add_argument("--nc_max", type=int, default=6)

    sea = p.add_argument_group("sea")
    sea.add_argument("--k_ratio", type=float, default=0.998)
    return p


def main():
    args = build_parser().parse_args()
    print(args)
    method = build_fit_method(args)

    deterministic(args.seed)
    task = get_task(args.task, model=args.model, character=args.character)

    print(f"=== {task} {args.model} method={args.method} ===")
    method.fit(task, args.model)


if __name__ == "__main__":
    main()
