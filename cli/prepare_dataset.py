import argparse
import inspect

from common.config import CHARACTERS, MODEL_IDS, deterministic
from tasks.registry import TASK_NAMES, get_task


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--seed", type=int, required=True)

    tv = p.add_argument_group("twinviews")
    tv.add_argument("--n", type=int, default=1000)

    dg = p.add_argument_group("dictatorgame")
    dg.add_argument("--model", choices=list(MODEL_IDS), default=None)
    dg.add_argument("--character", choices=CHARACTERS, default=None)
    dg.add_argument("--batch_size", type=int, default=1)
    return p


def main():
    args = build_parser().parse_args()
    print(args)

    deterministic(args.seed)
    task = get_task(args.task, model=args.model, character=args.character)

    params = inspect.signature(task.prepare).parameters
    kwargs = {k: v for k, v in vars(args).items() if k in params}
    task.prepare(**kwargs)


if __name__ == "__main__":
    main()
