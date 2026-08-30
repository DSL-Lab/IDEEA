import argparse

from common.config import CHARACTERS, MODEL_IDS, deterministic
from common.io import load_json, save_json
from common.modeling import LocalGenerator, load_model_and_tokenizer
from common.paths import eval_path, judged_path
from methods.ideea import VARIANTS as IDEEA_VARIANTS
from methods.registry import METHOD_NAMES, build_eval_method
from tasks.dictatorgame import PERSONA_STRENGTHS
from tasks.registry import TASK_NAMES, get_task

EVAL_METHODS = ("base",) + METHOD_NAMES


def base_run_id(args):
    """A prompted baseline is still unsteered, but it is not the neutral one, so it
    must not overwrite the generations base/ already holds."""
    if args.persona_strength == "none":
        return "base"
    return f"base_persona_{args.persona_strength}"


def generations_path(task, method, args):
    if args.method == "base":
        character = task.character if args.persona_strength != "none" else None
        return eval_path(task.name, args.model, (), base_run_id(args), character)
    return eval_path(task.name, args.model, method.path_segments(), method.run_id(), task.character)


def do_generate(task, method, args, out_path):
    deterministic(args.seed)
    requests = task.eval_requests()
    print(f"Eval prompts: {len(requests)}")
    chats = [r.chat for r in requests]

    steerer = None
    if args.method == "sae":
        resume, resume_skipped = [], []
        if out_path.exists() and not args.overwrite:
            previous = load_json(out_path)
            resume = previous.get("raw", [])
            resume_skipped = previous.get("skipped", [])

        def checkpoint(answers, skipped):
            save_json(out_path, {"config": vars(args), "raw": answers, "skipped": skipped, "partial": True})

        generator = method.make_generator(task.sae_max_new_tokens, task.sample_temperature or 0, task.sae_seeds(requests, args.seed), resume, checkpoint, resume_skipped)
        info = generator.info
    else:
        model, tokenizer = load_model_and_tokenizer(MODEL_IDS[args.model])
        info = {}
        if args.method != "base":
            steerer = method.install(model, task, args.model)
            info = steerer.info
            print(f"Steering installed: {info}")
        generator = LocalGenerator(model, tokenizer, task.continue_final_message, args.batch_size, task.max_new_tokens, task.sample_temperature)

    try:
        raw = generator.generate(chats)
    finally:
        if steerer is not None:
            steerer.remove()

    # Only SAE can refuse a prompt outright, so this is empty for every local method.
    # Refused rows are dropped from records rather than carried as None, so the judge
    # never scores a prompt that was never answered -- its `n` is the set evaluated.
    skipped = sorted(generator.skipped)
    if skipped:
        print(f"Refused by the API, excluded from scoring: {len(skipped)}/{len(requests)} prompts")
    records = [{"key": r.key, "meta": r.meta, "response": task.parse(text)}
               for i, (r, text) in enumerate(zip(requests, raw)) if i not in set(skipped)]
    log = {"config": vars(args), "steering": info, "raw": raw, "skipped": skipped, "records": records}
    save_json(out_path, log)
    print(f"Saved generations -> {out_path}")
    return log


def do_judge(task, args, gen_path):
    if not gen_path.exists():
        raise FileNotFoundError(f"{gen_path} not found. Run --phase generate first.")
    log = load_json(gen_path)
    if log.get("partial"):
        raise ValueError(f"{gen_path} holds a partial generation set; finish --phase generate before judging.")
    result = task.judge(log["records"], args.judge_batch_size)
    out = judged_path(gen_path)
    save_json(out, {"generations_path": str(gen_path), **result})
    print(f"Saved scores -> {out}")
    return result


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--model", choices=list(MODEL_IDS), required=True)
    p.add_argument("--method", choices=EVAL_METHODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--character", choices=CHARACTERS, default=None)
    p.add_argument("--phase", choices=["generate", "judge", "all"], default="all")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--judge_batch_size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")

    heads = p.add_argument_group("iti / ideea")
    heads.add_argument("--k", type=int, default=48)

    strength = p.add_argument_group("iti / ideea / caa / ideea_caa")
    strength.add_argument("--alpha", type=float, default=15.0)

    ideea = p.add_argument_group("ideea / ideea_caa")
    ideea.add_argument("--variant", choices=IDEEA_VARIANTS, default="nearest_pos_neg")
    ideea.add_argument("--n_clusters", type=int, default=4)
    ideea.add_argument("--nc_min", type=int, default=2)
    ideea.add_argument("--nc_max", type=int, default=6)

    caa = p.add_argument_group("caa / ideea_caa")
    caa.add_argument("--caa_layer", type=int, default=None)

    sae = p.add_argument_group("sae")
    sae.add_argument("--multiplier", type=float, default=1.0)

    sea = p.add_argument_group("sea")
    sea.add_argument("--k_ratio", type=float, default=0.998)
    sea.add_argument("--apply_sea_layers",
                     choices=["all", "first-L", "last-L", "last", "specific"],
                     default="last-L")
    sea.add_argument("--L", default="21")
    sea.add_argument("--combine", choices=["l2_norm", "average"], default="l2_norm")

    dg = p.add_argument_group("dictatorgame")
    dg.add_argument("--persona_strength", choices=PERSONA_STRENGTHS, default="none")
    return p


def check_persona_strength(p, args):
    if args.persona_strength != "none" and (args.task, args.method) != ("dictatorgame", "base"):
        p.error("--persona_strength is a dictatorgame baseline; it needs --task dictatorgame --method base")


def main():
    p = build_parser()
    args = p.parse_args()
    check_persona_strength(p, args)
    print(args)

    deterministic(args.seed)
    task = get_task(args.task, model=args.model, character=args.character, persona_strength=args.persona_strength)
    method = None if args.method == "base" else build_eval_method(args, task)

    print(f"=== {task} {args.model} method={args.method} ===")
    gen_path = generations_path(task, method, args)

    if args.phase in ("generate", "all"):
        existing = load_json(gen_path) if gen_path.exists() else None
        if args.overwrite or existing is None or existing.get("partial"):
            do_generate(task, method, args, gen_path)
        else:
            print(f"Generations exist, skipping (use --overwrite): {gen_path}")

    if args.phase in ("judge", "all"):
        do_judge(task, args, gen_path)


if __name__ == "__main__":
    main()
