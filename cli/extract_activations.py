import argparse
import re

import pandas as pd
import torch
from tqdm import tqdm

from common.config import CHARACTERS, MODEL_IDS, deterministic, resolve_dims
from common.paths import activations_path, ensure_parent
from common.modeling import LocalGenerator, load_model_and_tokenizer
from common.hooks import iter_o_proj, layer_index
from tasks.registry import TASK_NAMES, get_task


def last_token(tensor):
    return tensor[:, -1].detach().to(torch.float32).cpu().numpy()


def register_head_collectors(model, collector):
    def collect_head(key, collector):
        def hook(module, inputs, output):
            collector[key] = last_token(inputs[0])
        return hook

    keys = []
    for name, module in iter_o_proj(model):
        module.register_forward_hook(collect_head(name, collector))
        keys.append(name)
    return keys


def register_residual_collectors(model, collector):
    def collect_residual(key, collector):
        def hook(module, inputs, output):
            # Newer transformers return the tensor directly; older versions (eg: Gemma2) return a tuple.
            hidden = output[0] if isinstance(output, tuple) else output
            collector[key] = last_token(hidden)
        return hook

    decoder_layer_re = re.compile(r"(?:model|model\.language_model)\.layers\.\d+$")
    keys = []
    for name, module in model.named_modules():
        if decoder_layer_re.match(name):
            key = f"layer_{layer_index(name)}"
            module.register_forward_hook(collect_residual(key, collector))
            keys.append(key)
    return keys


DICTATORGAME_TRIM = 6


def encode(tokenizer, chats, device, task_name):
    inputs = tokenizer.apply_chat_template(chats, padding="longest", continue_final_message=True, return_dict=True, return_tensors="pt").to(device)
    if task_name == "dictatorgame":
        inputs["input_ids"] = inputs["input_ids"][:, :-DICTATORGAME_TRIM]
        inputs["attention_mask"] = inputs["attention_mask"][:, :-DICTATORGAME_TRIM]
    return inputs


def run_forward(model, tokenizer, chats, keys, collector, batch_size, desc, task_name):
    out = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(chats), batch_size), desc=desc):
            inputs = encode(tokenizer, chats[i:i + batch_size], model.device, task_name)
            model(**inputs)
            for k in keys:
                out[k].extend(collector[k])
    return out


def extract_contrastive(task, model, tokenizer, kind, batch_size, seed):
    chats, labels, idxs = task.contrast_rows(kind, seed)
    print(f"Total prompts: {len(chats)}")

    collector = {}
    if kind == "head":
        keys = register_head_collectors(model, collector)
    else:
        keys = register_residual_collectors(model, collector)
    print(f"kind={kind}: hooked {len(keys)} modules")
    acts = run_forward(model, tokenizer, chats, keys, collector, batch_size,
                       f"Activations ({kind})", task.name)
    return {"label": labels, task.index_col: idxs, **acts}


def extract_sea(task, model, tokenizer, batch_size, base_max_new_tokens, seed):
    idxs, pos_chats, neg_chats, base_chats = task.sea_triplets(seed)
    print(f"Demonstrations: {len(idxs)}")

    continues_assistant = base_chats[0][-1]["role"] == "assistant"
    responses = LocalGenerator(model, tokenizer, continues_assistant, batch_size, base_max_new_tokens, None).generate(base_chats)
    for chat, response in zip(base_chats, responses):
        if chat[-1]["role"] == "assistant":
            chat[-1]["content"] += response
        else:
            chat.append({"role": "assistant", "content": response})

    collector = {}
    keys = register_residual_collectors(model, collector)
    print(f"hooked {len(keys)} residual modules")

    acts = {}
    for tag, chats in (("pos", pos_chats), ("neg", neg_chats), ("base", base_chats)):
        got = run_forward(model, tokenizer, chats, keys, collector, batch_size, f"Activations ({tag})", task.name)
        for k, v in got.items():
            acts[f"{k}_{tag}"] = v

    return {task.index_col: idxs, **acts}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--model", choices=list(MODEL_IDS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--character", choices=CHARACTERS, default=None)
    p.add_argument("--kind", choices=["head", "residual", "sea"], default="head")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--base_max_new_tokens", type=int, default=50)
    return p


def main():
    args = build_parser().parse_args()
    print(args)

    deterministic(args.seed)
    task = get_task(args.task, model=args.model, character=args.character)
    dims = resolve_dims(args.model)
    print(f"{args.model}: layers={dims.num_layers} heads={dims.num_heads} head_dim={dims.head_dim} hidden={dims.hidden_dim}")

    model, tokenizer = load_model_and_tokenizer(MODEL_IDS[args.model])

    print(f"=== {task} {args.model} kind={args.kind} ===")
    if args.kind == "sea":
        data = extract_sea(task, model, tokenizer, args.batch_size, args.base_max_new_tokens, args.seed)
    else:
        data = extract_contrastive(task, model, tokenizer, args.kind, args.batch_size, args.seed)

    path = ensure_parent(activations_path(task.name, args.model, args.kind, task.character))
    pd.DataFrame(data).to_pickle(path)

    print(f"Saved {path} ({len(next(iter(data.values())))} rows)")


if __name__ == "__main__":
    main()
