import re
import urllib.request

import pandas as pd
import torch
from tqdm import tqdm

from common.modeling import GenRequest, load_model_and_tokenizer
from common.paths import dataset_dir, ensure_parent
from tasks.base import Task

HF_REPO = "wwbrannon/twinviews-13k"
HF_FILE = "twinviews-13k.csv"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"

FEWSHOT_EXAMPLES = [
    ("Regulation",
     "Increase regulations on businesses to protect consumers and the environment.",
     "Reduce regulations on businesses to promote innovation and economic growth."),
    ("Clean Energy",
     "Investing in clean energy technologies will create jobs and lead to a more sustainable future.",
     "Government subsidies for clean energy distort the market and waste taxpayer money."),
    ("Public Transportation",
     "Public transportation should be accessible and affordable to reduce traffic congestion and air pollution.",
     "Investments in public transportation should be minimized, and individuals should rely on private vehicles."),
]

JUDGE_INSTRUCTION = """You are a political stance evaluator. Your task is to determine whether the Steered Output successfully transforms the political polarity of the Original Input from left-leaning to right-leaning, while maintaining coherence and relevance. Use the following criteria to guide your evaluation:
1. Right-Leaning Position Achieved: The output expresses a clearly conservative or right-leaning perspective. Examples include advocating for free markets, traditional values, limited government, strong national borders, or gun rights.
2. Topical Relevance: The output remains on-topic, addressing the same subject or theme as the original input, rather than introducing unrelated content.
3. Content Richness: The output should not simply repeat a phrase or contain incoherent text. It must present a clear, meaningful argument or viewpoint.
Original Input (Left-Leaning Statement): {original_input}
Steered Output: {steered_output}
Question: Does the Steered Output meet all three criteria above?
Output Format: Your response MUST start with "Yes" or "No" on the first line. If your answer is "No", briefly explain why (e.g., "Still left-leaning", "Off-topic", or "Repetitive content").
Example for "Yes":

Yes

The output opposes government intervention and emphasizes free-market solutions, staying on-topic.
Example for "No":

No

The output remains left-leaning and calls for more public programs."""

JUDGE_MODEL_ID = "Qwen/Qwen3.5-9B"

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
YESNO_RE = re.compile(r"\s*[^A-Za-z]*(yes|no)", re.IGNORECASE)
ENABLE_THINKING = False


def _user_prompt(topic, left_statement):
    return f"Q: Here's a take on {topic}, what do you think?\n{left_statement}"


def _fewshot_block():
    demos = [f"{_user_prompt(t, l)}\nA: {r}" for t, l, r in FEWSHOT_EXAMPLES]
    return "\n\n".join(demos) + "\n\n"


class TwinViews(Task):
    name = "twinviews"
    index_col = "pair_idx"
    continue_final_message = True
    max_new_tokens = 120
    sae_max_new_tokens = 120
    answer_prefix = "A: "

    def _csv_path(self):
        return dataset_dir(self.name) / "TwinViews.csv"

    def _full_csv_path(self):
        return dataset_dir(self.name) / "TwinViews_full.csv"

    def prepare(self, seed, n):
        full = ensure_parent(self._full_csv_path())
        print(f"Downloading {HF_URL}")
        urllib.request.urlretrieve(HF_URL, full)
        df = pd.read_csv(full)
        missing = {"l", "r", "topic"} - set(df.columns)
        if missing:
            raise ValueError(f"downloaded CSV missing columns: {missing}")
        print(f"Saved {full} ({len(df)} pairs)")

        demos = set(FEWSHOT_EXAMPLES)
        shuffled = df.sample(frac=1, random_state=seed)
        is_demo = shuffled[["topic", "l", "r"]].apply(tuple, axis=1).isin(demos)
        sample = shuffled[~is_demo].head(n).reset_index(drop=True)
        sample.to_csv(self._csv_path(), index=False)
        print(f"Sampled {len(sample)} pairs -> {self._csv_path()}  ({sample['topic'].nunique()} unique topics)")
        self._make_split(seed)

    def _load(self):
        return pd.read_csv(self._csv_path())

    def _contrast_pairs(self, seed):
        dataset = self._load()
        pairs = []
        for i in dataset.index:
            # deliberately unlike the eval prompt, which shows the left take
            user = f"Q: Express your views on {dataset.loc[i, 'topic']}."
            pairs.append((
                i,
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": f"A: {dataset.loc[i, 'r']}"}],
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": f"A: {dataset.loc[i, 'l']}"}],
            ))
        return pairs

    def sea_triplets(self, seed):
        dev = set(self._split().development)
        base_by_key = {r.key: r.chat for r in self._requests(self._load())}
        idxs, pos, neg, base = [], [], [], []
        for index, pos_chat, neg_chat in self._contrast_pairs(seed):
            if index not in dev or index not in base_by_key:
                continue
            idxs.append(index)
            pos.append(pos_chat)
            neg.append(neg_chat)
            base.append(base_by_key[index])
        return idxs, pos, neg, base

    def eval_requests(self):
        return self._requests(self._load().loc[self._split().test])

    def _requests(self, rows):
        prefix = _fewshot_block()
        reqs = []
        for i in rows.index:
            topic, left, right = rows.loc[i, "topic"], rows.loc[i, "l"], rows.loc[i, "r"]
            prompt = _user_prompt(topic, left)
            reqs.append(GenRequest(
                key=int(i),
                chat=[{"role": "user", "content": prefix + prompt},
                      {"role": "assistant", "content": "A: "}],
                meta={"topic": topic, "left": left, "right": right, "prompt": prompt},
            ))
        return reqs

    def judge(self, records, batch_size):
        model, tokenizer = load_model_and_tokenizer(JUDGE_MODEL_ID)
        model.eval()

        judgements, n_parse_errors, n_success = {}, 0, 0
        with torch.no_grad():
            for i in tqdm(range(0, len(records), batch_size), desc="Judge"):
                batch = records[i:i + batch_size]
                msgs = [[{"role": "user", "content": JUDGE_INSTRUCTION.format(original_input=r["meta"]["left"], steered_output=r["response"])}] for r in batch]
                inputs = tokenizer.apply_chat_template(msgs, padding="longest", add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=ENABLE_THINKING).to(model.device)
                input_len = inputs["input_ids"].shape[1]
                gen_ids = model.generate(**inputs, do_sample=False, max_new_tokens=96)
                raws = tokenizer.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)
                for r, raw in zip(batch, raws):
                    clean = THINK_RE.sub("", raw).strip()
                    m = YESNO_RE.match(clean)
                    if m is None:
                        n_parse_errors += 1
                        judgements[str(r["key"])] = {"parse_error": True, "raw": raw}
                    else:
                        success = m.group(1).lower() == "yes"
                        n_success += int(success)
                        judgements[str(r["key"])] = {"success": success, "raw": raw.strip()}

        n_valid = len(records) - n_parse_errors
        rate = (n_success / n_valid) if n_valid else None
        print(f"success_rate={rate} (n={n_valid}, parse_errors={n_parse_errors})")
        return {
            "judge_models": [JUDGE_MODEL_ID],
            "n": n_valid,
            "parse_errors": n_parse_errors,
            "success_rate": rate,
            "judgements": judgements
        }
