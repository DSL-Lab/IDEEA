import re
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common.config import CHARACTERS, MODEL_IDS, NEGATIVE, POSITIVE, STATES
from common.io import Split, save_json
from common.modeling import GenRequest, load_model_and_tokenizer
from common.paths import dataset_dir
from tasks import synthetic
from tasks.base import Task

SYSTEM_PROMPT = "Imagine you are a lab experiment participant. You will recive $5 for your participation, in addition to other money to be paid as a result of decisions made in the experiment. You will be anonymously paired with one other people, so that your decision may affect the payoffs of others, just as the decisions of the other people in your group may affect your payoffs."

PERSONA_WEAK = {
    "self_interest": "Imagine that you are a self-interested person.",
    "competitive": "Imagine that you are a competitive person.",
    "social_welfare": "Imagine that you are a social-welfare-oriented person.",
    "difference_aversion": "Imagine that you are an equality-minded person.",
}

PERSONA_STRONG = {
    "self_interest": "Imagine that you prioritize maximizing your own payoff.",
    "competitive": "Imagine that you prioritize maximizing your relative advantage over the other player.",
    "social_welfare": "Imagine that you prioritize maximizing the total payoff for everyone.",
    "difference_aversion": "Imagine that you prioritize minimizing the payoff gap between players.",
}

PERSONA_STRENGTHS = ("none", "weak", "strong")
_PERSONAS = {"weak": PERSONA_WEAK, "strong": PERSONA_STRONG}

PAYOFFS = range(0, 800, 200)

EVAL_REPS = 5

PREPARE_REPS = 10
PREPARE_TEMPERATURE = 1.0
PREPARE_TOP_K = 50
PREPARE_TOP_P = 1.0
PREPARE_MAX_NEW_TOKENS = 2000

FEATURES = ["self_interest", "competitive", "difference_aversion", "social_welfare"]


def dictator_prompt(pay_b1_a, pay_b1_b, pay_b2_a, pay_b2_b):
    return f"""In this period, you are person B. You may choose B1 or B2. Person A has no choice in this game.

If you choose B1, person A will receive {pay_b1_a} and you will receive {pay_b1_b}.

If you choose B2, person A will receive {pay_b2_a} and you will receive {pay_b2_b}.

You should first briefly reason step by step and put in <reason></reason>. After all reasoning, you give your answer but put it in <answer></answer> (B1 or B2 only)."""


def _persona_prompt(context, scenario):
    return f"{context}\n{scenario}"


def _persona_chat(row, scenario):
    return [{"role": "user", "content": _persona_prompt(row["context"], scenario)},
            {"role": "assistant", "content": row["response"]}]


INTERPRET_PROMPT = (
    "Based on the given response, which option (B1, B2, other) did the person "
    "choose?.\nAnswer should be one of (B1, B2, other)\n\nresponse: {response}"
)

INTERPRETER_MODEL_ID = "Qwen/Qwen3.5-9B"

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
#: match on the first label to appear
INTERPRET_RE = re.compile(r"\b(B1|B2|other)\b", re.IGNORECASE)
ENABLE_THINKING = False


def _clean_answer(text):
    return text if text in ("B1", "B2") else None


def _interpreted_answer(text):
    m = INTERPRET_RE.search(THINK_RE.sub("", text))
    return _clean_answer(m.group(1).upper()) if m else None


def _parse_answer(record):
    output = record["response"]
    answer = re.search(r"<answer>(.*?)</answer>", output)
    reason = re.search(r"<reason>(.*?)</reason>", output)
    return {
        **record["meta"],
        "output": output,
        "answer": _clean_answer(re.sub(r"\s+", "", answer.group(1).upper()) if answer else ""),
        "reason": reason.group(1).replace("\n", "\t").strip() if reason else None,
    }


def _payoff_features(df):
    b1a, b1b = df["pay_b1_a"], df["pay_b1_b"]
    b2a, b2b = df["pay_b2_a"], df["pay_b2_b"]
    df["self_interest"] = np.sign(b1b - b2b)                            # larger own payoff
    df["competitive"] = np.sign((b1b - b1a) - (b2b - b2a))              # larger lead
    df["difference_aversion"] = np.sign(-abs(b1b - b1a) + abs(b2b - b2a))  # smaller gap
    df["social_welfare"] = np.sign(abs(b1b + b1a) - abs(b2b + b2a))     # larger total
    df["answer_sign"] = (df["answer"] == "B1").astype(int) - (df["answer"] == "B2").astype(int)
    return df


class DictatorGame(Task):
    name = "dictatorgame"
    index_col = "sample_idx"
    continue_final_message = False
    max_new_tokens = 500
    sae_max_new_tokens = 256
    sample_temperature = 0.2

    def __init__(self, model, character, persona_strength="none"):
        super().__init__(character)
        self.model = model
        if persona_strength not in PERSONA_STRENGTHS:
            raise ValueError(f"persona_strength must be one of {list(PERSONA_STRENGTHS)}")
        #: evaluation only, and only meaningful unsteered: it names the prompted
        #: baseline, so nothing upstream of eval_requests reads it
        self.persona_strength = persona_strength

    def _system_prompt(self):
        if self.persona_strength == "none":
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n{_PERSONAS[self.persona_strength][self.character]}"

    def _dataset_csv(self):
        return dataset_dir(self.name) / self.model / "dataset.csv"

    def _split_path(self):
        return self._dataset_csv().parent / "split.json"

    def prepare(self, seed, batch_size):
        df = pd.DataFrame([
            {"character": character, "state": state, "context": synthetic.contexts[character][state], "scenario": scenario}
            for character in CHARACTERS
            for state in STATES
            for scenario in synthetic.scenarios[character]
            for _ in range(PREPARE_REPS)
        ])
        prompts = [[{"role": "user", "content": _persona_prompt(r["context"], r["scenario"])}]
                   for _, r in df.iterrows()]

        lm, tokenizer = load_model_and_tokenizer(MODEL_IDS[self.model])
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling responses"):
            inputs = tokenizer.apply_chat_template(
                prompts[i:i + batch_size], add_generation_prompt=True,
                return_tensors="pt", padding="longest", return_dict=True).to(lm.device)
            # do_sample explicitly: models publishing no do_sample would otherwise emit
            # PREPARE_REPS identical greedy responses per (character, state, scenario)
            out = lm.generate(**inputs, max_new_tokens=PREPARE_MAX_NEW_TOKENS, do_sample=True,
                              temperature=PREPARE_TEMPERATURE, top_p=PREPARE_TOP_P, top_k=PREPARE_TOP_K)
            decoded = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            responses.extend(decoded)   # verbatim: stripping would retokenize
        df["response"] = responses

        path = self._dataset_csv()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df)} rows)")
        self._make_split(seed)
        return df

    def sae_seeds(self, requests, seed):
        """Reps of one payoff row share a prompt, so a single seed would return the
        identical completion for each."""
        return [r.meta["rep"] for r in requests]

    def _load(self):
        df = pd.read_csv(self._dataset_csv())
        df = df[df["character"] == self.character].reset_index(drop=True)
        df["sample_idx"] = np.arange(len(df), dtype=int)
        return df

    def _make_split(self, seed):
        """development = every row. Row counts are equal across characters, so one
        positional split serves all four."""
        counts = pd.read_csv(self._dataset_csv())["character"].value_counts().to_dict()
        assert set(counts) == set(CHARACTERS), f"unexpected characters: {set(counts)}"
        if len(set(counts.values())) != 1:
            raise ValueError(f"characters have unequal row counts: {counts}; a shared positional split would be wrong")
        n = next(iter(counts.values()))
        idx_train, idx_val = train_test_split(np.arange(n, dtype=int), train_size=0.8, random_state=seed)
        split = Split(idx_train.tolist(), idx_val.tolist(), list(range(n)), [])
        save_json(self._split_path(), split.to_dict())
        print(f"Saved {self._split_path()}  |  {split.summary()} (eval is a payoff grid, so test is empty)")
        return split

    def contrast_rows(self, kind, seed):
        df = self._load()
        chats = [_persona_chat(r, r["scenario"]) for _, r in df.iterrows()]
        labels = df["state"].astype(int).tolist()
        indices = df["sample_idx"].astype(int).tolist()
        return chats, labels, indices

    def _contrast_pairs(self, seed):
        """Positives and negatives come from different context prompts, so this pairs the
        i-th of each within a scenario rather than pairing on a shared prompt."""
        df = self._load()
        pairs = []
        for scenario, group in df.groupby("scenario", sort=True):
            pos = group[group["state"] == POSITIVE].reset_index(drop=True)
            neg = group[group["state"] == NEGATIVE].reset_index(drop=True)
            pairs += [(int(pos.loc[i, "sample_idx"]),
                       _persona_chat(pos.loc[i], scenario),
                       _persona_chat(neg.loc[i], scenario))
                      for i in range(min(len(pos), len(neg)))]
        return pairs

    def sea_triplets(self, seed):
        df = self._load().set_index("sample_idx")
        idxs, pos, neg, base = [], [], [], []
        for index, pos_chat, neg_chat in self._contrast_pairs(seed):
            idxs.append(index)
            pos.append(pos_chat)
            neg.append(neg_chat)
            base.append([{"role": "user", "content": df.loc[index, "scenario"]}])
        return idxs, pos, neg, base

    def eval_requests(self):
        return [
            GenRequest(
                key=f"{b1a}_{b1b}_{b2a}_{b2b}_rep{rep}",
                chat=[{"role": "system", "content": self._system_prompt()},
                      {"role": "user", "content": dictator_prompt(b1a, b1b, b2a, b2b)}],
                meta={"rep": rep, "pay_b1_a": b1a, "pay_b1_b": b1b, "pay_b2_a": b2a, "pay_b2_b": b2b},
            )
            for b1a, b1b, b2a, b2b in product(PAYOFFS, repeat=4)
            if (b1a, b1b) != (b2a, b2b)
            for rep in range(EVAL_REPS)
        ]

    def _interpret(self, outputs, batch_size):
        print(f"Interpreting {len(outputs)} malformed answers with {INTERPRETER_MODEL_ID}")
        model, tokenizer = load_model_and_tokenizer(INTERPRETER_MODEL_ID)
        chats = [[{"role": "user", "content": INTERPRET_PROMPT.format(response=out)}] for out in outputs]
        answers = []
        with torch.no_grad():
            for i in tqdm(range(0, len(chats), batch_size), desc="Interpreting"):
                inputs = tokenizer.apply_chat_template(chats[i:i + batch_size], add_generation_prompt=True, return_dict=True, return_tensors="pt", padding="longest", enable_thinking=ENABLE_THINKING).to(model.device)
                out_ids = model.generate(**inputs, do_sample=False, max_new_tokens=16)
                input_len = inputs["input_ids"].shape[-1]
                answers += [_interpreted_answer(t) for t in tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)]
        return answers

    def judge(self, records, batch_size):
        parsed = [_parse_answer(r) for r in records]
        malformed = [i for i, p in enumerate(parsed) if p["answer"] is None]

        n_none = 0
        if malformed:
            answers = self._interpret([parsed[i]["output"] for i in malformed], batch_size)
            for i, answer in zip(malformed, answers):
                parsed[i]["answer"] = answer
            n_none = sum(a is None for a in answers)

        df = _payoff_features(pd.DataFrame(parsed))
        print(f"Processed: {len(df)}, interpreted: {len(malformed)}, still None: {n_none}")

        fit = sm.OLS.from_formula("answer_sign ~ " + " + ".join(FEATURES), df).fit()
        coef = {f: {"val": float(fit.params[f]), "se": float(fit.bse[f])}
                for f in ["Intercept"] + FEATURES}
        coef["intercept"] = coef.pop("Intercept")
        for k, v in coef.items():
            print(f"  {k:22s} {v['val']:+.4f} (se {v['se']:.4f})")

        return {
            "judge_models": [INTERPRETER_MODEL_ID],
            "n": len(df),
            "unparsed": n_none,
            "interpreted": len(malformed),
            "coef": coef,
            "answers": df[["rep", "pay_b1_a", "pay_b1_b", "pay_b2_a", "pay_b2_b", "answer"]].to_dict(orient="records")
        }
