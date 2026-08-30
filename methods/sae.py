import os
import time

import requests
from tqdm import tqdm

from methods.base import EvalMethod

NEURONPEDIA_BASE = "https://www.neuronpedia.org"
MAX_RETRY = 5
CHECKPOINT_EVERY = 10


def _api_key():
    """Read at call time so the other methods stay usable without an SAE key set."""
    return os.environ["NEURONPEDIA_API_KEY"]

#: hand-audited per (task, model[, character]); see the README on SAE selection
SAE_FEATURES = {
    ("twinviews", "llama3_8b"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "19-resid-post-aa", "sae_index": 18918},
    ("truthfulqa", "llama3_8b"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "3-resid-post-aa", "sae_index": 125782},
    ("truthfulqa", "gemma2_9b"): {
        "neuronpedia_model_id": "gemma-2-9b-it",
        "sae_layer": "31-gemmascope-res-16k", "sae_index": 3613},
    ("truthfulqa", "gemma2_2b"): {
        "neuronpedia_model_id": "gemma-2-2b-it",
        "sae_layer": "20-axbench-reft-r1-res-16k", "sae_index": 4320},
    ("truthfulqa", "qwen2.5_7b"): {
        "neuronpedia_model_id": "qwen2.5-7b-it",
        "sae_layer": "15-resid-post-aa", "sae_index": 117121},
    ("toxicity", "llama3_8b"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "11-resid-post-aa", "sae_index": 34614},
    ("dictatorgame", "llama3_8b", "self_interest"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "27-resid-post-aa", "sae_index": 2141},
    ("dictatorgame", "llama3_8b", "competitive"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "7-resid-post-aa", "sae_index": 2451},
    ("dictatorgame", "llama3_8b", "social_welfare"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "7-resid-post-aa", "sae_index": 79612},
    ("dictatorgame", "llama3_8b", "difference_aversion"): {
        "neuronpedia_model_id": "llama3.1-8b-it",
        "sae_layer": "11-resid-post-aa", "sae_index": 75406},
}


def lookup_feature(task_name, model_name, character):
    keys = [(task_name, model_name)]
    if character:
        keys.insert(0, (task_name, model_name, character))
    for key in keys:
        if key in SAE_FEATURES:
            return SAE_FEATURES[key]
    return None


def resolve_sae_feature(task, model_name):
    feature = lookup_feature(task.name, model_name, task.character)
    if feature is None:
        raise ValueError(f"No SAE_FEATURES entry for ({task.name}, {model_name}, {task.character}); add one after auditing the feature.")
    return feature


def get_max_act_approx(neuronpedia_model_id, sae_layer, sae_index):
    url = f"{NEURONPEDIA_BASE}/api/feature/{neuronpedia_model_id}/{sae_layer}/{sae_index}"
    r = requests.get(url, headers={"x-api-key": _api_key()})
    r.raise_for_status()
    return float(r.json()["maxActApprox"])


class PromptRejected(Exception):
    ...


def steer_chat(neuronpedia_model_id, sae_layer, sae_index, strength, chat, seed, n_tokens, temperature):
    feature = {"modelId": neuronpedia_model_id, "layer": sae_layer, "index": sae_index, "strength": strength}
    for attempt in range(MAX_RETRY):
        response = requests.post(
            f"{NEURONPEDIA_BASE}/api/steer-chat",
            headers={"Content-Type": "application/json", "x-api-key": _api_key()},
            json={
                "defaultChatMessages": chat,
                "steeredChatMessages": chat,
                "modelId": neuronpedia_model_id,
                "features": [feature],
                "temperature": temperature,
                "n_tokens": n_tokens,
                "freq_penalty": 1,
                "seed": seed,
                "strength_multiplier": 1,
                "steer_special_tokens": True,
                "steer_method": "SIMPLE_ADDITIVE"
            },
        )
        if response.status_code == 200:
            return response.json()["STEERED"]["chatTemplate"][-1]["content"]
        if response.status_code == 429:
            print("rate limited; sleeping 1.2h")
            time.sleep(1.2 * 3600)
            continue
        # 400 -> prompt too long
        if response.status_code == 400:
            raise PromptRejected(f"400 {response.text[:200]}")
        print(f"attempt {attempt} got {response.status_code} {response.text[:200]}")
        if attempt + 1 < MAX_RETRY:
            print(f"sleeping {attempt + 1}min")
            time.sleep((attempt + 1) * 60)
    raise RuntimeError("Error getting Neuronpedia results")


class NeuronpediaGenerator:
    """Drop-in replacement for LocalGenerator backed by the Neuronpedia API."""

    def __init__(self, feature, strength, max_new_tokens, temperature, seeds, resume, on_progress, skipped):
        self.feature = feature
        self.strength = strength
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        #: one seed per chat; repetitions of the same prompt must differ, or the
        #: API returns the identical completion for each
        self.seeds = seeds
        self.resume = resume or []
        self.on_progress = on_progress
        self.skipped = set(skipped)

    def generate(self, chats):
        answers = list(self.resume) + [None] * max(0, len(chats) - len(self.resume))
        answers = answers[:len(chats)]
        n_done = sum(1 for a in answers if a is not None)
        if n_done or self.skipped:
            print(f"Resuming from {n_done}/{len(chats)} ({len(self.skipped)} skipped)")
        for i, chat in enumerate(tqdm(chats, desc="SAE generation")):
            if answers[i] is not None or i in self.skipped:
                continue
            try:
                answers[i] = steer_chat(self.feature["neuronpedia_model_id"], self.feature["sae_layer"], self.feature["sae_index"], self.strength, chat, self.seeds[i], self.max_new_tokens, self.temperature)
            except PromptRejected as exc:
                self.skipped.add(i)
                print(f"skipping prompt {i} ({sum(len(m['content']) for m in chat)} chars): {exc}")
            if (i + 1) % CHECKPOINT_EVERY == 0:
                self.on_progress(answers, sorted(self.skipped))
        self.on_progress(answers, sorted(self.skipped))
        return answers


class SAE(EvalMethod):
    name = "sae"

    def __init__(self, feature, multiplier):
        self.feature = feature
        self.multiplier = multiplier

    def path_segments(self):
        return ("sae", f"layer_{self.feature['sae_layer']}_idx_{self.feature['sae_index']}")

    def run_id(self):
        return f"mul_{self.multiplier}"

    def make_generator(self, max_new_tokens, temperature, seeds, resume, on_progress, skipped):
        #: anchoring to the feature's own maxActApprox is what makes one multiplier
        #: comparable across models whose activation scales differ by orders of magnitude
        max_act = get_max_act_approx(self.feature["neuronpedia_model_id"], self.feature["sae_layer"], self.feature["sae_index"])
        strength = self.multiplier * max_act
        print(f"maxActApprox={max_act:.4f}  multiplier={self.multiplier}  strength={strength:.4f}")
        gen = NeuronpediaGenerator(self.feature, strength, max_new_tokens, temperature, seeds, resume, on_progress, skipped)
        gen.info = {"feature": self.feature, "sae_strength": strength, "max_act_approx": max_act, "multiplier": self.multiplier}
        return gen
