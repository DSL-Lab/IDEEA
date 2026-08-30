from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@dataclass
class GenRequest:
    """`meta` is copied verbatim into the saved record for the judge to read."""
    key: object
    chat: list
    meta: dict


class LocalGenerator:
    """do_sample is set from temperature rather than left to the model: mistral_7b and
    both gemma2 models publish no do_sample, so a bare temperature would be dropped and
    every repetition would come back identical."""

    #: a local model answers every prompt it is given; only the Neuronpedia API can
    #: refuse one, so this sits on both generators rather than being probed for
    skipped = ()

    def __init__(self, model, tokenizer, continue_final_message, batch_size,
                 max_new_tokens, temperature):
        self.model = model
        self.tokenizer = tokenizer
        self.continue_final_message = continue_final_message
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _encode(self, chats):
        if self.continue_final_message:
            return self.tokenizer.apply_chat_template(chats, padding="longest", continue_final_message=True, return_dict=True, return_tensors="pt")
        return self.tokenizer.apply_chat_template(chats, padding="longest", add_generation_prompt=True, return_dict=True, return_tensors="pt")

    def generate(self, chats):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(chats), self.batch_size), desc="Generation"):
                inputs = self._encode(chats[i:i + self.batch_size]).to(self.model.device)
                input_len = inputs["input_ids"].shape[1]
                gen_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature is not None, temperature=self.temperature)
                out.extend(self.tokenizer.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True))
        return out
