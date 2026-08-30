from abc import ABC, abstractmethod

import numpy as np

from common.config import NEGATIVE, POSITIVE
from common.io import Split, load_json, save_json
from common.paths import dataset_dir


def build_split(n, seed):
    """50% development / 50% test; development is 80/20 train/val, fold 0 as val."""
    idx = np.arange(n, dtype=int)
    np.random.seed(seed)
    np.random.shuffle(idx)

    development_set, test_set = np.array_split(idx, 2)
    development_folds = np.array_split(development_set, 5)
    val_set = np.concatenate(development_folds[:1])
    train_set = np.concatenate(development_folds[1:])
    development_set = np.concatenate(development_folds)

    return Split(train_set.tolist(), val_set.tolist(), development_set.tolist(), test_set.tolist())


class Task(ABC):
    name = None
    index_col = "sample_idx"
    #: True when eval prompts end in a partial assistant turn the model continues
    #: ("A: " style); False when a generation prompt should be appended instead.
    continue_final_message = True
    answer_prefix = None
    sample_temperature = None

    def __init__(self, character=None):
        self.character = character

    @abstractmethod
    def prepare(self, seed):
        ...

    @abstractmethod
    def _load(self):
        ...

    def _split_path(self):
        return dataset_dir(self.name) / "split.json"

    def _make_split(self, seed):
        split = build_split(len(self._load()), seed)
        save_json(self._split_path(), split.to_dict())
        print(f"Saved {self._split_path()}  |  {split.summary()}")
        return split

    def _split(self):
        p = self._split_path()
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `prepare_dataset --task {self.name}` first.")
        return Split.from_dict(load_json(p))

    def contrast_rows(self, kind, seed):
        chats, labels, indices = [], [], []
        for index, pos_chat, neg_chat in self._contrast_pairs(seed):
            chats.extend([pos_chat, neg_chat])
            labels.extend([POSITIVE, NEGATIVE])
            indices.extend([index, index])
        return chats, labels, indices

    @abstractmethod
    def _contrast_pairs(self, seed):
        """One (index, positive_chat, negative_chat) per row, differing only in the
        behaviour being contrasted."""

    @abstractmethod
    def sea_triplets(self, seed):
        """positive, negative and a neutral base, aligned per index, over development."""

    def dev_rows(self, df):
        dev = set(self._split().development)
        return df[df[self.index_col].isin(dev)].reset_index(drop=True)

    def probe_indices(self, df):
        """Indices are positional into dev_df, not dataset ids."""
        split = self._split()
        dev_df = df[df[self.index_col].isin(set(split.development))].reset_index(drop=True)
        idx_train = np.where(dev_df[self.index_col].isin(set(split.train)))[0].astype(int)
        idx_val = np.where(dev_df[self.index_col].isin(set(split.val)))[0].astype(int)
        return dev_df, idx_train, idx_val

    @abstractmethod
    def eval_requests(self):
        """The task's fixed evaluation set: the test split, or the whole prompt set for
        tasks that hold one out separately."""

    def sae_seeds(self, requests, seed):
        return [seed] * len(requests)

    def parse(self, text):
        text = text.strip()
        if self.answer_prefix and text.startswith(self.answer_prefix):
            text = text[len(self.answer_prefix):].strip()
        return text

    @abstractmethod
    def judge(self, records, batch_size):
        """The judge and its prompts are fixed task protocol; batch_size is the only knob."""

    def __repr__(self):
        return f"<Task {self.name}{'/' + self.character if self.character else ''}>"
