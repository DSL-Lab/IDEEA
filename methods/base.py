from abc import ABC, abstractmethod

import numpy as np

from common.io import save_json
from common.paths import directions_path


class Steerer:
    def __init__(self, handles, info):
        self.handles = list(handles)
        self.info = info

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class SteeringMethod(ABC):
    name = None

    @abstractmethod
    def path_segments(self):
        """Segments under {model}/ identifying this configuration."""

    def _directions_segments(self):
        """IDEEA overrides this so its three fixed-nc variants share one file."""
        return self.path_segments()


class FitMethod(SteeringMethod):
    @abstractmethod
    def fit(self, task, model_name):
        """Fit and save; returns the saved result."""

    def _save_directions(self, task, model_name, result):
        path = directions_path(task.name, model_name, self._directions_segments(), task.character)
        save_json(path, result)
        print(f"Saved {path}")
        return result


class EvalMethod(SteeringMethod):
    @abstractmethod
    def run_id(self):
        """Names the generations file within path_segments()."""


class HookedMethod(EvalMethod):
    """SAE is not one of these: it generates remotely and supplies a generator."""

    @abstractmethod
    def install(self, model, task, model_name):
        """Attach hooks; returns a Steerer that removes them."""


def top_k_mask(sorted_idx, k, shape):
    mask = np.zeros(shape)
    mask[np.asarray(sorted_idx[0])[:k], np.asarray(sorted_idx[1])[:k]] = 1
    return mask
