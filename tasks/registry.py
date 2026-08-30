from tasks.dictatorgame import DictatorGame
from tasks.toxicity import Toxicity
from tasks.truthfulqa import TruthfulQA
from tasks.twinviews import TwinViews

_TASKS = {
    t.name: t for t in (TruthfulQA, TwinViews, Toxicity, DictatorGame)
}

TASK_NAMES = tuple(_TASKS)


def get_task(name, model=None, character=None, persona_strength="none"):
    cls = _TASKS[name]
    if cls is DictatorGame:
        if model is None or character is None:
            raise ValueError("dictatorgame requires --model and --character")
        return cls(model, character, persona_strength)
    return cls()
