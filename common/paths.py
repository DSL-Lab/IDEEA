from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def dataset_dir(task):
    return RESULTS / task / "dataset"


def model_dir(task, model, character=None):
    base = RESULTS / task / model
    return base / character if character else base


def activations_path(task, model, kind, character=None):
    return model_dir(task, model, character) / "activations" / f"{kind}.pkl"


def directions_dir(task, model, segments, character=None):
    return model_dir(task, model, character) / "directions" / Path(*segments)


def directions_path(task, model, segments, character=None):
    return directions_dir(task, model, segments, character) / "directions.json"


def eval_path(task, model, segments, run_id, character=None):
    return model_dir(task, model, character) / "eval" / Path(*segments) / f"{run_id}.json"


def judged_path(generations_path):
    p = Path(generations_path)
    return p.with_name(f"{p.stem}.judged.json")


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)
