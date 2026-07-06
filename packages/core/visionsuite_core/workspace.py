import os
from pathlib import Path

_SUBDIRS = ("datasets", "runs", "models", "exports")


def workspace_root() -> Path:
    return Path(os.environ.get("VISIONSUITE_WORKSPACE", "./workspace")).resolve()


def ensure_workspace() -> Path:
    root = workspace_root()
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _child(sub: str, name: str) -> Path:
    d = ensure_workspace() / sub / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_dir(dataset_id: str) -> Path:
    return _child("datasets", dataset_id)


def run_dir(run_id: str) -> Path:
    return _child("runs", run_id)


def model_dir(model_id: str) -> Path:
    return _child("models", model_id)


def export_dir(model_id: str) -> Path:
    return _child("exports", model_id)
