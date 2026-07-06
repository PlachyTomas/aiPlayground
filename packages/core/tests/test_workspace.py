from visionsuite_core import workspace


def test_root_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    assert workspace.workspace_root() == tmp_path / "ws"


def test_ensure_creates_subdirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    root = workspace.ensure_workspace()
    for sub in ("datasets", "runs", "models", "exports"):
        assert (root / sub).is_dir()


def test_run_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    d = workspace.run_dir("abc123")
    assert d.is_dir() and d.name == "abc123" and d.parent.name == "runs"
