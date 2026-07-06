from sqlmodel import Session

from visionsuite_api.db import Run, init_db, make_engine


def test_run_roundtrip():
    engine = make_engine("sqlite://")  # in-memory
    init_db(engine)
    with Session(engine) as s:
        s.add(Run(id="r1", model_id="dummy/echo", dataset_id="d1"))
        s.commit()
    with Session(engine) as s:
        got = s.get(Run, "r1")
    assert got is not None and got.status == "pending"


def test_default_db_lives_under_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path))
    engine = make_engine()  # default URL derives from VISIONSUITE_WORKSPACE
    init_db(engine)
    db_path = (tmp_path / "db.sqlite").resolve()
    assert db_path.exists()
    assert str(engine.url).endswith(str(db_path))
