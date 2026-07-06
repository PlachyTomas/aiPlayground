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
