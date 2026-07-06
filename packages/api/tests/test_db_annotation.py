from sqlmodel import Session, select

from visionsuite_api.db import Annotation, Dataset, init_db, make_engine


def test_dataset_ls_fields_and_annotation():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        ds = Dataset(name="d", task="detection", ls_project_id=7, class_names='["car"]')
        s.add(ds); s.commit(); s.refresh(ds)
        s.add(Annotation(dataset_id=ds.id, image_id="abc", coco_json="[]", n_objects=0))
        s.commit()
    with Session(engine) as s:
        ds = s.exec(select(Dataset)).one()
        ann = s.exec(select(Annotation)).one()
    assert ds.ls_project_id == 7 and ds.class_names == '["car"]'
    assert ann.image_id == "abc" and ann.n_objects == 0
