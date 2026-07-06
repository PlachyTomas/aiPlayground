from sqlmodel import Session

from visionsuite_api.db import Image, init_db, make_engine


def test_image_has_ingestion_fields():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        s.add(Image(dataset_id=1, image_id="abc123", filename="abc123.png",
                    width=64, height=48, source="folder",
                    path="datasets/1/images/abc123.png",
                    thumb_path="datasets/1/thumbs/abc123.webp"))
        s.commit()
    with Session(engine) as s:
        got = s.exec(__import__("sqlmodel").select(Image)).one()
    assert got.image_id == "abc123" and got.source == "folder" and got.width == 64
