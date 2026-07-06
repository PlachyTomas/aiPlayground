import imageio.v3 as iio
import numpy as np

from visionsuite_api.db import init_db, make_engine
from visionsuite_api.imports import video_producer


async def test_video_producer_deletes_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    engine = make_engine("sqlite://")
    init_db(engine)
    ds_id = 1

    vid = tmp_path / "clip.mp4"
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in (10, 40, 70, 100)]
    iio.imwrite(vid, np.stack(frames), fps=4, codec="libx264")
    assert vid.exists()

    async for _ in video_producer(engine, ds_id, str(vid), every_n=1):
        pass

    assert not vid.exists()
