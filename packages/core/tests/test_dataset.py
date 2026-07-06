import pytest
from visionsuite_core.dataset import Dataset, ImageRecord
from visionsuite_core.types import VisionTask


def _sample() -> Dataset:
    return Dataset(
        dataset_id="d1",
        task=VisionTask.CLASSIFICATION,
        class_names=["cat", "dog"],
        images=[ImageRecord(image_id="i1", path="i1.jpg", width=64, height=48)],
    )


def test_to_coco_shape():
    coco = _sample().to_coco()
    assert {"images", "categories", "annotations"} <= coco.keys()
    assert coco["images"][0]["file_name"] == "i1.jpg"
    assert [c["name"] for c in coco["categories"]] == ["cat", "dog"]


def test_from_coco_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        Dataset.from_coco({})


def test_from_label_studio_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        Dataset.from_label_studio([])
