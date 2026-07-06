from __future__ import annotations

from dataclasses import dataclass, field

from .types import VisionTask


@dataclass
class ImageRecord:
    image_id: str
    path: str
    width: int
    height: int


@dataclass
class Dataset:
    dataset_id: str
    task: VisionTask
    class_names: list[str] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)

    def to_coco(self) -> dict:
        return {
            "images": [
                {"id": i, "file_name": im.path, "width": im.width, "height": im.height}
                for i, im in enumerate(self.images)
            ],
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(self.class_names)
            ],
            "annotations": [],
        }

    @classmethod
    def from_coco(cls, coco: dict) -> "Dataset":
        raise NotImplementedError("lands in Sub-project 1")

    @classmethod
    def from_label_studio(cls, tasks: list) -> "Dataset":
        raise NotImplementedError("lands in Sub-project 2")
