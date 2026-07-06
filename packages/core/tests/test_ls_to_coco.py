import math

from visionsuite_core.labelstudio_convert import image_id_from_task, ls_json_to_coco


def _det_task(image_id, results):
    return {"data": {"image": f"/data/local-files/?d=datasets/1/images/{image_id}.png"},
            "annotations": [{"result": results}]}


def test_image_id_from_task():
    t = _det_task("abcd1234", [])
    assert image_id_from_task(t) == "abcd1234"


def test_detection_percent_to_pixels():
    res = [{"type": "rectanglelabels", "original_width": 200, "original_height": 100,
            "image_rotation": 0,
            "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0,
                      "rectanglelabels": ["car"]}}]
    coco = ls_json_to_coco([_det_task("img1", res)], ["car", "person"])
    assert coco["categories"] == [{"id": 0, "name": "car"}, {"id": 1, "name": "person"}]
    box = coco["images"][0]["annotations"][0]
    assert box["category_id"] == 0
    assert box["bbox"] == [20.0, 20.0, 60.0, 40.0]  # x=10%*200, y=20%*100, w=30%*200, h=40%*100


def test_rotation_makes_enclosing_box_larger():
    res = [{"type": "rectanglelabels", "original_width": 100, "original_height": 100,
            "image_rotation": 0,
            "value": {"x": 40, "y": 40, "width": 20, "height": 20, "rotation": 45,
                      "rectanglelabels": ["car"]}}]
    coco = ls_json_to_coco([_det_task("i", res)], ["car"])
    x, y, w, h = coco["images"][0]["annotations"][0]["bbox"]
    assert w > 20 and h > 20  # rotated 20x20 encloses larger than axis-aligned
    assert math.isclose(w, h, rel_tol=0.05)


def test_classification():
    t = {"data": {"image": "/data/local-files/?d=datasets/1/images/z.png"},
         "annotations": [{"result": [{"type": "choices", "value": {"choices": ["dog"]}}]}]}
    coco = ls_json_to_coco([t], ["cat", "dog"])
    img = coco["images"][0]
    assert img["classification"] == 1 and img["annotations"] == []


def test_unannotated_skipped():
    t = {"data": {"image": "/data/local-files/?d=x/y.png"}, "annotations": []}
    assert ls_json_to_coco([t], ["a"])["images"] == []
