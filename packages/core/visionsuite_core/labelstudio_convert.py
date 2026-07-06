from __future__ import annotations


def _task_str(task) -> str:
    return task.value if hasattr(task, "value") else str(task)


def ls_config_for(task, class_names: list[str]) -> str:
    if _task_str(task) == "detection":
        labels = "\n".join(f'    <Label value="{c}"/>' for c in class_names)
        return (
            '<View>\n  <Image name="image" value="$image"/>\n'
            '  <RectangleLabels name="label" toName="image">\n'
            f"{labels}\n  </RectangleLabels>\n</View>"
        )
    choices = "\n".join(f'    <Choice value="{c}"/>' for c in class_names)
    return (
        '<View>\n  <Image name="image" value="$image"/>\n'
        '  <Choices name="choice" toName="image" choice="single-radio">\n'
        f"{choices}\n  </Choices>\n</View>"
    )


import math
from pathlib import Path
from urllib.parse import unquote


def image_id_from_task(task: dict) -> str:
    raw = str(task.get("data", {}).get("image", ""))
    rel = raw.split("d=")[-1] if "d=" in raw else raw
    return Path(unquote(rel)).stem


def _enclosing_bbox(x, y, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    pts = [(x + cx * ca - cy * sa, y + cx * sa + cy * ca) for cx, cy in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def ls_json_to_coco(tasks: list[dict], class_names: list[str]) -> dict:
    cat_id = {name: i for i, name in enumerate(class_names)}
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images = []
    for task in tasks:
        anns = task.get("annotations") or []
        results = anns[0].get("result", []) if anns else []
        if not results:
            continue
        boxes: list[dict] = []
        classification = None
        for r in results:
            rtype = r.get("type")
            val = r.get("value", {})
            if rtype == "rectanglelabels":
                name = (val.get("rectanglelabels") or [None])[0]
                if name not in cat_id:
                    continue
                ow, oh = r.get("original_width", 0), r.get("original_height", 0)
                x = val.get("x", 0) / 100 * ow
                y = val.get("y", 0) / 100 * oh
                w = val.get("width", 0) / 100 * ow
                h = val.get("height", 0) / 100 * oh
                rot = val.get("rotation", 0) or r.get("image_rotation", 0)
                bbox = _enclosing_bbox(x, y, w, h, rot) if rot else [x, y, w, h]
                boxes.append({"bbox": bbox, "category_id": cat_id[name]})
            elif rtype == "choices":
                name = (val.get("choices") or [None])[0]
                if name in cat_id:
                    classification = cat_id[name]
        images.append({"image_id": image_id_from_task(task), "annotations": boxes,
                       "classification": classification})
    return {"categories": categories, "images": images}
