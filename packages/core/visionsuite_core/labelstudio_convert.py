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
