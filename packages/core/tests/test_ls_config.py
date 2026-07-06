from visionsuite_core.labelstudio_convert import ls_config_for
from visionsuite_core.types import VisionTask


def test_detection_config():
    xml = ls_config_for(VisionTask.DETECTION, ["car", "person"])
    assert 'RectangleLabels name="label" toName="image"' in xml
    assert '<Label value="car"/>' in xml and '<Label value="person"/>' in xml


def test_classification_config_accepts_string_task():
    xml = ls_config_for("classification", ["cat", "dog"])
    assert 'Choices name="choice" toName="image"' in xml
    assert '<Choice value="cat"/>' in xml and 'choice="single-radio"' in xml
