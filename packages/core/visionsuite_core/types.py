from enum import Enum


class VisionTask(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"


class CompatVerdict(str, Enum):
    KNOWN_GOOD_MPS = "known_good_mps"
    UNTESTED = "untested"
    UNSUPPORTED = "unsupported"
