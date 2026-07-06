from visionsuite_core.registry import ModelAdapter, DummyAdapter, ModelRegistry, default_registry
from visionsuite_core.types import CompatVerdict, VisionTask


def test_dummy_is_adapter():
    assert isinstance(DummyAdapter(), ModelAdapter)


def test_dummy_verdict():
    assert DummyAdapter().mps_compat() == CompatVerdict.KNOWN_GOOD_MPS
    assert DummyAdapter().task == VisionTask.CLASSIFICATION


def test_registry_register_and_get():
    reg = ModelRegistry()
    reg.register(DummyAdapter())
    assert reg.get("dummy/echo").hf_id == "dummy/echo"
    assert [a.hf_id for a in reg.list()] == ["dummy/echo"]


def test_default_registry_has_dummy():
    assert default_registry.get("dummy/echo") is not None
