from typing import Protocol, runtime_checkable

from .types import CompatVerdict, VisionTask


@runtime_checkable
class ModelAdapter(Protocol):
    hf_id: str
    task: VisionTask

    def mps_compat(self) -> CompatVerdict: ...


class DummyAdapter:
    hf_id = "dummy/echo"
    task = VisionTask.CLASSIFICATION

    def mps_compat(self) -> CompatVerdict:
        return CompatVerdict.KNOWN_GOOD_MPS


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.hf_id] = adapter

    def get(self, hf_id: str) -> ModelAdapter:
        return self._adapters[hf_id]

    def list(self) -> list[ModelAdapter]:
        return list(self._adapters.values())


default_registry = ModelRegistry()
default_registry.register(DummyAdapter())
