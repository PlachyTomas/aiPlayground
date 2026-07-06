from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Protocol


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunSpec:
    run_id: str
    model_id: str
    dataset_id: str
    hyperparams: dict = field(default_factory=dict)


@dataclass
class RunEvent:
    type: str
    message: str = ""
    progress: float | None = None
    status: RunStatus | None = None


class TrainingBackend(Protocol):
    def stream(self, spec: RunSpec) -> AsyncIterator[RunEvent]: ...


class LocalBackend:
    steps = 5

    async def stream(self, spec: RunSpec) -> AsyncIterator[RunEvent]:
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        for i in range(1, self.steps + 1):
            yield RunEvent(type="log", message=f"[{spec.run_id}] step {i}/{self.steps}")
            yield RunEvent(type="progress", progress=i / self.steps)
        yield RunEvent(type="status", status=RunStatus.DONE)
