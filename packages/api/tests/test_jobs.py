import asyncio

from visionsuite_api.jobs import JobManager
from visionsuite_core.backends import RunEvent, RunSpec, RunStatus


async def _wait_for(predicate, iterations: int = 400) -> None:
    for _ in range(iterations):
        if predicate():
            return
        await asyncio.sleep(0.01)


async def test_job_runs_to_done():
    jm = JobManager()
    job = await jm.submit(RunSpec(run_id="r1", model_id="m", dataset_id="d"))
    await _wait_for(lambda: job.status == RunStatus.DONE, 200)
    assert job.status == RunStatus.DONE
    assert any(e.type == "progress" and e.progress == 1.0 for e in job.events)
    assert jm.get("r1") is job


class RaisingBackend:
    async def stream(self, spec: RunSpec):
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        raise RuntimeError("boom")


async def test_backend_exception_drives_job_to_failed():
    jm = JobManager(backend=RaisingBackend())
    job = await jm.submit(RunSpec(run_id="rfail", model_id="m", dataset_id="d"))
    await _wait_for(lambda: job.status in {RunStatus.FAILED, RunStatus.DONE}, 200)
    assert job.status == RunStatus.FAILED
    assert any(e.type == "log" and "run failed" in e.message for e in job.events)


class OverlapBackend:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def stream(self, spec: RunSpec):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        await asyncio.sleep(0.02)  # yield control so an unserialized run could interleave here
        self.active -= 1
        yield RunEvent(type="status", status=RunStatus.DONE)


async def test_lock_serializes_two_runs():
    backend = OverlapBackend()
    jm = JobManager(backend=backend)
    j1 = await jm.submit(RunSpec(run_id="a", model_id="m", dataset_id="d"))
    j2 = await jm.submit(RunSpec(run_id="b", model_id="m", dataset_id="d"))
    await _wait_for(lambda: j1.status == RunStatus.DONE and j2.status == RunStatus.DONE)
    assert j1.status == RunStatus.DONE
    assert j2.status == RunStatus.DONE
    assert backend.max_active == 1
