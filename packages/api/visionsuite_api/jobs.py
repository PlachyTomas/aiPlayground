import asyncio

from visionsuite_core.backends import LocalBackend, RunEvent, RunSpec, RunStatus, TrainingBackend

TERMINAL = {RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED}


class Job:
    def __init__(self, spec: RunSpec, kind: str = "train") -> None:
        self.spec = spec
        self.kind = kind
        self.events: list[RunEvent] = []
        self.status: RunStatus = RunStatus.PENDING


class JobManager:
    def __init__(self, backend: TrainingBackend | None = None) -> None:
        self.backend = backend or LocalBackend()
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def submit(self, spec: RunSpec) -> Job:
        return await self.submit_stream(spec.run_id, "train", lambda: self.backend.stream(spec), spec=spec)

    async def submit_stream(self, job_id, kind, producer, spec=None) -> Job:
        job = Job(spec or RunSpec(run_id=job_id, model_id="", dataset_id=""), kind=kind)
        self._jobs[job_id] = job
        task = asyncio.create_task(self._run(job, producer))
        self._tasks.add(task)  # retain reference so the task isn't GC'd mid-flight
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job, producer) -> None:
        async with self._lock:  # one run at a time
            job.status = RunStatus.RUNNING
            try:
                async for event in producer():
                    job.events.append(event)
                    if event.status is not None:
                        job.status = event.status
            except Exception as exc:
                job.events.append(RunEvent(type="log", message=f"run failed: {exc}"))
                job.status = RunStatus.FAILED

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)
