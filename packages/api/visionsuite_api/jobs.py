import asyncio

from visionsuite_core.backends import LocalBackend, RunEvent, RunSpec, RunStatus, TrainingBackend

TERMINAL = {RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED}


class Job:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.events: list[RunEvent] = []
        self.status: RunStatus = RunStatus.PENDING


class JobManager:
    def __init__(self, backend: TrainingBackend | None = None) -> None:
        self.backend = backend or LocalBackend()
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit(self, spec: RunSpec) -> Job:
        job = Job(spec)
        self._jobs[spec.run_id] = job
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        async with self._lock:  # one run at a time
            job.status = RunStatus.RUNNING
            async for event in self.backend.stream(job.spec):
                job.events.append(event)
                if event.status is not None:
                    job.status = event.status

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)
