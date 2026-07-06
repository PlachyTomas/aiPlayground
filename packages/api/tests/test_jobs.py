import asyncio

from visionsuite_api.jobs import JobManager
from visionsuite_core.backends import RunSpec, RunStatus


async def test_job_runs_to_done():
    jm = JobManager()
    job = await jm.submit(RunSpec(run_id="r1", model_id="m", dataset_id="d"))
    for _ in range(200):
        if job.status == RunStatus.DONE:
            break
        await asyncio.sleep(0.01)
    assert job.status == RunStatus.DONE
    assert any(e.type == "progress" and e.progress == 1.0 for e in job.events)
    assert jm.get("r1") is job
