from visionsuite_core.backends import LocalBackend, RunSpec, RunStatus


async def test_local_backend_streams_to_done():
    spec = RunSpec(run_id="r1", model_id="dummy/echo", dataset_id="d1")
    events = [e async for e in LocalBackend().stream(spec)]
    assert events[0].type == "status" and events[0].status == RunStatus.RUNNING
    assert events[-1].status == RunStatus.DONE
    progresses = [e.progress for e in events if e.type == "progress"]
    assert progresses[-1] == 1.0
    assert any(e.type == "log" and "r1" in e.message for e in events)
