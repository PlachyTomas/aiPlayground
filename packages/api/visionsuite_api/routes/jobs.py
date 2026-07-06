import asyncio

from fastapi import APIRouter, WebSocket

from ..jobs import TERMINAL

router = APIRouter()


async def stream_job_events(websocket: WebSocket, job) -> None:
    await websocket.accept()
    if job is None:
        await websocket.close(code=4404)
        return
    sent = 0
    while True:
        while sent < len(job.events):
            e = job.events[sent]
            sent += 1
            await websocket.send_json({
                "type": e.type, "message": e.message,
                "progress": e.progress, "status": e.status.value if e.status else None,
            })
        if job.status in TERMINAL:
            break
        await asyncio.sleep(0.05)
    await websocket.close()


@router.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await stream_job_events(websocket, websocket.app.state.manager.get(job_id))
