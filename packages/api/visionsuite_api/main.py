from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, make_engine
from .jobs import JobManager
from .routes import datasets, health, jobs, runs


def create_app(engine=None, manager=None) -> FastAPI:
    app = FastAPI(title="VisionSuite API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine or make_engine()
    init_db(app.state.engine)
    app.state.manager = manager or JobManager()
    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(datasets.router)

    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
