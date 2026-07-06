from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import Field, SQLModel, create_engine

from visionsuite_core.workspace import workspace_root


class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    task: str
    project_id: int | None = Field(default=None, foreign_key="project.id")


class Image(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    image_id: str = Field(index=True)
    filename: str
    width: int
    height: int
    source: str
    path: str
    thumb_path: str


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    model_id: str
    dataset_id: str
    status: str = "pending"


class Model(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    path: str


def make_engine(url: str | None = None):
    if url is None:
        url = f"sqlite:///{workspace_root() / 'db.sqlite'}"
    kwargs = {"connect_args": {"check_same_thread": False}}
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    else:
        kwargs["poolclass"] = StaticPool  # one shared connection so in-memory DB survives across threads
    return create_engine(url, **kwargs)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)
