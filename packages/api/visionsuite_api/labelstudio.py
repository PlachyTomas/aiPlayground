import json
import os
import time


class LabelStudioGateway:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._client = None

    def _sdk(self):
        if self._client is None:
            from label_studio_sdk import LabelStudio
            self._client = LabelStudio(base_url=self.url, api_key=self.api_key)
        return self._client

    def status(self) -> dict:
        try:
            self._sdk().projects.list()
            return {"connected": True, "url": self.url}
        except Exception as exc:  # noqa: BLE001
            return {"connected": False, "url": self.url, "detail": str(exc)}

    def create_project(self, title: str, label_config: str) -> int:
        return self._sdk().projects.create(title=title, label_config=label_config).id

    def connect_local_storage(self, project_id: int, abs_path: str, regex: str) -> None:
        sdk = self._sdk()
        storage = sdk.import_storage.local.create(
            project=project_id, path=abs_path, use_blob_urls=True, regex_filter=regex)
        sdk.import_storage.local.sync(id=storage.id)

    def project_stats(self, project_id: int) -> dict:
        p = self._sdk().projects.get(id=project_id)
        return {"total": getattr(p, "task_number", 0) or 0,
                "annotated": getattr(p, "num_tasks_with_annotations", 0) or 0}

    def export_json(self, project_id: int) -> list:
        sdk = self._sdk()
        ex = sdk.projects.exports.create(id=project_id)
        for _ in range(120):
            got = sdk.projects.exports.get(id=project_id, export_pk=ex.id)
            status = getattr(got, "status", "")
            if status == "completed":
                break
            if status in {"failed", "error"}:
                raise RuntimeError(f"Label Studio export failed with status {status!r}")
            time.sleep(0.5)
        else:
            raise TimeoutError("Label Studio export did not complete in time")
        chunks = sdk.projects.exports.download(id=project_id, export_pk=ex.id, export_type="JSON")
        return json.loads(b"".join(chunks))


def get_gateway(request) -> LabelStudioGateway:
    url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    key = os.environ.get("LABEL_STUDIO_API_KEY", "")
    return LabelStudioGateway(url, key)
