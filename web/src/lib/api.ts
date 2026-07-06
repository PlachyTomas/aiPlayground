const BASE = import.meta.env.VITE_API_BASE ?? "";

export interface RunCreated {
  run_id: string;
  status: string;
}

export async function createRun(
  body: { model_id?: string; dataset_id?: string } = {},
): Promise<RunCreated> {
  const res = await fetch(`${BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json() as Promise<RunCreated>;
}

export function runEventsUrl(runId: string): string {
  const origin = BASE || window.location.origin;
  return origin.replace(/^http/, "ws") + `/api/runs/${runId}/events`;
}

export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

export interface DatasetInfo { id: number; name: string; task: string; image_count: number; }
export interface ImageInfo {
  image_id: string; filename: string; width: number; height: number; source: string;
  thumb_url: string; file_url: string;
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return (await fetch(apiUrl("/api/datasets"))).json();
}
export async function createDataset(name: string, task: string): Promise<DatasetInfo> {
  return (await fetch(apiUrl("/api/datasets"), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, task }),
  })).json();
}
export async function deleteDataset(id: number): Promise<void> {
  await fetch(apiUrl(`/api/datasets/${id}`), { method: "DELETE" });
}
export async function listImages(dsId: number, offset = 0, limit = 60):
  Promise<{ total: number; images: ImageInfo[] }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/images?offset=${offset}&limit=${limit}`))).json();
}

export async function importFolder(dsId: number, path: string): Promise<{ job_id: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/folder`), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }),
  })).json();
}
export async function importHf(dsId: number, dataset_id: string, limit = 200): Promise<{ job_id: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/hf`), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id, limit }),
  })).json();
}
export async function importVideo(dsId: number, file: File, everyN = 30): Promise<{ job_id: string }> {
  const fd = new FormData(); fd.append("file", file); fd.append("every_n", String(everyN));
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/video`), { method: "POST", body: fd })).json();
}
export async function importWebcam(dsId: number, blob: Blob): Promise<ImageInfo> {
  const fd = new FormData(); fd.append("file", blob, "frame.png");
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/webcam`), { method: "POST", body: fd })).json();
}
