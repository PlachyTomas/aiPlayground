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
