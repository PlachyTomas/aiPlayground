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
