import { afterEach, describe, expect, it, vi } from "vitest";
import { createRun, runEventsUrl } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("POSTs to /api/runs and returns the body", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ run_id: "r1", status: "pending" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const res = await createRun({});
    expect(res.run_id).toBe("r1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("builds a ws:// events url", () => {
    const url = runEventsUrl("r1");
    expect(url).toMatch(/^wss?:\/\/.*\/api\/runs\/r1\/events$/);
  });
});
