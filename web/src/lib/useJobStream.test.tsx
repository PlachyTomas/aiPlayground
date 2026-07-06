import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJobStream } from "./useJobStream";

class FakeWS {
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(_url: string) {
    setTimeout(() => {
      this.onmessage?.({ data: JSON.stringify({ type: "progress", progress: 1, status: null, message: "" }) });
      this.onmessage?.({ data: JSON.stringify({ type: "status", progress: null, status: "done", message: "" }) });
      this.onclose?.();
    }, 0);
  }
  close() {}
}

afterEach(() => vi.restoreAllMocks());

describe("useJobStream", () => {
  it("tracks progress to done", async () => {
    vi.stubGlobal("WebSocket", FakeWS as unknown as typeof WebSocket);
    const { result } = renderHook(() => useJobStream());
    act(() => result.current.watch("j1"));
    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.progress).toBe(1);
  });
});
