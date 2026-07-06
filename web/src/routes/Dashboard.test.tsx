import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Dashboard from "./Dashboard";
import * as api from "../lib/api";

class FakeWS {
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(_url: string) {
    setTimeout(() => {
      this.onmessage?.({ data: JSON.stringify({ type: "log", message: "hi", progress: null, status: null }) });
      this.onmessage?.({ data: JSON.stringify({ type: "progress", message: "", progress: 1, status: null }) });
      this.onmessage?.({ data: JSON.stringify({ type: "status", message: "", progress: null, status: "done" }) });
      this.onclose?.();
    }, 0);
  }
  close() {}
}

afterEach(() => vi.restoreAllMocks());

describe("Dashboard", () => {
  it("starts a run and streams to done", async () => {
    vi.spyOn(api, "createRun").mockResolvedValue({ run_id: "r1", status: "pending" });
    vi.stubGlobal("WebSocket", FakeWS as unknown as typeof WebSocket);

    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: /start dummy run/i }));

    await waitFor(() => expect(screen.getByText(/status: done/i)).toBeTruthy());
    expect(screen.getByText("hi")).toBeTruthy();
  });
});
