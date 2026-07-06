import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import WebcamCapture from "./WebcamCapture";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("WebcamCapture", () => {
  it("captures a frame and posts it", async () => {
    vi.stubGlobal("navigator", {
      mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
    });
    // canvas.toBlob → provide a blob synchronously
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({ drawImage: vi.fn() });
    HTMLCanvasElement.prototype.toBlob = function (cb: BlobCallback) { cb(new Blob(["x"])); };
    const post = vi.spyOn(api, "importWebcam").mockResolvedValue({
      image_id: "z", filename: "z.png", width: 1, height: 1, source: "webcam",
      thumb_url: "/thumb/z.png", file_url: "/file/z.png",
    });
    const onCaptured = vi.fn();
    render(<WebcamCapture dsId={1} onCaptured={onCaptured} />);
    fireEvent.click(await screen.findByRole("button", { name: /capture/i }));
    await waitFor(() => expect(post).toHaveBeenCalled());
    await waitFor(() => expect(onCaptured).toHaveBeenCalled());
  });
});
