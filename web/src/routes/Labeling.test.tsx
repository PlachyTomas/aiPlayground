import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Labeling from "./Labeling";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("Labeling", () => {
  it("shows LS connection and creates a project", async () => {
    vi.spyOn(api, "labelStudioStatus").mockResolvedValue({ connected: true, url: "http://ls:8080" });
    vi.spyOn(api, "listDatasets").mockResolvedValue([{ id: 1, name: "cars", task: "detection", image_count: 3 }]);
    vi.spyOn(api, "labelingStatus").mockResolvedValue({ configured: false });
    const create = vi.spyOn(api, "createLabelingProject").mockResolvedValue({ ls_project_id: 5, ls_url: "http://ls:8080/projects/5/data" });
    render(<Labeling />);
    await waitFor(() => expect(screen.getByText(/connected/i)).toBeTruthy());
    fireEvent.click(await screen.findByText(/cars/));
    fireEvent.change(screen.getByPlaceholderText(/classes/i), { target: { value: "car, bus" } });
    fireEvent.click(screen.getByRole("button", { name: /create labeling project/i }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(1, ["car", "bus"]));
  });
});
