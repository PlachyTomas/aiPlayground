import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Datasets from "./Datasets";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("Datasets", () => {
  it("lists datasets and creates one", async () => {
    vi.spyOn(api, "listDatasets").mockResolvedValue([
      { id: 1, name: "cars", task: "detection", image_count: 5 },
    ]);
    const create = vi.spyOn(api, "createDataset").mockResolvedValue({
      id: 2, name: "pets", task: "classification", image_count: 0,
    });
    render(<Datasets />);
    await waitFor(() => expect(screen.getByText(/cars/)).toBeTruthy());
    fireEvent.change(screen.getByPlaceholderText(/dataset name/i), { target: { value: "pets" } });
    fireEvent.click(screen.getByRole("button", { name: /create/i }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("pets", expect.any(String)));
  });
});
