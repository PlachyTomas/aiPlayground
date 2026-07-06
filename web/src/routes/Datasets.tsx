import { useEffect, useState } from "react";
import { apiUrl, createDataset, deleteDataset, listDatasets, listImages,
  type DatasetInfo, type ImageInfo } from "../lib/api";

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [name, setName] = useState("");
  const [task, setTask] = useState("detection");
  const [selected, setSelected] = useState<number | null>(null);
  const [images, setImages] = useState<ImageInfo[]>([]);

  const refresh = () => listDatasets().then(setDatasets);
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (selected != null) listImages(selected).then((r) => setImages(r.images));
  }, [selected]);

  async function onCreate() {
    if (!name) return;
    await createDataset(name, task);
    setName(""); refresh();
  }

  return (
    <div>
      <h1>Datasets</h1>
      <div>
        <input placeholder="dataset name" value={name} onChange={(e) => setName(e.target.value)} />
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          <option value="detection">detection</option>
          <option value="classification">classification</option>
        </select>
        <button onClick={onCreate}>Create</button>
      </div>
      <ul>
        {datasets.map((d) => (
          <li key={d.id}>
            <button onClick={() => setSelected(d.id)}>{d.name}</button>
            {" "}({d.task}, {d.image_count} images){" "}
            <button onClick={() => deleteDataset(d.id).then(refresh)}>delete</button>
          </li>
        ))}
      </ul>
      {selected != null && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {images.map((im) => (
            <img key={im.image_id} src={apiUrl(im.thumb_url)} width={96} height={96}
                 alt={im.filename} style={{ objectFit: "cover" }} />
          ))}
        </div>
      )}
    </div>
  );
}
