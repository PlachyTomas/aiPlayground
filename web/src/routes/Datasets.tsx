import { useEffect, useState } from "react";
import { apiUrl, createDataset, deleteDataset, importFolder, importHf, importVideo,
  listDatasets, listImages,
  type DatasetInfo, type ImageInfo } from "../lib/api";
import { useJobStream } from "../lib/useJobStream";

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [name, setName] = useState("");
  const [task, setTask] = useState("detection");
  const [selected, setSelected] = useState<number | null>(null);
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [folderPath, setFolderPath] = useState("");
  const [hfId, setHfId] = useState("");
  const job = useJobStream();

  const refresh = () => listDatasets().then(setDatasets);
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (selected != null) listImages(selected).then((r) => setImages(r.images));
  }, [selected]);
  useEffect(() => {
    if (job.status === "done" && selected != null)
      listImages(selected).then((r) => setImages(r.images));
  }, [job.status]);

  async function onCreate() {
    if (!name) return;
    await createDataset(name, task);
    setName(""); refresh();
  }

  function onImportFolder() {
    if (selected == null || !folderPath) return;
    importFolder(selected, folderPath).then((r) => job.watch(r.job_id));
  }
  function onImportHf() {
    if (selected == null || !hfId) return;
    importHf(selected, hfId).then((r) => job.watch(r.job_id));
  }
  function onImportVideo(file: File | undefined) {
    if (selected == null || !file) return;
    importVideo(selected, file).then((r) => job.watch(r.job_id));
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
        <div>
          <h2>Import into dataset {selected}</h2>
          <div>
            <input placeholder="folder path" value={folderPath}
                   onChange={(e) => setFolderPath(e.target.value)} />
            <button onClick={onImportFolder}>Import folder</button>
          </div>
          <div>
            <input placeholder="HF dataset id" value={hfId}
                   onChange={(e) => setHfId(e.target.value)} />
            <button onClick={onImportHf}>Import HF</button>
          </div>
          <div>
            <label>Import video{" "}
              <input type="file" accept="video/*"
                     onChange={(e) => onImportVideo(e.target.files?.[0])} />
            </label>
          </div>
          <p>Status: {job.status}</p>
          <progress value={job.progress} max={1} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {images.map((im) => (
              <img key={im.image_id} src={apiUrl(im.thumb_url)} width={96} height={96}
                   alt={im.filename} style={{ objectFit: "cover" }} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
