import { useEffect, useState } from "react";
import { createLabelingProject, labelingStatus, labelStudioStatus, listDatasets,
  pullAnnotations, type DatasetInfo } from "../lib/api";
import { useJobStream } from "../lib/useJobStream";

export default function Labeling() {
  const [conn, setConn] = useState<{ connected: boolean } | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [classes, setClasses] = useState("");
  const [status, setStatus] = useState<Awaited<ReturnType<typeof labelingStatus>> | null>(null);
  const job = useJobStream();

  useEffect(() => { labelStudioStatus().then(setConn); listDatasets().then(setDatasets); }, []);
  useEffect(() => { if (selected != null) labelingStatus(selected).then(setStatus); }, [selected]);
  useEffect(() => { if (job.status === "done" && selected != null) labelingStatus(selected).then(setStatus); }, [job.status]);

  async function onCreate() {
    if (selected == null) return;
    const names = classes.split(",").map((c) => c.trim()).filter(Boolean);
    await createLabelingProject(selected, names);
    labelingStatus(selected).then(setStatus);
  }

  return (
    <div>
      <h1>Labeling</h1>
      <p>Label Studio: {conn?.connected ? "connected" : "not connected — start it and set LABEL_STUDIO_URL/API_KEY"}</p>
      <ul>{datasets.map((d) => (
        <li key={d.id}><button onClick={() => setSelected(d.id)}>{d.name}</button> ({d.task})</li>
      ))}</ul>
      {selected != null && (
        <div>
          {!status?.configured && (
            <div>
              <input placeholder="classes (comma-separated)" value={classes}
                     onChange={(e) => setClasses(e.target.value)} />
              <button onClick={onCreate}>Create labeling project</button>
            </div>
          )}
          {status?.configured && (
            <div>
              <a href={status.ls_url} target="_blank" rel="noreferrer">Open in Label Studio</a>
              <p>{status.annotated ?? 0} / {status.total ?? 0} annotated</p>
              <button onClick={() => pullAnnotations(selected).then((r) => job.watch(r.job_id))}>
                Pull annotations
              </button>
              <p>{job.status} <progress value={job.progress} max={1} /></p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
