import { useRef, useState } from "react";
import { createRun, runEventsUrl } from "../lib/api";

interface Line {
  text: string;
}

export default function Dashboard() {
  const [logs, setLogs] = useState<Line[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef<WebSocket | null>(null);

  async function start() {
    setLogs([]);
    setProgress(0);
    setStatus("pending");
    const { run_id } = await createRun({});
    const ws = new WebSocket(runEventsUrl(run_id));
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "log") setLogs((prev) => [...prev, { text: ev.message }]);
      if (ev.type === "progress" && ev.progress != null) setProgress(ev.progress);
      if (ev.type === "status" && ev.status) setStatus(ev.status);
    };
    ws.onclose = () => setStatus((s) => (s === "pending" ? "closed" : s));
  }

  return (
    <div>
      <h1>Dashboard</h1>
      <button onClick={start}>Start dummy run</button>
      <p>Status: {status}</p>
      <progress value={progress} max={1} />
      <ul>
        {logs.map((l, i) => (
          <li key={i}>{l.text}</li>
        ))}
      </ul>
    </div>
  );
}
