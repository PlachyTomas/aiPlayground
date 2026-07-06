import { useEffect, useRef, useState } from "react";
import { apiUrl } from "./api";

export function useJobStream() {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => () => wsRef.current?.close(), []);

  function watch(jobId: string) {
    setLogs([]); setProgress(0); setStatus("running");
    const base = apiUrl(`/api/jobs/${jobId}/events`);
    const url = (base.startsWith("http") ? base : window.location.origin + base).replace(/^http/, "ws");
    wsRef.current?.close();
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "log" && ev.message) setLogs((p) => [...p, ev.message]);
      if (ev.type === "progress" && ev.progress != null) setProgress(ev.progress);
      if (ev.type === "status" && ev.status) setStatus(ev.status);
    };
  }
  return { logs, progress, status, watch };
}
