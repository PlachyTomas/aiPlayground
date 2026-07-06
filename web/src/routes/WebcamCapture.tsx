import { useEffect, useRef } from "react";
import { importWebcam } from "../lib/api";

export default function WebcamCapture({ dsId, onCaptured }: { dsId: number; onCaptured: () => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices.getUserMedia({ video: true }).then((s) => {
      stream = s;
      if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play?.(); }
    }).catch(() => {});
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  async function capture() {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video?.videoWidth || 320;
    canvas.height = video?.videoHeight || 240;
    canvas.getContext("2d")?.drawImage(video as CanvasImageSource, 0, 0, canvas.width, canvas.height);
    await new Promise<void>((resolve) =>
      canvas.toBlob(async (blob) => {
        if (blob) { await importWebcam(dsId, blob); onCaptured(); }
        resolve();
      }, "image/png"));
  }

  return (
    <div>
      <video ref={videoRef} width={320} height={240} muted playsInline />
      <div><button onClick={capture}>Capture</button></div>
    </div>
  );
}
