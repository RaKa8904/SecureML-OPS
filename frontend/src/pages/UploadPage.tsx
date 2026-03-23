import { ChangeEvent, DragEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud } from "lucide-react";
import { uploadModel } from "../api/service";
import { saveLastModel } from "../store/sessionStore";

const allowed = [".pt", ".pth", ".onnx", ".h5"];

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const fileExt = useMemo(() => {
    if (!file) return "";
    const pieces = file.name.split(".");
    return pieces.length > 1 ? `.${pieces.pop()!.toLowerCase()}` : "";
  }, [file]);

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    const picked = event.dataTransfer.files?.[0];
    if (picked) setFile(picked);
  }

  function onFilePick(event: ChangeEvent<HTMLInputElement>) {
    const picked = event.target.files?.[0] || null;
    setFile(picked);
  }

  async function handleUpload() {
    if (!file) {
      setError("Pick a model file first.");
      return;
    }

    if (fileExt && !allowed.includes(fileExt)) {
      setError(`Unsupported format ${fileExt}. Allowed: ${allowed.join(", ")}`);
      return;
    }

    setError(null);
    setBusy(true);
    try {
      const uploaded = await uploadModel(file);
      saveLastModel(uploaded);
      navigate("/configure", { state: { modelId: uploaded.model_id } });
    } catch (err) {
      setError("Upload failed. Verify backend is running on port 8000.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mx-auto max-w-4xl px-4 py-10">
      <div className="rounded-[2rem] border border-white/50 bg-white/80 p-8 shadow-card backdrop-blur">
        <h1 className="font-display text-4xl text-ink">Upload Your Model</h1>
        <p className="mt-2 max-w-2xl text-ink/70">
          Drag and drop a model to start robustness testing. PyTorch is primary; ONNX and Keras are supported as secondary formats.
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          className={`mt-8 rounded-3xl border-2 border-dashed p-10 text-center transition ${
            dragActive ? "border-ocean bg-ocean/10" : "border-ink/20 bg-mist"
          }`}
        >
          <UploadCloud className="mx-auto h-12 w-12 text-ocean" />
          <p className="mt-3 font-semibold text-ink">Drop model here</p>
          <p className="mt-1 text-sm text-ink/60">or choose a file from disk</p>
          <input className="mt-4" type="file" accept={allowed.join(",")} onChange={onFilePick} />
        </div>

        {file ? (
          <div className="mt-4 rounded-xl bg-ink px-4 py-3 font-mono text-sm text-mist">
            Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
          </div>
        ) : null}

        {error ? <p className="mt-4 rounded-xl bg-red-100 px-4 py-3 text-sm text-red-700">{error}</p> : null}

        <button
          disabled={busy}
          onClick={handleUpload}
          className="mt-6 rounded-full bg-coral px-6 py-3 font-semibold text-white transition hover:brightness-110 disabled:opacity-50"
        >
          {busy ? "Uploading..." : "Upload & Continue"}
        </button>
      </div>
    </section>
  );
}
