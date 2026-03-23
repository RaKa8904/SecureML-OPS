import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { listModels, runAttacks } from "../api/service";
import type { ModelRecord } from "../types";
import { getLastModel, saveLastJob } from "../store/sessionStore";

const attackOptions = ["FGSM", "PGD", "C&W", "Transfer", "HopSkipJump", "Square"];

export default function ConfigurePage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState("");
  const [attacks, setAttacks] = useState<string[]>(["FGSM", "PGD", "C&W"]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    async function load() {
      try {
        const items = await listModels();
        setModels(items);

        const locationModel = (location.state as { modelId?: string } | null)?.modelId;
        const fallbackModel = getLastModel()?.model_id || items[0]?.model_id || "";
        setModelId(locationModel || fallbackModel);
      } catch (_err) {
        setError("Failed to load models. Is backend running?");
      }
    }
    load();
  }, [location.state]);

  const selectedModel = useMemo(() => models.find((m) => m.model_id === modelId), [models, modelId]);

  function toggleAttack(name: string) {
    setAttacks((current) =>
      current.includes(name) ? current.filter((a) => a !== name) : [...current, name]
    );
  }

  async function launchTest() {
    if (!modelId) {
      setError("Select a model first.");
      return;
    }
    if (attacks.length === 0) {
      setError("Select at least one attack.");
      return;
    }

    setError(null);
    setBusy(true);
    try {
      const job = await runAttacks(modelId, attacks);
      saveLastJob(job.job_id);
      navigate(`/results?job=${job.job_id}&model=${modelId}&attacks=${encodeURIComponent(attacks.join(","))}`);
    } catch (_err) {
      setError("Could not queue attack job.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mx-auto max-w-5xl px-4 py-10">
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-3xl bg-white p-6 shadow-card">
          <h1 className="font-display text-3xl text-ink">Configure Test</h1>
          <p className="mt-2 text-sm text-ink/70">Pick a model and select attacks to run in the worker queue.</p>

          <label className="mt-6 block text-sm font-semibold text-ink">Model</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="mt-2 w-full rounded-xl border border-ink/20 bg-mist px-3 py-2"
          >
            <option value="">Select model</option>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.filename}
              </option>
            ))}
          </select>

          <div className="mt-6">
            <p className="text-sm font-semibold text-ink">Attacks</p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {attackOptions.map((attack) => (
                <label key={attack} className="flex cursor-pointer items-center gap-2 rounded-xl border border-ink/10 bg-mist px-3 py-2 text-sm">
                  <input
                    type="checkbox"
                    checked={attacks.includes(attack)}
                    onChange={() => toggleAttack(attack)}
                  />
                  {attack}
                </label>
              ))}
            </div>
          </div>

          {error ? <p className="mt-4 rounded-xl bg-red-100 px-3 py-2 text-sm text-red-700">{error}</p> : null}

          <button
            disabled={busy}
            onClick={launchTest}
            className="mt-6 rounded-full bg-ocean px-6 py-3 font-semibold text-white disabled:opacity-50"
          >
            {busy ? "Queuing..." : "Launch Robustness Test"}
          </button>
        </div>

        <aside className="rounded-3xl bg-ink p-6 text-mist">
          <h2 className="font-display text-2xl">Current Selection</h2>
          <p className="mt-4 text-sm text-mist/80">Model</p>
          <p className="font-semibold">{selectedModel?.filename || "-"}</p>
          <p className="mt-4 text-sm text-mist/80">Attack Count</p>
          <p className="font-semibold">{attacks.length}</p>
          <p className="mt-4 text-sm text-mist/80">Selected</p>
          <ul className="mt-2 space-y-1 text-sm">
            {attacks.map((a) => (
              <li key={a}>• {a}</li>
            ))}
          </ul>
        </aside>
      </div>
    </section>
  );
}
