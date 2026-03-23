import { useEffect, useState } from "react";
import { getRunHistory, type StoredRun } from "../store/sessionStore";
import { listModels } from "../api/service";
import type { ModelRecord } from "../types";

export default function HistoryPage() {
  const [history, setHistory] = useState<StoredRun[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);

  useEffect(() => {
    setHistory(getRunHistory());
    listModels().then(setModels).catch(() => setModels([]));
  }, []);

  function modelName(id: string): string {
    return models.find((m) => m.model_id === id)?.filename || id;
  }

  return (
    <section className="mx-auto max-w-5xl px-4 py-10">
      <div className="rounded-3xl bg-white p-6 shadow-card">
        <h1 className="font-display text-3xl text-ink">Run History</h1>
        <p className="mt-2 text-sm text-ink/70">Recent robustness scans stored locally in your browser.</p>

        {history.length === 0 ? (
          <p className="mt-6 rounded-xl bg-mist px-4 py-3 text-sm text-ink/70">No runs yet. Launch attacks from Configure page.</p>
        ) : (
          <div className="mt-6 overflow-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-ink/10 text-ink/60">
                  <th className="py-2">When</th>
                  <th className="py-2">Model</th>
                  <th className="py-2">Score</th>
                  <th className="py-2">Severity</th>
                  <th className="py-2">Attacks</th>
                </tr>
              </thead>
              <tbody>
                {history.map((run) => (
                  <tr key={`${run.timestamp}-${run.modelId}`} className="border-b border-ink/5">
                    <td className="py-3">{new Date(run.timestamp).toLocaleString()}</td>
                    <td className="py-3">{modelName(run.modelId)}</td>
                    <td className="py-3 font-semibold">{run.score.toFixed(1)}</td>
                    <td className="py-3">{run.severity}</td>
                    <td className="py-3">{run.attacks.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
