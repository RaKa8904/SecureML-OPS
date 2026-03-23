import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import AttackCard from "../components/AttackCard";
import DefensePanel from "../components/DefensePanel";
import PerturbationView from "../components/PerturbationView";
import ScoreGauge from "../components/ScoreGauge";
import { getAttackStatus } from "../api/service";
import type { AttackJobResult, AttackStatusResponse } from "../types";
import { getLastJob, pushRunHistory } from "../store/sessionStore";

export default function ResultsPage() {
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<AttackStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const jobId = useMemo(() => searchParams.get("job") || getLastJob() || "", [searchParams]);
  const modelId = useMemo(() => searchParams.get("model") || "unknown", [searchParams]);
  const attacks = useMemo(
    () => (searchParams.get("attacks") ? decodeURIComponent(searchParams.get("attacks")!).split(",") : []),
    [searchParams]
  );

  useEffect(() => {
    if (!jobId) return;

    let mounted = true;
    let historySaved = false;

    async function poll() {
      try {
        const next = await getAttackStatus(jobId);
        if (!mounted) return;

        setStatus(next);

        if (next.status === "SUCCESS" && !historySaved) {
          historySaved = true;
          pushRunHistory({
            timestamp: new Date().toISOString(),
            modelId,
            score: next.result.score,
            severity: next.result.severity,
            attacks,
          });
        }
      } catch (_err) {
        if (mounted) setError("Failed to fetch status. Check backend and worker.");
      }
    }

    poll();
    const interval = setInterval(poll, 2000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [jobId, modelId, attacks]);

  if (!jobId) {
    return (
      <section className="mx-auto max-w-4xl px-4 py-10">
        <div className="rounded-2xl bg-white p-6 shadow-card">
          <h1 className="font-display text-3xl text-ink">Results</h1>
          <p className="mt-2 text-ink/70">No job selected yet. Run attacks from Configure page first.</p>
        </div>
      </section>
    );
  }

  if (error) {
    return <p className="mx-auto mt-10 max-w-3xl rounded-xl bg-red-100 px-4 py-3 text-red-700">{error}</p>;
  }

  if (!status || ["PENDING", "RECEIVED", "STARTED", "ATTACKING"].includes(status.status)) {
    const progress = status && "progress" in status ? status.progress : 0;
    const currentAttack = status && "current_attack" in status ? status.current_attack : "initializing";

    return (
      <section className="mx-auto max-w-4xl px-4 py-10">
        <div className="rounded-3xl bg-white p-8 shadow-card">
          <h1 className="font-display text-3xl text-ink">Running Attack Suite</h1>
          <p className="mt-2 text-ink/70">Job ID: {jobId}</p>
          <div className="mt-6 h-4 overflow-hidden rounded-full bg-mist">
            <div className="h-full bg-ocean transition-all" style={{ width: `${progress}%` }} />
          </div>
          <div className="mt-3 flex justify-between text-sm text-ink/70">
            <span>Progress: {progress}%</span>
            <span>Current: {currentAttack || "-"}</span>
          </div>
        </div>
      </section>
    );
  }

  if (status.status === "FAILURE") {
    return (
      <section className="mx-auto max-w-4xl px-4 py-10">
        <div className="rounded-2xl bg-red-100 p-6 text-red-700">
          <h1 className="font-display text-2xl">Attack Job Failed</h1>
          <p className="mt-2 text-sm">{status.error}</p>
        </div>
      </section>
    );
  }

  if (status.status !== "SUCCESS") {
    return null;
  }

  const result: AttackJobResult = status.result;
  const firstAttackName = Object.keys(result.breakdown)[0];
  const firstAttackResult = firstAttackName ? result.breakdown[firstAttackName] : undefined;

  return (
    <section className="mx-auto max-w-6xl px-4 py-10">
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <ScoreGauge score={result.score} severity={result.severity} />
        </div>
        <div className="lg:col-span-2">
          <DefensePanel recommendations={result.recommendations} />
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Object.entries(result.breakdown).map(([name, attackResult]) => (
          <AttackCard key={name} name={name} result={attackResult} />
        ))}
      </div>

      {firstAttackName && firstAttackResult ? (
        <div className="mt-6">
          <PerturbationView attackName={firstAttackName} hasAdversarialSamples={Boolean(firstAttackResult.x_adv)} />
        </div>
      ) : null}
    </section>
  );
}
