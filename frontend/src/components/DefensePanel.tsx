import type { DefenseRecommendation } from "../types";

type Props = {
  recommendations: DefenseRecommendation[];
};

const priorityStyles: Record<string, string> = {
  critical: "bg-red-600 text-white",
  high: "bg-coral text-white",
  medium: "bg-sun text-ink",
  low: "bg-ocean text-white",
};

export default function DefensePanel({ recommendations }: Props) {
  return (
    <section className="rounded-3xl bg-white p-6 shadow-card">
      <h3 className="font-display text-xl text-ink">Defense Recommendations</h3>
      <div className="mt-4 space-y-4">
        {recommendations.map((rec) => (
          <article key={rec.defense} className="rounded-2xl border border-ink/10 p-4">
            <div className="flex items-center justify-between gap-3">
              <h4 className="font-semibold text-ink">{rec.defense}</h4>
              <span className={`rounded-full px-2 py-1 text-xs font-bold uppercase ${priorityStyles[rec.priority] || "bg-ink text-white"}`}>
                {rec.priority}
              </span>
            </div>
            <p className="mt-2 text-sm text-ink/70">{rec.reason}</p>
            <pre className="mt-3 overflow-auto rounded-xl bg-ink p-3 font-mono text-xs text-mist">
              {JSON.stringify(rec.config, null, 2)}
            </pre>
          </article>
        ))}
      </div>
    </section>
  );
}
