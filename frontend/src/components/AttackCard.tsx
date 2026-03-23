import type { AttackResult } from "../types";

type Props = {
  name: string;
  result: AttackResult;
};

export default function AttackCard({ name, result }: Props) {
  const delta = Math.max(0, result.clean_accuracy - result.adv_accuracy);

  return (
    <article className="rounded-2xl border border-ink/10 bg-white p-4">
      <div className="flex items-start justify-between">
        <h4 className="font-display text-lg text-ink">{name}</h4>
        <span className="rounded-full bg-mist px-2 py-1 font-mono text-xs uppercase text-ink/70">{result.type}</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-ink/60">Clean Acc.</p>
          <p className="font-semibold text-ink">{(result.clean_accuracy * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-ink/60">Adv Acc.</p>
          <p className="font-semibold text-ink">{(result.adv_accuracy * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-ink/60">Damage</p>
          <p className="font-semibold text-coral">-{(delta * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-ink/60">Epsilon</p>
          <p className="font-semibold text-ink">{result.epsilon.toFixed(3)}</p>
        </div>
      </div>
    </article>
  );
}
