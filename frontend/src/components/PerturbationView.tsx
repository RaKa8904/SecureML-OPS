type Props = {
  attackName: string;
  hasAdversarialSamples: boolean;
};

export default function PerturbationView({ attackName, hasAdversarialSamples }: Props) {
  return (
    <section className="rounded-2xl border border-dashed border-ink/20 bg-white p-4">
      <h4 className="font-display text-lg text-ink">Perturbation Preview: {attackName}</h4>
      <p className="mt-2 text-sm text-ink/70">
        {hasAdversarialSamples
          ? "Adversarial samples were generated. Image rendering endpoint can be wired in report phase for side-by-side visual previews."
          : "No adversarial samples available for preview."}
      </p>
    </section>
  );
}
