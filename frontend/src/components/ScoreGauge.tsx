import { PolarAngleAxis, RadialBar, RadialBarChart, ResponsiveContainer } from "recharts";

type Props = {
  score: number;
  severity: string;
};

const severityColor: Record<string, string> = {
  CRITICAL: "#D7263D",
  HIGH: "#FF6B4A",
  MODERATE: "#F2C14E",
  STRONG: "#6B8E23",
  EXCELLENT: "#146C94",
};

export default function ScoreGauge({ score, severity }: Props) {
  const chartData = [{ name: "score", value: score, fill: severityColor[severity] || "#146C94" }];

  return (
    <div className="rounded-3xl bg-white p-6 shadow-card">
      <h3 className="font-display text-xl text-ink">Robustness Score</h3>
      <div className="mt-4 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart cx="50%" cy="50%" innerRadius="45%" outerRadius="85%" barSize={26} data={chartData} startAngle={180} endAngle={0}>
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar background dataKey="value" cornerRadius={14} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="-mt-6 text-center">
        <p className="font-display text-5xl text-ink">{score.toFixed(1)}</p>
        <p className="mt-1 inline-flex rounded-full px-3 py-1 text-xs font-bold text-white" style={{ backgroundColor: severityColor[severity] || "#146C94" }}>
          {severity}
        </p>
      </div>
    </div>
  );
}
