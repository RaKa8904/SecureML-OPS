export type ModelRecord = {
  model_id: string;
  filename: string;
  path: string;
  format: string;
  uploaded_at: string;
};

export type AttackResult = {
  attack: string;
  type: string;
  clean_accuracy: number;
  adv_accuracy: number;
  epsilon: number;
  x_adv?: number[][][][];
};

export type DefenseRecommendation = {
  defense: string;
  priority: "critical" | "high" | "medium" | "low";
  reason: string;
  config: Record<string, string | number | boolean>;
};

export type AttackJobResult = {
  score: number;
  severity: "CRITICAL" | "HIGH" | "MODERATE" | "STRONG" | "EXCELLENT";
  breakdown: Record<string, AttackResult>;
  recommendations: DefenseRecommendation[];
};

export type AttackStatusResponse =
  | {
      status: "PENDING" | "RECEIVED" | "STARTED" | "ATTACKING";
      progress: number;
      current_attack?: string;
    }
  | {
      status: "FAILURE";
      error: string;
    }
  | {
      status: "SUCCESS";
      progress: 100;
      result: AttackJobResult;
    };
