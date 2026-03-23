import type { AttackJobResult, ModelRecord } from "../types";

const LAST_MODEL_KEY = "secureml:last-model";
const LAST_JOB_KEY = "secureml:last-job";
const RUN_HISTORY_KEY = "secureml:run-history";

export type StoredRun = {
  timestamp: string;
  modelId: string;
  score: number;
  severity: AttackJobResult["severity"];
  attacks: string[];
};

export function saveLastModel(model: ModelRecord): void {
  localStorage.setItem(LAST_MODEL_KEY, JSON.stringify(model));
}

export function getLastModel(): ModelRecord | null {
  const raw = localStorage.getItem(LAST_MODEL_KEY);
  return raw ? (JSON.parse(raw) as ModelRecord) : null;
}

export function saveLastJob(jobId: string): void {
  localStorage.setItem(LAST_JOB_KEY, jobId);
}

export function getLastJob(): string | null {
  return localStorage.getItem(LAST_JOB_KEY);
}

export function pushRunHistory(run: StoredRun): void {
  const existing = getRunHistory();
  const updated = [run, ...existing].slice(0, 20);
  localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(updated));
}

export function getRunHistory(): StoredRun[] {
  const raw = localStorage.getItem(RUN_HISTORY_KEY);
  return raw ? (JSON.parse(raw) as StoredRun[]) : [];
}
