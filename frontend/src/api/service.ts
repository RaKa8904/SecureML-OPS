import { api } from "./client";
import type { AttackStatusResponse, ModelRecord } from "../types";

export async function uploadModel(file: File): Promise<ModelRecord> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<ModelRecord>("/api/models/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listModels(): Promise<ModelRecord[]> {
  const { data } = await api.get<{ models: ModelRecord[] }>("/api/models");
  return data.models;
}

export async function runAttacks(modelId: string, attacks: string[]): Promise<{ job_id: string; status: string }> {
  const { data } = await api.post<{ job_id: string; status: string }>("/api/attacks/run", {
    model_id: modelId,
    attacks,
  });
  return data;
}

export async function getAttackStatus(jobId: string): Promise<AttackStatusResponse> {
  const { data } = await api.get<AttackStatusResponse>(`/api/attacks/status/${jobId}`);
  return data;
}
