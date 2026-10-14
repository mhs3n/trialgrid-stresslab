import type {
  ControllerEvidence,
  EvidenceStep,
  Finding,
  ReplayResult,
  RunDetail,
  RunSummary,
} from "./types";

async function requestJson(path: string, options?: RequestInit): Promise<unknown> {
  const response = await fetch(path, {
    ...options,
    headers: { Accept: "application/json", ...options?.headers },
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: unknown };
      detail = typeof body.detail === "string" ? `: ${body.detail}` : "";
    } catch {
      // The status remains useful when an error response is not JSON.
    }
    throw new Error(`${response.status} ${response.statusText}${detail}`);
  }
  return response.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  const value = await requestJson("/api/runs");
  const rows = isRecord(value) && Array.isArray(value.runs) ? value.runs : [];
  return rows.map(normalizeRunSummary);
}

export async function getRun(id: string): Promise<RunDetail> {
  return normalizeRunDetail(await requestJson(`/api/runs/${encodeURIComponent(id)}`));
}

export async function replayRun(id: string): Promise<ReplayResult> {
  const detail = normalizeRunDetail(
    await requestJson(`/api/runs/${encodeURIComponent(id)}/replay`, { method: "POST" }),
  );
  return {
    ok: detail.replayVerified,
    message: detail.replayVerified
      ? "Recorded actions reproduced the saved evidence."
      : "Replay did not verify.",
    detail,
  };
}

export async function rerunControllers(id: string): Promise<RunDetail> {
  return normalizeRunDetail(
    await requestJson(`/api/runs/${encodeURIComponent(id)}/rerun`, { method: "POST" }),
  );
}

export async function createDemo(): Promise<RunDetail> {
  return normalizeRunDetail(await requestJson("/api/demo", { method: "POST" }));
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function nullableText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function boolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

export function normalizeRunSummary(value: unknown): RunSummary {
  const row = isRecord(value) ? value : {};
  return {
    id: text(row.id, "unknown-run"),
    createdAt: nullableText(row.created_at_utc),
    targetName: text(row.target_name, "Unknown target"),
    comparatorName: text(row.comparator_name, "Unknown comparator"),
    hours: number(row.hours),
    targetProfit: number(row.target_profit_eur),
    comparatorProfit: number(row.comparator_profit_eur),
    targetReserveViolations: number(row.target_reserve_violations),
    targetRejectedActions: number(row.target_rejected_actions),
    findingCount: number(row.finding_count),
    status: text(row.status, "unknown"),
  };
}

export function normalizeRunDetail(value: unknown): RunDetail {
  if (!isRecord(value)) throw new Error("Run response was not an object");
  const scenario = isRecord(value.scenario) ? value.scenario : {};
  const report = isRecord(value.report) ? value.report : {};
  return {
    id: text(value.id, "unknown-run"),
    createdAt: nullableText(value.created_at_utc),
    replayVerified: value.replay_verified === true,
    scenario: {
      hours: number(scenario.hours),
      seed: number(scenario.seed),
      initialEnergy: number(scenario.initial_energy_kwh),
      energyFloor: number(scenario.energy_floor_kwh),
      energyCeiling: number(scenario.energy_ceiling_kwh),
    },
    target: normalizeController(value.target, "Target"),
    comparator: normalizeController(value.comparator, "Comparator"),
    report: {
      findings: Array.isArray(report.findings) ? report.findings.map(normalizeFinding) : [],
      profitGap: number(report.profit_gap_eur),
      economicEligible: boolean(report.economic_eligible),
      testedEndingEnergy: number(report.tested_ending_energy_kwh),
      comparatorEndingEnergy: number(report.comparator_ending_energy_kwh),
    },
    notices: strings(value.notices),
  };
}

function normalizeController(value: unknown, fallback: string): ControllerEvidence {
  const row = isRecord(value) ? value : {};
  const metadata = isRecord(row.metadata) ? row.metadata : {};
  const metrics = isRecord(row.metrics) ? row.metrics : {};
  return {
    name: text(metadata.name, fallback),
    version: text(metadata.version, "Unknown"),
    horizon: number(metadata.horizon),
    informationMode: text(metadata.information_mode, "Unknown"),
    label: text(metadata.label, fallback),
    configuration: metadata.configuration ?? null,
    energyPoints: Array.isArray(row.energy_points)
      ? row.energy_points.map((point, index) => normalizeEnergyPoint(point, index))
      : [],
    metrics: {
      revenue: number(metrics.revenue_eur),
      cost: number(metrics.degradation_cost_eur),
      profit: number(metrics.profit_eur),
      rejectedActions: number(metrics.rejected_action_count),
      reserveViolations: number(metrics.reserve_violation_count),
      endingEnergy: number(metrics.ending_energy_kwh),
    },
    steps: Array.isArray(row.steps) ? row.steps.map(normalizeStep) : [],
  };
}

function normalizeEnergyPoint(value: unknown, index: number) {
  const row = isRecord(value) ? value : {};
  return {
    step: number(row.step) ?? index,
    energy: number(row.energy_kwh),
  };
}

function normalizeStep(value: unknown, index: number): EvidenceStep {
  const row = isRecord(value) ? value : {};
  return {
    step: number(row.step) ?? index,
    actualPrice: number(row.actual_price_eur_per_mwh),
    publishedForecast: Array.isArray(row.published_forecast_eur_per_mwh)
      ? row.published_forecast_eur_per_mwh.flatMap((item) => {
          const parsed = number(item);
          return parsed === null ? [] : [parsed];
        })
      : [],
    reserve: number(row.reserve_requirement_kwh),
    startEnergy: number(row.start_energy_kwh),
    endEnergy: number(row.end_energy_kwh),
    requested: number(row.requested_p_kw),
    applied: number(row.applied_p_kw),
    revenue: number(row.revenue_eur),
    cost: number(row.degradation_cost_eur),
    profit: number(row.profit_eur),
    physicalViolations: strings(row.physical_violations),
    reserveShortfall: number(row.reserve_shortfall_kwh),
    violations: strings(row.violations),
  };
}

function normalizeFinding(value: unknown): Finding {
  const row = isRecord(value) ? value : {};
  return {
    kind: text(row.kind, "Unknown finding"),
    step: number(row.step_index ?? row.step),
    severity: number(row.severity),
    classification: nullableText(row.classification),
  };
}
