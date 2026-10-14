import { describe, expect, it } from "vitest";
import { normalizeRunDetail, normalizeRunSummary } from "./api";

describe("evidence API normalization", () => {
  it("preserves zero values and exposes missing values as null", () => {
    const summary = normalizeRunSummary({
      id: "demo-1",
      created_at_utc: "2026-10-14T10:00:00Z",
      target_name: "hold",
      comparator_name: "optimizer",
      hours: 2,
      target_profit_eur: 0,
      comparator_profit_eur: 1.5,
      target_reserve_violations: 0,
      target_rejected_actions: 0,
      finding_count: 0,
      status: "verified",
    });
    expect(summary.targetProfit).toBe(0);
    expect(summary.findingCount).toBe(0);
    expect(normalizeRunSummary({ id: "partial" }).targetProfit).toBeNull();
  });

  it("maps projected steps without accepting raw prompt fields", () => {
    const detail = normalizeRunDetail({
      id: "demo-1",
      created_at_utc: null,
      replay_verified: true,
      scenario: {
        hours: 1,
        seed: 7,
        initial_energy_kwh: 50,
        energy_floor_kwh: 10,
        energy_ceiling_kwh: 90,
      },
      target: controller("target", 2),
      comparator: controller("comparator", 0),
      report: {
        findings: [{ kind: "reserve_shortfall", step_index: 0, severity: 2 }],
        profit_gap_eur: 1,
        economic_eligible: true,
        tested_ending_energy_kwh: 48,
        comparator_ending_energy_kwh: 50,
      },
      notices: ["Synthetic fixture"],
      raw_prompt: "must never be rendered",
    });
    expect(detail.replayVerified).toBe(true);
    expect(detail.target.steps[0]).toMatchObject({
      requested: 20,
      applied: 0,
      reserveShortfall: 2,
      publishedForecast: [40, 41],
    });
    expect(detail.target.energyPoints).toEqual([
      { step: 0, energy: 50 },
      { step: 1, energy: 50 },
    ]);
    expect(JSON.stringify(detail)).not.toContain("must never be rendered");
  });
});

function controller(name: string, shortfall: number) {
  return {
    metadata: {
      name,
      version: "1",
      horizon: 2,
      configuration: {},
      information_mode: "published_forecast_only",
      label: name,
    },
    metrics: {
      revenue_eur: 0,
      degradation_cost_eur: 0,
      profit_eur: 0,
      rejected_action_count: 1,
      reserve_violation_count: shortfall ? 1 : 0,
      ending_energy_kwh: 50,
    },
    energy_points: [
      { step: 0, energy_kwh: 50 },
      { step: 1, energy_kwh: 50 },
    ],
    steps: [{
      step: 0,
      actual_price_eur_per_mwh: 40,
      published_forecast_eur_per_mwh: [40, 41],
      reserve_requirement_kwh: 50,
      start_energy_kwh: 50,
      end_energy_kwh: 50,
      requested_p_kw: 20,
      applied_p_kw: 0,
      revenue_eur: 0,
      degradation_cost_eur: 0,
      profit_eur: 0,
      physical_violations: ["energy_floor"],
      reserve_shortfall_kwh: shortfall,
      violations: ["energy_floor", "reserve_shortfall"],
      response_json: "ignored",
    }],
  };
}
