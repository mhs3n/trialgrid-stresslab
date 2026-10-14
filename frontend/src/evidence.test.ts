import { describe, expect, it } from "vitest";
import { currentForecast, formatDate, formatNumber, stepHasViolation } from "./evidence";
import type { EvidenceStep } from "./types";

describe("evidence display helpers", () => {
  it("labels unknown numeric and date evidence explicitly", () => {
    expect(formatNumber(null)).toBe("Unknown");
    expect(formatNumber(Number.NaN)).toBe("Unknown");
    expect(formatDate(null)).toBe("Unknown");
    expect(formatDate("not-a-date")).toBe("Unknown");
  });

  it("uses only the current published forecast entry for the comparison series", () => {
    const step = sampleStep({ publishedForecast: [17, 999] });
    expect(currentForecast(step)).toBe(17);
    expect(currentForecast(sampleStep({ publishedForecast: [] }))).toBeNull();
  });

  it("marks physical and reserve violations", () => {
    expect(stepHasViolation(sampleStep({ violations: ["power_limit"] }))).toBe(true);
    expect(stepHasViolation(sampleStep({ reserveShortfall: 1 }))).toBe(true);
    expect(stepHasViolation(sampleStep({}))).toBe(false);
  });
});

function sampleStep(change: Partial<EvidenceStep>): EvidenceStep {
  return {
    step: 0,
    actualPrice: 10,
    publishedForecast: [],
    reserve: 0,
    startEnergy: 50,
    endEnergy: 50,
    requested: 0,
    applied: 0,
    revenue: 0,
    cost: 0,
    profit: 0,
    physicalViolations: [],
    reserveShortfall: 0,
    violations: [],
    ...change,
  };
}
