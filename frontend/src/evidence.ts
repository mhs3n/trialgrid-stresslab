import type { EvidenceStep } from "./types";

export function formatNumber(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "Unknown";
}

export function formatDate(value: string | null): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString();
}

export function stepHasViolation(step: EvidenceStep): boolean {
  return step.violations.length > 0 || (step.reserveShortfall ?? 0) > 0;
}

export function currentForecast(step: EvidenceStep): number | null {
  return step.publishedForecast[0] ?? null;
}
