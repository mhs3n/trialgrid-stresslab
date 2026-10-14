export interface RunSummary {
  id: string;
  createdAt: string | null;
  targetName: string;
  comparatorName: string;
  hours: number | null;
  targetProfit: number | null;
  comparatorProfit: number | null;
  targetReserveViolations: number | null;
  targetRejectedActions: number | null;
  findingCount: number | null;
  status: string;
}

export interface EvidenceStep {
  step: number;
  actualPrice: number | null;
  publishedForecast: number[];
  reserve: number | null;
  startEnergy: number | null;
  endEnergy: number | null;
  requested: number | null;
  applied: number | null;
  revenue: number | null;
  cost: number | null;
  profit: number | null;
  physicalViolations: string[];
  reserveShortfall: number | null;
  violations: string[];
}

export interface ControllerEvidence {
  name: string;
  version: string;
  horizon: number | null;
  informationMode: string;
  label: string;
  configuration: unknown;
  metrics: {
    revenue: number | null;
    cost: number | null;
    profit: number | null;
    rejectedActions: number | null;
    reserveViolations: number | null;
    endingEnergy: number | null;
  };
  steps: EvidenceStep[];
}

export interface Finding {
  kind: string;
  step: number | null;
  severity: number | null;
  classification: string | null;
}

export interface RunDetail {
  id: string;
  createdAt: string | null;
  replayVerified: boolean;
  scenario: {
    hours: number | null;
    seed: number | null;
    initialEnergy: number | null;
    energyFloor: number | null;
    energyCeiling: number | null;
  };
  target: ControllerEvidence;
  comparator: ControllerEvidence;
  report: {
    findings: Finding[];
    profitGap: number | null;
    economicEligible: boolean | null;
    testedEndingEnergy: number | null;
    comparatorEndingEnergy: number | null;
  };
  notices: string[];
}

export interface ReplayResult {
  ok: boolean;
  message: string;
  detail?: RunDetail;
}
