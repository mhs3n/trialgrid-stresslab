import { useEffect, useMemo, useState } from "react";
import { createDemo, getRun, listRuns, replayRun, rerunControllers } from "./api";
import { Chart } from "./Chart";
import { currentForecast, formatDate, formatNumber, stepHasViolation } from "./evidence";
import type { ControllerEvidence, RunDetail, RunSummary } from "./types";

type EvidenceSide = "target" | "comparator";

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [side, setSide] = useState<EvidenceSide>("target");
  const [selectedStep, setSelectedStep] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    void refreshRuns();
  }, []);

  async function refreshRuns(preferredId?: string) {
    setLoading(true);
    setError(null);
    try {
      const next = await listRuns();
      setRuns(next);
      const id = preferredId ?? detail?.id ?? next[0]?.id;
      if (id) setDetail(await getRun(id));
      else setDetail(null);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setLoading(false);
    }
  }

  async function selectRun(id: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      setDetail(await getRun(id));
      setSelectedStep(0);
      setSide("target");
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  }

  async function demo() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await createDemo();
      setDetail(created);
      setSelectedStep(0);
      setSide("target");
      setNotice("Demo evidence created and verified.");
      const next = await listRuns();
      setRuns(next);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  }

  async function replay() {
    if (!detail) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await replayRun(detail.id);
      if (result.detail) setDetail(result.detail);
      setNotice(result.message);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  }

  async function rerun() {
    if (!detail) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await rerunControllers(detail.id);
      setDetail(created);
      setSelectedStep(0);
      setNotice("Current controller code ran on the saved scenario and produced a new artifact.");
      setRuns(await listRuns());
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  }

  const evidence = detail?.[side] ?? null;
  const step = evidence?.steps[Math.min(selectedStep, Math.max(0, evidence.steps.length - 1))];
  const violations = evidence?.steps.map(stepHasViolation) ?? [];
  const findingsByStep = useMemo(() => {
    const map = new Map<number, string[]>();
    for (const finding of detail?.report.findings ?? []) {
      if (finding.step === null) continue;
      map.set(finding.step, [...(map.get(finding.step) ?? []), finding.kind]);
    }
    return map;
  }, [detail]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
        <div className="brand-copy">
          <strong>TRIALGRID</strong>
          <span>Evidence lab</span>
        </div>
        <div className="header-actions">
          <span className="offline-badge">Offline simulation</span>
          <button className="primary-button" onClick={demo} disabled={busy}>Run demo</button>
        </div>
      </header>

      <div className="workspace">
        <aside className="run-rail" aria-label="Saved runs">
          <div className="rail-heading">
            <div><span className="eyebrow">Evidence index</span><h1>Runs</h1></div>
            <button className="icon-button" onClick={() => void refreshRuns()} aria-label="Refresh runs" disabled={loading}>↻</button>
          </div>
          {loading && runs.length === 0 ? <p className="empty-state">Loading evidence…</p> : null}
          {!loading && runs.length === 0 ? (
            <div className="empty-state"><strong>No saved runs</strong><span>Create the deterministic demo to begin.</span></div>
          ) : null}
          <div className="run-list">
            {runs.map((run) => (
              <button
                key={run.id}
                className={`run-row ${detail?.id === run.id ? "selected" : ""}`}
                onClick={() => void selectRun(run.id)}
                disabled={busy}
              >
                <div className="run-row-top">
                  <span className={`status-dot ${run.status === "verified" ? "verified" : ""}`} />
                  <strong>{run.targetName}</strong>
                  <span className="finding-count">{formatCount(run.findingCount)} findings</span>
                </div>
                <span>vs {run.comparatorName}</span>
                <div className="run-meta"><span>{formatCount(run.hours)} h</span><span>{formatDate(run.createdAt)}</span></div>
              </button>
            ))}
          </div>
        </aside>

        <main className="main-content">
          {error && <div className="alert error" role="alert">{error}</div>}
          {notice && <div className="alert success" role="status">{notice}</div>}
          {!detail ? (
            <section className="welcome-card">
              <span className="eyebrow">Deterministic evidence viewer</span>
              <h2>Inspect what the controller saw, asked for, and caused.</h2>
              <p>Choose a saved run or create the demo. TrialGrid keeps future realized prices outside the controller observation and makes requested versus applied power explicit.</p>
              <button className="primary-button" onClick={demo} disabled={busy}>Create demo evidence</button>
            </section>
          ) : evidence && step ? (
            <>
              <section className="run-header">
                <div>
                  <div className="title-line">
                    <span className={detail.replayVerified ? "verified-pill" : "unknown-pill"}>
                      {detail.replayVerified ? "✓ Replay verified" : "Replay unknown"}
                    </span>
                    <span className="run-id">{detail.id}</span>
                  </div>
                  <h2>{detail.target.label} <span>against</span> {detail.comparator.label}</h2>
                  <p>Seed {formatNumber(detail.scenario.seed, 0)} · {formatNumber(detail.scenario.hours, 0)} hourly steps · saved {formatDate(detail.createdAt)}</p>
                </div>
                <div className="run-actions">
                  <button className="secondary-button" onClick={replay} disabled={busy}>{busy ? "Working…" : "Verify replay"}</button>
                  <button className="primary-button" onClick={rerun} disabled={busy}>Run current controllers</button>
                </div>
              </section>

              <section className="candidate-switch" aria-label="Controller evidence selection">
                <ControllerTab label="Target candidate" evidence={detail.target} active={side === "target"} onClick={() => { setSide("target"); setSelectedStep(0); }} />
                <ControllerTab label="Comparator" evidence={detail.comparator} active={side === "comparator"} onClick={() => { setSide("comparator"); setSelectedStep(0); }} />
              </section>

              <section className="metric-grid" aria-label="Controller metrics">
                <Metric label="Realized profit" value={`€${formatNumber(evidence.metrics.profit, 2)}`} tone={(evidence.metrics.profit ?? 0) < 0 ? "bad" : "good"} />
                <Metric label="Ending energy" value={`${formatNumber(evidence.metrics.endingEnergy)} kWh`} />
                <Metric label="Reserve violations" value={formatNumber(evidence.metrics.reserveViolations, 0)} tone={(evidence.metrics.reserveViolations ?? 0) > 0 ? "bad" : "good"} />
                <Metric label="Rejected actions" value={formatNumber(evidence.metrics.rejectedActions, 0)} tone={(evidence.metrics.rejectedActions ?? 0) > 0 ? "bad" : "good"} />
                <Metric label="Profit gap" value={`€${formatNumber(detail.report.profitGap, 2)}`} hint="Comparator minus target" />
              </section>

              <div className="evidence-layout">
                <div className="charts-column">
                  <Chart
                    title="Market signal"
                    subtitle="Realized current price and the current-hour value from the published forecast"
                    unit="EUR / MWh"
                    selectedIndex={selectedStep}
                    onSelect={setSelectedStep}
                    violations={violations}
                    series={[
                      { label: "Actual realized", color: "#cb5a38", values: evidence.steps.map((item) => item.actualPrice) },
                      { label: "Published forecast", color: "#74c7a5", dashed: true, values: evidence.steps.map(currentForecast) },
                    ]}
                  />
                  <Chart
                    title="Stored energy and commitment"
                    subtitle="End-of-step battery energy against the post-step minimum reserve"
                    unit="kWh"
                    selectedIndex={selectedStep}
                    onSelect={setSelectedStep}
                    violations={violations}
                    series={[
                      { label: "Energy", color: "#f1bd58", values: evidence.steps.map((item) => item.endEnergy) },
                      { label: "Post-step reserve", color: "#7897d5", dashed: true, values: evidence.steps.map((item) => item.reserve) },
                    ]}
                  />
                  <Chart
                    title="Dispatch request and application"
                    subtitle="Positive power discharges to the grid; negative power charges the battery"
                    unit="grid kW"
                    selectedIndex={selectedStep}
                    onSelect={setSelectedStep}
                    violations={violations}
                    series={[
                      { label: "Requested", color: "#d7dcdf", values: evidence.steps.map((item) => item.requested) },
                      { label: "Applied", color: "#a987e6", dashed: true, values: evidence.steps.map((item) => item.applied) },
                    ]}
                  />
                </div>

                <aside className="step-panel" aria-label={`Evidence for hour ${step.step}`}>
                  <div className="step-panel-heading"><span className="eyebrow">Selected step</span><strong>Hour {step.step}</strong></div>
                  {stepHasViolation(step) ? <div className="step-alert">! Violation recorded</div> : <div className="step-ok">✓ No violation</div>}
                  <StepRow label="Actual price" value={`€${formatNumber(step.actualPrice)} / MWh`} />
                  <StepRow label="Published forecast" value={step.publishedForecast.length ? step.publishedForecast.map((value) => `€${formatNumber(value)}`).join(" · ") : "Unknown"} />
                  <StepRow label="Start → end" value={`${formatNumber(step.startEnergy)} → ${formatNumber(step.endEnergy)} kWh`} />
                  <StepRow label="Reserve" value={`${formatNumber(step.reserve)} kWh`} />
                  <StepRow label="Requested power" value={`${signed(step.requested)} kW`} />
                  <StepRow label="Applied power" value={`${signed(step.applied)} kW`} />
                  <StepRow label="Step profit" value={`€${formatNumber(step.profit, 2)}`} />
                  <StepRow label="Reserve shortfall" value={`${formatNumber(step.reserveShortfall)} kWh`} />
                  <div className="violation-list">
                    <span>Evidence labels</span>
                    {step.violations.length || findingsByStep.has(step.step) ? (
                      [...new Set([...step.violations, ...(findingsByStep.get(step.step) ?? [])])].map((item) => <strong key={item}>{humanize(item)}</strong>)
                    ) : <em>None</em>}
                  </div>
                </aside>
              </div>

              {detail.notices.length > 0 && <section className="notice-card"><strong>Evidence notes</strong>{detail.notices.map((item) => <p key={item}>{item}</p>)}</section>}
            </>
          ) : (
            <section className="welcome-card"><h2>Evidence unavailable</h2><p>This run has no displayable controller steps.</p></section>
          )}
        </main>
      </div>
    </div>
  );
}

function ControllerTab({ label, evidence, active, onClick }: { label: string; evidence: ControllerEvidence; active: boolean; onClick: () => void }) {
  return (
    <button className={active ? "active" : ""} onClick={onClick} aria-pressed={active}>
      <span>{label}</span><strong>{evidence.label}</strong><small>{humanize(evidence.informationMode)} · horizon {formatNumber(evidence.horizon, 0)}</small>
    </button>
  );
}

function Metric({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "good" | "bad" }) {
  return <article className={`metric-card ${tone ?? ""}`}><span>{label}</span><strong>{value}</strong>{hint && <small>{hint}</small>}</article>;
}

function StepRow({ label, value }: { label: string; value: string }) {
  return <div className="step-row"><span>{label}</span><strong>{value}</strong></div>;
}

function formatCount(value: number | null): string {
  return value === null ? "Unknown" : String(value);
}

function signed(value: number | null): string {
  return `${(value ?? 0) > 0 ? "+" : ""}${formatNumber(value)}`;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function message(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The evidence request failed.";
}
