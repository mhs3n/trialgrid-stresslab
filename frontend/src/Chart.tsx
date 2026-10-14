import type { ReactNode } from "react";

export interface Series {
  label: string;
  color: string;
  dashed?: boolean;
  values: Array<number | null>;
}

interface ChartProps {
  title: string;
  subtitle: string;
  unit: string;
  series: Series[];
  violations?: boolean[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  children?: ReactNode;
}

const WIDTH = 800;
const HEIGHT = 250;
const PAD = { top: 20, right: 22, bottom: 35, left: 58 };

export function Chart({
  title,
  subtitle,
  unit,
  series,
  violations = [],
  selectedIndex,
  onSelect,
  children,
}: ChartProps) {
  const count = Math.max(1, ...series.map((item) => item.values.length));
  const finite = series.flatMap((item) => item.values).filter(isNumber);
  let min = finite.length ? Math.min(...finite) : 0;
  let max = finite.length ? Math.max(...finite) : 1;
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const space = (max - min) * 0.08;
  min -= space;
  max += space;
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (index: number) => PAD.left + (count === 1 ? plotW / 2 : (index / (count - 1)) * plotW);
  const y = (value: number) => PAD.top + ((max - value) / (max - min)) * plotH;
  const ticks = [max, (max + min) / 2, min];
  const selectedX = x(Math.min(selectedIndex, count - 1));

  return (
    <section className="chart-card">
      <div className="chart-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="legend" aria-label={`${title} legend`}>
          {series.map((item) => (
            <span key={item.label}>
              <i style={{ background: item.color }} className={item.dashed ? "dashed" : ""} />
              {item.label}
            </span>
          ))}
          {violations.some(Boolean) && (
            <span><i className="violation-dot" />Violation</span>
          )}
        </div>
      </div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${title}. ${subtitle}`}
        onPointerDown={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const viewX = ((event.clientX - box.left) / box.width) * WIDTH;
          const index = Math.round(((viewX - PAD.left) / plotW) * (count - 1));
          onSelect(Math.max(0, Math.min(count - 1, index)));
        }}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(tick)} y2={y(tick)} className="grid-line" />
            <text x={PAD.left - 8} y={y(tick) + 4} textAnchor="end" className="axis-text">
              {compact(tick)}
            </text>
          </g>
        ))}
        <text x={12} y={HEIGHT / 2} transform={`rotate(-90 12 ${HEIGHT / 2})`} className="unit-text">
          {unit}
        </text>
        {series.flatMap((item) =>
          segments(item.values).map((segment, segmentIndex) => (
            <polyline
              key={`${item.label}-${segmentIndex}`}
              points={segment.map(([index, value]) => `${x(index)},${y(value)}`).join(" ")}
              fill="none"
              stroke={item.color}
              strokeWidth="3"
              strokeDasharray={item.dashed ? "8 7" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )),
        )}
        {violations.map((flag, index) => flag && (
          <g key={index} aria-label={`Violation at hour ${index}`}>
            <circle cx={x(index)} cy={PAD.top + 10} r="7" className="violation-marker" />
            <text x={x(index)} y={PAD.top + 14} textAnchor="middle" className="violation-glyph">!</text>
          </g>
        ))}
        <line x1={selectedX} x2={selectedX} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="selection-line" />
        <circle cx={selectedX} cy={HEIGHT - PAD.bottom} r="4" className="selection-handle" />
        <text x={PAD.left} y={HEIGHT - 10} className="axis-text">Hour 0</text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 10} textAnchor="end" className="axis-text">
          Hour {count - 1}
        </text>
      </svg>
      {children}
    </section>
  );
}

function isNumber(value: number | null): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function segments(values: Array<number | null>): Array<Array<[number, number]>> {
  const result: Array<Array<[number, number]>> = [];
  let current: Array<[number, number]> = [];
  values.forEach((value, index) => {
    if (value === null) {
      if (current.length) result.push(current);
      current = [];
    } else {
      current.push([index, value]);
    }
  });
  if (current.length) result.push(current);
  return result;
}

function compact(value: number): string {
  return Math.abs(value) >= 1000
    ? value.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}
