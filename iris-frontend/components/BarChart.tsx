"use client";

/** Horizontal bar list in SVG, drawn from the theme tokens so it reads in
 *  light, dark and high-contrast themes. One hue (the accent): identity
 *  comes from the row labels and group headers, not from color. Optional
 *  95% CI whiskers. Native <title> per bar provides hover + SR detail; the
 *  adjacent data table is always the full record. */

export interface BarRow {
  label: string;
  value: number;
  ci?: [number, number];
  detail?: string;
}

export interface BarGroup {
  title?: string;
  rows: BarRow[];
}

const ROW_H = 30;
const GROUP_H = 34;
const LABEL_W = 210;
const CHART_W = 640;
const TOP = 26;

export default function BarChart({
  groups,
  maxValue,
  unit = "%",
  ariaLabel,
}: {
  groups: BarGroup[];
  maxValue: number;
  unit?: string;
  ariaLabel: string;
}) {
  const nRows = groups.reduce((n, g) => n + g.rows.length, 0);
  const nHeads = groups.filter((g) => g.title).length;
  const height = TOP + nRows * ROW_H + nHeads * GROUP_H + 24;
  const plotW = CHART_W - LABEL_W - 56;
  const x = (v: number) => LABEL_W + (v / maxValue) * plotW;
  const ticks = Array.from(
    { length: Math.floor(maxValue / 10) + 1 },
    (_, i) => i * 10,
  );

  let y = TOP;
  const elements: React.ReactNode[] = [];
  for (const g of groups) {
    if (g.title) {
      elements.push(
        <text
          key={`h-${g.title}`}
          x={0}
          y={y + GROUP_H - 12}
          fontSize={14}
          fontWeight={700}
          fill="var(--muted)"
        >
          {g.title.toUpperCase()}
        </text>,
      );
      y += GROUP_H;
    }
    for (const r of g.rows) {
      const cy = y + ROW_H / 2;
      elements.push(
        <g key={r.label}>
          <title>{`${r.label}: ${r.value}${unit}${
            r.ci ? ` (95% CI ${r.ci[0]}–${r.ci[1]})` : ""
          }${r.detail ? ` — ${r.detail}` : ""}`}</title>
          <text
            x={LABEL_W - 8}
            y={cy + 4}
            fontSize={13}
            textAnchor="end"
            fill="var(--fg)"
          >
            {r.label.length > 26 ? r.label.slice(0, 25) + "…" : r.label}
          </text>
          <rect
            x={LABEL_W}
            y={cy - 9}
            width={Math.max(2, x(r.value) - LABEL_W)}
            height={18}
            rx={3}
            fill="var(--accent)"
          />
          {r.ci && (
            <g stroke="var(--fg)" strokeWidth={2}>
              <line x1={x(r.ci[0])} x2={x(r.ci[1])} y1={cy} y2={cy} />
              <line x1={x(r.ci[0])} x2={x(r.ci[0])} y1={cy - 6} y2={cy + 6} />
              <line x1={x(r.ci[1])} x2={x(r.ci[1])} y1={cy - 6} y2={cy + 6} />
            </g>
          )}
          <text
            x={x(r.ci ? Math.max(r.value, r.ci[1]) : r.value) + 6}
            y={cy + 4}
            fontSize={13}
            fontWeight={700}
            fill="var(--fg)"
          >
            {r.value}
          </text>
        </g>,
      );
      y += ROW_H;
    }
  }

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${CHART_W} ${height}`}
        role="img"
        aria-label={ariaLabel}
        className="min-w-[560px] max-w-full"
        style={{ fontFamily: "var(--font-sans)" }}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={TOP - 8}
              y2={height - 20}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={TOP - 12}
              fontSize={12}
              textAnchor="middle"
              fill="var(--muted)"
            >
              {t}
              {unit}
            </text>
          </g>
        ))}
        {elements}
      </svg>
    </div>
  );
}
