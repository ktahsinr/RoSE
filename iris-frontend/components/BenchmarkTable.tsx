"use client";

/** The published ChartQAPro table: sortable (buttons in the headers with
 *  aria-sort), screen-reader friendly (caption, row/column headers), best
 *  score per column in bold as in the source. */

import { useMemo, useState } from "react";
import type { BenchmarksPayload } from "@/lib/api";

type Row = BenchmarksPayload["published"]["rows"][number];

export default function BenchmarkTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Row[];
}) {
  // sortKey: -1 = source order (grouped); 0..n = score column index.
  const [sortKey, setSortKey] = useState(-1);
  const [desc, setDesc] = useState(true);

  const sorted = useMemo(() => {
    if (sortKey === -1) return rows;
    return [...rows].sort(
      (a, b) => (desc ? -1 : 1) * (a.scores[sortKey] - b.scores[sortKey]),
    );
  }, [rows, sortKey, desc]);

  const toggle = (idx: number) => {
    if (sortKey === idx) setDesc((d) => !d);
    else {
      setSortKey(idx);
      setDesc(true);
    }
  };

  return (
    <div className="overflow-x-auto rounded-lg border-2 border-line">
      <table className="w-full min-w-[720px] border-collapse text-base">
        <caption className="p-2 text-left text-sm text-muted">
          Accuracy (%) by question type. <strong>Bold</strong> marks the best
          score in each column. Click a column header to sort; the first
          column restores the paper&apos;s grouping.
        </caption>
        <thead>
          <tr className="border-b-2 border-line bg-surface">
            <th scope="col" className="p-2 text-left">
              <button
                type="button"
                onClick={() => {
                  setSortKey(-1);
                }}
                className="font-bold underline-offset-4 hover:underline"
              >
                Model (grouped)
              </button>
            </th>
            {columns.map((c, i) => (
              <th
                scope="col"
                key={c}
                aria-sort={
                  sortKey === i ? (desc ? "descending" : "ascending") : "none"
                }
                className="p-2 text-right"
              >
                <button
                  type="button"
                  onClick={() => toggle(i)}
                  className="font-bold underline-offset-4 hover:underline"
                >
                  {c}
                  {sortKey === i && (
                    <span aria-hidden="true">{desc ? " ↓" : " ↑"}</span>
                  )}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, ri) => {
            const groupHeader =
              sortKey === -1 && (ri === 0 || sorted[ri - 1].group !== r.group);
            return (
              <GroupedRow
                key={r.model}
                row={r}
                columns={columns}
                groupHeader={groupHeader}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function GroupedRow({
  row,
  columns,
  groupHeader,
}: {
  row: Row;
  columns: string[];
  groupHeader: boolean;
}) {
  return (
    <>
      {groupHeader && (
        <tr>
          <th
            scope="colgroup"
            colSpan={columns.length + 1}
            className="bg-surface2 p-2 text-left text-sm uppercase tracking-wide text-muted"
          >
            {row.group}
          </th>
        </tr>
      )}
      <tr className="border-b border-line">
        <th scope="row" className="p-2 text-left font-semibold">
          {row.model}
        </th>
        {row.scores.map((s, i) => (
          <td
            key={i}
            className={`p-2 text-right tabular-nums ${
              row.best[i] ? "font-bold" : ""
            }`}
          >
            {row.best[i] ? <strong>{s.toFixed(2)}</strong> : s.toFixed(2)}
          </td>
        ))}
      </tr>
    </>
  );
}
