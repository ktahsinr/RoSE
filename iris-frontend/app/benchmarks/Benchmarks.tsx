"use client";

// NOTE / verify before release: the column headers rendered here come from
// the backend's transcription of the ChartQAPro paper (Masry et al., 2025)
// — Factoid, Multiple Choice, Hypothetical, Fact Checking, Conversational,
// Overall. They must be checked against the paper's table before release
// (see iris_backend/benchmarks.py, which carries the same flag).

import { useEffect, useState } from "react";
import { getJSON, ApiError, type BenchmarksPayload } from "@/lib/api";
import BarChart from "@/components/BarChart";
import BenchmarkTable from "@/components/BenchmarkTable";

export default function Benchmarks() {
  const [data, setData] = useState<BenchmarksPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getJSON<BenchmarksPayload>("/benchmarks")
      .then(setData)
      .catch((e: ApiError) => setError(e.message));
  }, []);

  if (error)
    return (
      <p
        role="alert"
        className="rounded-lg border-2 border-err bg-err-soft p-3 font-semibold text-err"
      >
        {error}
      </p>
    );
  if (!data) return <p>Loading benchmark data…</p>;

  const groups = ["Closed-source", "Open-source, small", "Open-source, large"].map(
    (g) => ({
      title: g,
      rows: data.published.rows
        .filter((r) => r.group === g)
        .map((r) => ({
          label: r.model,
          value: r.scores[r.scores.length - 1],
          detail: "Overall accuracy",
        })),
    }),
  );

  const irisGroups = [
    {
      rows: data.iris.rows.map((r) => ({
        label: r.label,
        value: r.accuracy_pct,
        ci: r.ci,
        detail: `${r.correct}/${r.n} correct${r.settings ? ` · ${r.settings}` : ""}`,
      })),
    },
  ];

  return (
    <div className="flex flex-col gap-10">
      <h1 className="text-2xl font-bold">Benchmarks</h1>

      <section aria-labelledby="published-h" className="flex flex-col gap-4">
        <h2 id="published-h" className="text-xl font-bold">
          Published ChartQAPro results
        </h2>
        <p className="max-w-3xl">
          Accuracy of closed- and open-source vision-language models on
          ChartQAPro, as reported in the ChartQAPro paper (Masry et al.,
          2025).
        </p>
        <BenchmarkTable
          columns={data.published.columns}
          rows={data.published.rows}
        />
        <h3 className="font-bold">Overall accuracy at a glance</h3>
        <BarChart
          groups={groups}
          maxValue={50}
          ariaLabel={
            "Horizontal bar chart of each model's Overall accuracy on " +
            "ChartQAPro, grouped as in the paper. The full numbers are in " +
            "the table above. Best overall: Gemini-Flash-2.0 at 46.85%."
          }
        />
      </section>

      <section aria-labelledby="iris-h" className="flex flex-col gap-4">
        <h2 id="iris-h" className="text-xl font-bold">
          {data.iris.title}
        </h2>
        <BarChart
          groups={irisGroups}
          maxValue={60}
          ariaLabel={
            "Horizontal bar chart of the IRIS arms' accuracy on the first " +
            "250 ChartQAPro factoid questions, with 95% confidence " +
            "intervals as whiskers. The numbers follow in the table below."
          }
        />
        <div className="overflow-x-auto rounded-lg border-2 border-line">
          <table className="w-full min-w-[560px] border-collapse">
            <caption className="p-2 text-left text-sm text-muted">
              IRIS arms on ChartQAPro factoid, first 250 questions, with 95%
              confidence intervals.
            </caption>
            <thead>
              <tr className="border-b-2 border-line bg-surface">
                <th scope="col" className="p-2 text-left">Arm</th>
                <th scope="col" className="p-2 text-right">Accuracy</th>
                <th scope="col" className="p-2 text-right">Correct</th>
                <th scope="col" className="p-2 text-right">95% CI</th>
                <th scope="col" className="p-2 text-left">Settings</th>
              </tr>
            </thead>
            <tbody>
              {data.iris.rows.map((r) => (
                <tr key={r.key} className="border-b border-line">
                  <th scope="row" className="p-2 text-left font-semibold">
                    {r.label}
                  </th>
                  <td className="p-2 text-right font-bold tabular-nums">
                    {r.accuracy_pct}%
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {r.correct}/{r.n}
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {r.ci[0]}–{r.ci[1]}
                  </td>
                  <td className="p-2 text-left">{r.settings || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="max-w-3xl rounded-lg border-2 border-line bg-surface p-3 text-sm">
          <strong>Footnote:</strong> {data.iris.footnote}
        </p>
      </section>
    </div>
  );
}
