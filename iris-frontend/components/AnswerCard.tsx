"use client";

/** The answer, pinned above the trace: IRIS's answer vs the gold answer,
 *  the scorer's verdict, the agreement badge, and read-aloud. */

import ReadAloud from "./ReadAloud";
import { agreementLabel, type FinalStage } from "@/lib/trace";

function gold(groundTruth: string | string[] | undefined): string {
  if (groundTruth == null) return "";
  return Array.isArray(groundTruth) ? groundTruth.join(" | ") : String(groundTruth);
}

export default function AnswerCard({ final }: { final: FinalStage }) {
  const badge = agreementLabel(final.agreement);
  const goldText = gold(final.groundTruth);
  const spoken =
    `The answer is ${final.answer}. Confidence: ${badge.text}.` +
    (goldText
      ? ` The expected answer is ${goldText}; the scorer marks this ${
          final.isCorrect ? "correct" : "incorrect"
        }.`
      : "");

  return (
    <section
      aria-label="Answer"
      className="rounded-lg border-2 border-amber bg-surface p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h2 className="text-lg font-bold">Answer</h2>
        <ReadAloud text={spoken} label="the answer" />
      </div>
      <p className="my-1 text-2xl font-bold">{final.answer || "—"}</p>
      <div className="flex flex-wrap items-center gap-2">
        {goldText && (
          <span
            className={`rounded-full border-2 px-3 py-0.5 font-bold ${
              final.isCorrect
                ? "border-ok bg-ok-soft text-ok"
                : "border-err bg-err-soft text-err"
            }`}
          >
            {final.isCorrect ? "✓" : "✗"} gold: {goldText} —{" "}
            {final.isCorrect ? "correct" : "incorrect"} per the scorer
          </span>
        )}
        <span
          className={`rounded-full border-2 px-3 py-0.5 font-semibold ${
            badge.tone === "ok"
              ? "border-ok bg-ok-soft text-ok"
              : badge.tone === "mid"
                ? "border-amber bg-amber-soft text-amber"
                : "border-err bg-err-soft text-err"
          }`}
        >
          {badge.text}
        </span>
      </div>
      {!goldText && (
        <p className="mt-2 text-sm text-muted">
          No gold answer for this chart — use the agreement badge as the
          confidence indicator and verify against the chart.
        </p>
      )}
    </section>
  );
}
