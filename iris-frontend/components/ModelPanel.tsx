"use client";

/** Model selection: planners as checkboxes, the final model as a radio
 *  group, and the fast / spec-exact switch. */

import type { ModelInfo } from "@/lib/api";

export interface ModelChoice {
  planners: string[];
  finalModel: string;
  fastMode: boolean;
}

export default function ModelPanel({
  models,
  choice,
  onChange,
  disabled,
}: {
  models: ModelInfo[];
  choice: ModelChoice;
  onChange: (c: ModelChoice) => void;
  disabled: boolean;
}) {
  const planners = models.filter((m) => m.role === "planner");
  const finals = models.filter((m) => m.role === "final");

  const togglePlanner = (key: string) => {
    const has = choice.planners.includes(key);
    onChange({
      ...choice,
      planners: has
        ? choice.planners.filter((p) => p !== key)
        : [...choice.planners, key],
    });
  };

  return (
    <section aria-label="Models" className="flex flex-col gap-3">
      <h2 className="text-lg font-bold">Models</h2>
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <fieldset disabled={disabled} className="min-w-0">
          <legend className="text-sm font-semibold">
            Planner models (propose sub-queries)
          </legend>
          {planners.map((m) => (
            <label key={m.key} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={choice.planners.includes(m.key)}
                onChange={() => togglePlanner(m.key)}
                className="size-5 accent-[var(--accent)]"
              />
              {m.label}{" "}
              <span className="text-sm text-muted">({m.params_b}B)</span>
            </label>
          ))}
          {choice.planners.length === 0 && (
            <p role="alert" className="mt-1 text-sm font-semibold text-err">
              Pick at least one planner.
            </p>
          )}
        </fieldset>

        <fieldset disabled={disabled} className="min-w-0">
          <legend className="text-sm font-semibold">
            Final model (answers with RoSE)
          </legend>
          {finals.map((m) => (
            <label key={m.key} className="flex items-center gap-2">
              <input
                type="radio"
                name="final-model"
                checked={choice.finalModel === m.key}
                onChange={() => onChange({ ...choice, finalModel: m.key })}
                className="size-5 accent-[var(--accent)]"
              />
              {m.label}{" "}
              <span className="text-sm text-muted">({m.params_b}B)</span>
            </label>
          ))}
        </fieldset>

        <fieldset disabled={disabled} className="min-w-0">
          <legend className="text-sm font-semibold">Speed</legend>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              role="switch"
              checked={choice.fastMode}
              onChange={() =>
                onChange({ ...choice, fastMode: !choice.fastMode })
              }
              className="size-5 accent-[var(--accent)]"
            />
            Fast mode
          </label>
          <p className="max-w-60 text-sm text-muted">
            Fast: 3 reasoning paths. Off: the spec-exact pipeline (5 paths) —
            slower but matches the published runs.
          </p>
        </fieldset>
      </div>
    </section>
  );
}
