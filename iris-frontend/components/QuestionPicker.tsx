"use client";

/** Searchable, filterable, keyboard-navigable question list (a listbox with
 *  roving tabindex: arrows move, Home/End jump, Enter/Space select). */

import { useEffect, useRef, useState } from "react";
import type { QuestionItem } from "@/lib/api";

export default function QuestionPicker({
  items,
  total,
  selectedId,
  onSelect,
  search,
  onSearch,
  typeFilter,
  onTypeFilter,
  types,
}: {
  items: QuestionItem[];
  total: number;
  selectedId: string | null;
  onSelect: (q: QuestionItem) => void;
  search: string;
  onSearch: (v: string) => void;
  typeFilter: string;
  onTypeFilter: (v: string) => void;
  types: string[];
}) {
  const listRef = useRef<HTMLUListElement>(null);
  const [focusIdx, setFocusIdx] = useState(0);

  useEffect(() => {
    setFocusIdx((i) => Math.min(i, Math.max(items.length - 1, 0)));
  }, [items.length]);

  const focusOption = (idx: number) => {
    const clamped = Math.min(Math.max(idx, 0), items.length - 1);
    setFocusIdx(clamped);
    const el = listRef.current?.children[clamped] as HTMLElement | undefined;
    el?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent, idx: number) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        focusOption(idx + 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        focusOption(idx - 1);
        break;
      case "Home":
        e.preventDefault();
        focusOption(0);
        break;
      case "End":
        e.preventDefault();
        focusOption(items.length - 1);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        onSelect(items[idx]);
        break;
    }
  };

  return (
    <section aria-label="Pick a question" className="flex flex-col gap-2">
      <h2 className="text-lg font-bold">Pick a question</h2>
      <div className="flex flex-wrap gap-2">
        <label className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="text-sm font-semibold">Search questions</span>
          <input
            type="search"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
            placeholder="e.g. revenue, rate, share…"
            className="w-full rounded border-2 border-line bg-bg px-2 py-1"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold">Type</span>
          <select
            value={typeFilter}
            onChange={(e) => onTypeFilter(e.target.value)}
            className="rounded border-2 border-line bg-bg px-2 py-1"
          >
            <option value="">All types</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="text-sm text-muted" aria-live="off">
        {total} question{total === 1 ? "" : "s"} · ⚡ marks a stored result
      </p>

      <ul
        ref={listRef}
        role="listbox"
        aria-label={`Questions, ${total} available`}
        className="max-h-72 overflow-y-auto rounded-lg border-2 border-line"
      >
        {items.map((q, idx) => {
          const selected = q.id === selectedId;
          return (
            <li
              key={q.id}
              role="option"
              aria-selected={selected}
              tabIndex={idx === focusIdx ? 0 : -1}
              onKeyDown={(e) => onKeyDown(e, idx)}
              onClick={() => onSelect(q)}
              className={`cursor-pointer border-b border-line px-3 py-2 last:border-b-0 ${
                selected
                  ? "border-l-4 border-l-accent bg-accent-soft font-semibold"
                  : "hover:bg-surface2"
              }`}
            >
              <span className="mr-2 text-sm text-muted">{q.id}</span>
              {Object.values(q.has_replay).some(Boolean) && (
                <span aria-label="stored result available" title="Stored result">
                  ⚡{" "}
                </span>
              )}
              {q.question}
            </li>
          );
        })}
        {items.length === 0 && (
          <li className="px-3 py-2 text-muted">No questions match.</li>
        )}
      </ul>
    </section>
  );
}
