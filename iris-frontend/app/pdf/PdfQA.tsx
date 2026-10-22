"use client";

/** PDF Chart QA: upload a PDF, pick a page, confirm a chart region (detected
 *  or hand-drawn), then ask factoid questions against the crop — several in a
 *  row without re-uploading. Results reuse the Explorer's two-pane layout;
 *  with no gold answer the agreement badge is the confidence indicator. */

import { useEffect, useState } from "react";
import {
  apiUrl,
  getJSON,
  postForm,
  postJSON,
  ApiError,
  type Health,
  type ModelInfo,
} from "@/lib/api";
import { useLiveRun } from "@/lib/useLiveRun";
import { useAnnounce } from "@/lib/announcer";
import AnswerCard from "@/components/AnswerCard";
import ChartViewer from "@/components/ChartViewer";
import CropStage, { type Box, type Candidate } from "@/components/CropStage";
import ModelPanel, { type ModelChoice } from "@/components/ModelPanel";
import RunStatus from "@/components/RunStatus";
import Trace from "@/components/Trace";
import { agreementLabel } from "@/lib/trace";

interface PdfMeta {
  pdf_id: string;
  pages: number;
  thumbnail_urls: string[];
}

interface Crop {
  image_id: string;
  url: string;
}

interface Past {
  question: string;
  answer: string;
  agreement: number | null;
}

export default function PdfQA() {
  const announce = useAnnounce();
  const { trace, running, startedAt, start } = useLiveRun();

  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [pdf, setPdf] = useState<PdfMeta | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [page, setPage] = useState<number | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [crop, setCrop] = useState<Crop | null>(null);

  const [question, setQuestion] = useState("");
  const [choice, setChoice] = useState<ModelChoice>({
    planners: ["gemma3-4b", "qwen3.5-4b"],
    finalModel: "qwen2.5-vl-7b",
    fastMode: true,
  });
  const [past, setPast] = useState<Past[]>([]);

  useEffect(() => {
    Promise.all([getJSON<Health>("/health"), getJSON<ModelInfo[]>("/models")])
      .then(([h, m]) => {
        setHealth(h);
        setModels(m);
      })
      .catch((e: ApiError) => setError(e.message));
  }, []);

  const upload = async (file: File) => {
    setError(null);
    setPdf(null);
    setPage(null);
    setBox(null);
    setCrop(null);
    setPast([]);
    const form = new FormData();
    form.append("file", file);
    try {
      const meta = await postForm<PdfMeta>("/pdf", form);
      setPdf(meta);
      setPage(0);
      announce(
        `PDF uploaded: ${meta.pages} page${meta.pages === 1 ? "" : "s"}. Detecting charts…`,
      );
      const out = await postJSON<{ candidates: Candidate[] }>(
        `/pdf/${meta.pdf_id}/extract`,
        {},
      );
      setCandidates(out.candidates);
      announce(
        out.candidates.length
          ? `${out.candidates.length} candidate chart${
              out.candidates.length === 1 ? "" : "s"
            } detected across the document.`
          : "No charts auto-detected — you can draw a box on any page.",
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const confirmCrop = async () => {
    if (!pdf || page === null || !box) return;
    setError(null);
    try {
      const out = await postJSON<Crop>(`/pdf/${pdf.pdf_id}/crop`, {
        page,
        bbox: box,
      });
      setCrop(out);
      setPast([]);
      announce("Chart confirmed. Type a question about it.");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const runQuestion = () => {
    if (!crop || !question.trim()) return;
    start(
      {
        image_id: crop.image_id,
        question: question.trim(),
        planners: choice.planners,
        final_model: choice.finalModel,
        fast_mode: choice.fastMode,
      },
      (finished) => {
        if (finished.final)
          setPast((p) => [
            ...p,
            {
              question: question.trim(),
              answer: finished.final!.answer,
              agreement: finished.final!.agreement,
            },
          ]);
      },
    );
  };

  const demoMode = health?.mode === "demo";

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold">PDF Chart QA</h1>
      <p className="max-w-3xl">
        Upload a PDF, confirm the chart you care about, and ask IRIS factoid
        questions about it: a value, a label, a year, a comparison.
      </p>

      {error && (
        <p
          role="alert"
          className="rounded-lg border-2 border-err bg-err-soft p-3 font-semibold text-err"
        >
          {error}
        </p>
      )}
      {demoMode && (
        <p className="rounded-lg border-2 border-amber bg-amber-soft p-3 font-semibold text-amber">
          Demo mode: no GPU is available, so questions cannot run. You can
          still upload a PDF and try the chart detection and cropping.
        </p>
      )}

      {/* Step 1 — upload */}
      <section aria-label="Step 1: upload" className="flex flex-col gap-2">
        <h2 className="text-lg font-bold">1 · Upload a PDF</h2>
        <label className="w-fit cursor-pointer rounded-lg border-2 border-accent bg-accent-soft px-4 py-2 font-bold text-accent">
          Choose a PDF file
          <input
            type="file"
            accept="application/pdf,.pdf"
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) upload(f);
            }}
          />
        </label>
        {pdf && (
          <p className="text-muted">
            Uploaded: {pdf.pages} page{pdf.pages === 1 ? "" : "s"}.
          </p>
        )}
      </section>

      {/* Step 2 — pick a page */}
      {pdf && (
        <section aria-label="Step 2: pick a page" className="flex flex-col gap-2">
          <h2 className="text-lg font-bold">2 · Pick the page with the chart</h2>
          <ul className="flex flex-wrap gap-2">
            {pdf.thumbnail_urls.map((u, i) => {
              const n = candidates.filter((c) => c.page === i).length;
              return (
                <li key={u}>
                  <button
                    type="button"
                    aria-pressed={page === i}
                    aria-label={`Page ${i + 1}${
                      n ? `, ${n} detected chart${n === 1 ? "" : "s"}` : ""
                    }`}
                    onClick={() => {
                      setPage(i);
                      setBox(null);
                    }}
                    className={`rounded border-4 p-1 ${
                      page === i ? "border-accent" : "border-line"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={apiUrl(u)} alt="" className="h-28 w-auto" />
                    <span className="block text-sm font-semibold">
                      p. {i + 1}
                      {n > 0 && <span aria-hidden="true"> · {n} 📊</span>}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {/* Step 3 — crop */}
      {pdf && page !== null && (
        <section aria-label="Step 3: choose the chart region" className="flex flex-col gap-2">
          <h2 className="text-lg font-bold">3 · Choose the chart region</h2>
          <CropStage
            pdfId={pdf.pdf_id}
            page={page}
            candidates={candidates}
            crop={box}
            onCrop={setBox}
          />
          <button
            type="button"
            onClick={confirmCrop}
            disabled={!box}
            className="w-fit rounded-lg border-2 border-amber bg-amber-soft px-4 py-2 font-bold text-amber hover:opacity-90 disabled:opacity-40"
          >
            ✓ Confirm this chart
          </button>
        </section>
      )}

      {/* Step 4 — ask */}
      {crop && (
        <section aria-label="Step 4: ask questions" className="flex flex-col gap-4">
          <h2 className="text-lg font-bold">4 · Ask questions about the chart</h2>

          <RunStatus
            statusText={trace.statusText}
            stage={trace.stage}
            running={running}
            startedAt={startedAt}
            error={trace.error}
          />

          <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
            <div className="min-w-0 lg:sticky lg:top-4 lg:order-2 lg:self-start">
              <ChartViewer
                src={apiUrl(crop.url)}
                description={trace.chartDescription}
                title="Your confirmed chart"
              />
            </div>

            <div className="flex min-w-0 flex-col gap-5 lg:order-1">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  runQuestion();
                }}
                className="flex flex-col gap-2"
              >
                <label className="flex flex-col gap-1">
                  <span className="font-semibold">Your question</span>
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="e.g. In which year did the line peak?"
                    className="rounded border-2 border-line bg-bg px-2 py-2"
                  />
                </label>
                <p className="text-sm text-muted">
                  IRIS answers factoid questions best: ask for a value, a
                  label, a year, or a comparison readable from this chart.
                </p>
                <ModelPanel
                  models={models}
                  choice={choice}
                  onChange={setChoice}
                  disabled={demoMode || running}
                />
                <button
                  type="submit"
                  disabled={
                    demoMode ||
                    running ||
                    !question.trim() ||
                    choice.planners.length === 0
                  }
                  className="w-fit rounded-lg border-2 border-amber bg-amber-soft px-4 py-2 font-bold text-amber hover:opacity-90 disabled:opacity-40"
                >
                  ▶ Ask IRIS
                </button>
              </form>

              {trace.final && <AnswerCard final={trace.final} />}
              <Trace trace={trace} />

              {past.length > 0 && (
                <section aria-label="Previous questions on this chart">
                  <h3 className="font-bold">Previous questions on this chart</h3>
                  <ul className="mt-1 flex flex-col gap-1">
                    {past.map((p, i) => (
                      <li key={i} className="rounded border border-line p-2">
                        <span className="font-semibold">{p.question}</span> →{" "}
                        {p.answer}{" "}
                        <span className="text-sm text-muted">
                          ({agreementLabel(p.agreement).text})
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
