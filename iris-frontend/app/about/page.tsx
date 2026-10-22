import type { Metadata } from "next";
import Logo from "@/components/Logo";

export const metadata: Metadata = { title: "About" };

export default function AboutPage() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <div className="flex items-center gap-4">
        <Logo size={56} />
        <h1 className="text-2xl font-bold">About IRIS</h1>
      </div>

      <section aria-labelledby="rose-h" className="flex flex-col gap-2">
        <h2 id="rose-h" className="text-xl font-bold">
          The reasoning framework: RoSE
        </h2>
        <p>
          IRIS&apos;s reasoning engine is <strong>RoSE</strong> —{" "}
          <em>
            Making Large Language Models Better Reasoners with Orchestrated
            Streaming Experiences
          </em>{" "}
          by Qinhao Liu, Zhen He and Xipeng Qiu (Fudan University), published
          at EMNLP 2024. RoSE lets a model improve as it answers: every
          answered question joins a streaming experience pool, and new
          questions retrieve diverse, confident, complex past experiences as
          demonstrations. IRIS extends RoSE to chart images and adds query
          decomposition: small models plan, a vote picks the plan, and the
          large model answers along it. RoSE is the framework; IRIS is the
          product built on it.
        </p>
      </section>

      <section aria-labelledby="data-h" className="flex flex-col gap-2">
        <h2 id="data-h" className="text-xl font-bold">Benchmark and data</h2>
        <p>
          Questions and charts in the Explorer come from{" "}
          <strong>ChartQAPro</strong> (Masry et al., 2025), a benchmark of
          real-world chart questions. The published results table on the
          Benchmarks page is transcribed from that paper.
        </p>
      </section>

      <section aria-labelledby="models-h" className="flex flex-col gap-2">
        <h2 id="models-h" className="text-xl font-bold">Models</h2>
        <p>
          IRIS runs entirely on open-weight models, loaded locally in 4-bit:
          Gemma 3 4B-it and Qwen3.5-4B as planners, Qwen2.5-VL-7B-Instruct as
          the answering model, and all-mpnet-base-v2 for the embedding vote
          and experience retrieval. There are no cloud APIs anywhere: text to
          speech is your browser&apos;s own, and the frontend talks only to
          the IRIS backend.
        </p>
      </section>

      <section aria-labelledby="team-h" className="flex flex-col gap-2">
        <h2 id="team-h" className="text-xl font-bold">Team</h2>
        <p>
          IRIS is a <strong>North South University</strong> CSE499B capstone
          project, supervised by <strong>Dr. Md Adnan Arefeen</strong>, built
          by Kazi Tahsin Raihan and team.
        </p>
      </section>
    </div>
  );
}
