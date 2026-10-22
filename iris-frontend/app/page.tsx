import Link from "next/link";
import Logo from "@/components/Logo";

export default function Home() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <div className="flex flex-wrap items-center gap-4">
        <Logo size={80} />
        <div>
          <h1 className="text-3xl font-bold">IRIS</h1>
          <p className="text-lg text-muted">
            Iterative Reasoning over Inferred Sub-queries
          </p>
        </div>
      </div>

      <section aria-labelledby="who-h" className="flex flex-col gap-2">
        <h2 id="who-h" className="text-xl font-bold">
          Charts, answered — for readers who can&apos;t see them well
        </h2>
        <p>
          IRIS answers questions about charts: a value, a label, a year, a
          comparison. It is built for low-vision and visually impaired
          readers — every answer comes with a plain-language description of
          the chart, a confidence indicator, and the full reasoning behind
          it, all readable by screen reader or read aloud by the browser.
        </p>
      </section>

      <section aria-labelledby="how-h" className="flex flex-col gap-2">
        <h2 id="how-h" className="text-xl font-bold">How IRIS answers</h2>
        <p>
          Two small models each read the chart and propose a two-step plan
          for answering your question. A vote compares the plans: if they
          agree, they are merged; if not, the plan that stays closer to your
          question wins. A larger model then answers step by step, following
          the chosen plan, trying several lines of reasoning and keeping the
          answer most of them agree on — the more they agree, the higher the
          confidence IRIS reports.
        </p>
      </section>

      <section aria-labelledby="try-h" className="flex flex-col gap-3">
        <h2 id="try-h" className="text-xl font-bold">Try it</h2>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/explorer"
            className="rounded-lg border-2 border-accent bg-accent-soft px-4 py-2 font-bold text-accent"
          >
            Explore ChartQAPro questions
          </Link>
          <Link
            href="/pdf"
            className="rounded-lg border-2 border-accent bg-accent-soft px-4 py-2 font-bold text-accent"
          >
            Ask about a chart in your PDF
          </Link>
          <Link
            href="/benchmarks"
            className="rounded-lg border-2 border-line px-4 py-2 font-bold"
          >
            See the benchmarks
          </Link>
        </div>
      </section>

      <section aria-labelledby="a11y-h" className="flex flex-col gap-2">
        <h2 id="a11y-h" className="text-xl font-bold">
          Accessibility controls
        </h2>
        <ul className="list-disc pl-6">
          <li>
            <strong>Text size</strong>: the A− / A / A+ buttons in the header
            scale all text from 18px up to 200%, and the layout reflows.
          </li>
          <li>
            <strong>Themes</strong>: light, dark and high-contrast; by
            default IRIS follows your system preference. Text contrast meets
            WCAG AAA.
          </li>
          <li>
            <strong>Dyslexia-friendly font</strong>: the “Aa font” toggle
            switches to OpenDyslexic (the default face is Atkinson
            Hyperlegible, designed for low-vision readers).
          </li>
          <li>
            <strong>Read aloud</strong>: every answer and chart description
            has a Read aloud button with a stop control, using your
            browser&apos;s own speech — nothing leaves your machine.
          </li>
          <li>
            <strong>Keyboard and screen reader</strong>: everything works
            without a mouse — skip link, visible focus rings, and each
            pipeline stage announced politely as it finishes.
          </li>
          <li>
            <strong>Chart viewer</strong>: zoom to 400%, pan with arrow keys,
            invert colors — and IRIS&apos;s own text description of the chart
            is always alongside the image.
          </li>
          <li>
            <strong>Reduced motion</strong>: all animation (including the
            logo&apos;s pupil) stops when your system asks for reduced motion.
          </li>
        </ul>
      </section>

      <section aria-labelledby="credits-h" className="flex flex-col gap-2">
        <h2 id="credits-h" className="text-xl font-bold">Credits</h2>
        <p>
          IRIS is built on the <strong>RoSE</strong> reasoning framework
          (Liu, He &amp; Qiu, EMNLP 2024) and evaluated on the{" "}
          <strong>ChartQAPro</strong> benchmark (Masry et al., 2025).
          Developed as a North South University CSE499B capstone, supervised
          by Dr. Md Adnan Arefeen.{" "}
          <Link href="/about" className="font-semibold text-accent underline">
            Full credits on the About page.
          </Link>
        </p>
      </section>
    </div>
  );
}
