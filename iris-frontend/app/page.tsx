import Link from "next/link";
import Logo from "@/components/Logo";

export default function Home() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6">
      <div className="flex items-center gap-4">
        <Logo size={72} />
        <div>
          <h1 className="text-3xl font-bold">IRIS</h1>
          <p className="text-lg text-muted">
            Iterative Reasoning over Inferred Sub-queries
          </p>
        </div>
      </div>
      <p className="text-lg">
        IRIS answers questions about charts, built from the ground up for
        low-vision and visually impaired readers.
      </p>
      <p>
        <Link
          href="/explorer"
          className="rounded-lg border-2 border-accent bg-accent-soft px-4 py-2 font-bold text-accent"
        >
          Open the Explorer
        </Link>
      </p>
      <p className="text-muted">
        The full home page (how IRIS answers, accessibility features and
        credits) lands with the Benchmarks step of this build.
      </p>
    </div>
  );
}
