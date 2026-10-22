import type { Metadata } from "next";
import Benchmarks from "./Benchmarks";

export const metadata: Metadata = { title: "Benchmarks" };

export default function BenchmarksPage() {
  return <Benchmarks />;
}
