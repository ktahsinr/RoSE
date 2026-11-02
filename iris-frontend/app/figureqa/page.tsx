import type { Metadata } from "next";
import Explorer, { type DatasetSection } from "../explorer/Explorer";

export const metadata: Metadata = { title: "FigureQA Explorer" };

const SECTION: DatasetSection = {
  key: "figureqa",
  title: "FigureQA Explorer",
  intro:
    "100 synthetic charts from the FigureQA benchmark (Kahou et al., 2018) " +
    "with yes/no questions about colors, maxima and comparisons.",
};

export default function FigureQAPage() {
  return <Explorer section={SECTION} />;
}
