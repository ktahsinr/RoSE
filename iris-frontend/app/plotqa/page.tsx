import type { Metadata } from "next";
import Explorer, { type DatasetSection } from "../explorer/Explorer";

export const metadata: Metadata = { title: "PlotQA Explorer" };

const SECTION: DatasetSection = {
  key: "plotqa",
  title: "PlotQA Explorer",
  intro:
    "100 scientific plots from the PlotQA benchmark (Methani et al., 2020) " +
    "with short factoid questions read off the plotted values.",
};

export default function PlotQAPage() {
  return <Explorer section={SECTION} />;
}
