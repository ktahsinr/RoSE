import type { Metadata } from "next";
import Explorer from "./Explorer";

export const metadata: Metadata = { title: "ChartQAPro Explorer" };

export default function ExplorerPage() {
  return <Explorer />;
}
