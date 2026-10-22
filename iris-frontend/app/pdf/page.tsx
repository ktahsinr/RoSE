import type { Metadata } from "next";
import PdfQA from "./PdfQA";

export const metadata: Metadata = { title: "PDF Chart QA" };

export default function PdfPage() {
  return <PdfQA />;
}
