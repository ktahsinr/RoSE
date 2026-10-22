import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import { AnnouncerProvider } from "@/lib/announcer";
import { SettingsProvider, BOOT_SCRIPT } from "@/lib/settings";

export const metadata: Metadata = {
  title: {
    default: "IRIS — Iterative Reasoning over Inferred Sub-queries",
    template: "%s · IRIS",
  },
  description:
    "IRIS is a chart question-answering assistant built for low-vision and visually impaired readers.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <head>
        <script dangerouslySetInnerHTML={{ __html: BOOT_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <a href="#main" className="skip-link">
          Skip to main content
        </a>
        <SettingsProvider>
          <AnnouncerProvider>
            <Header />
            <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
              {children}
            </main>
            <footer className="border-t-2 border-line bg-surface px-4 py-4 text-sm text-muted">
              <p className="mx-auto max-w-7xl">
                IRIS — Iterative Reasoning over Inferred Sub-queries. Built on
                the RoSE reasoning framework (Liu, He &amp; Qiu, EMNLP 2024).
                North South University CSE499B.
              </p>
            </footer>
          </AnnouncerProvider>
        </SettingsProvider>
      </body>
    </html>
  );
}
