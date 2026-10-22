"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Logo from "./Logo";
import { TEXT_STEPS, useSettings, type Theme } from "@/lib/settings";
import { useAnnounce } from "@/lib/announcer";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/explorer", label: "Explorer" },
  { href: "/pdf", label: "PDF chart QA" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/about", label: "About" },
];

export default function Header() {
  const pathname = usePathname();
  const { settings, update } = useSettings();
  const announce = useAnnounce();

  const setStep = (delta: number | null) => {
    const step =
      delta === null
        ? 0
        : Math.min(Math.max(settings.textStep + delta, 0), TEXT_STEPS.length - 1);
    update({ textStep: step });
    announce(`Text size ${Math.round(TEXT_STEPS[step] * 100)} percent`);
  };

  return (
    <header className="border-b-2 border-line bg-surface">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link
          href="/"
          className="flex items-center gap-2 text-xl font-bold text-accent"
        >
          <Logo size={34} />
          IRIS
        </Link>

        <nav aria-label="Main">
          <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {NAV.map((item) => {
              const current =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={current ? "page" : undefined}
                    className={`rounded px-2 py-1 font-semibold ${
                      current
                        ? "bg-accent-soft text-accent underline underline-offset-4"
                        : "text-fg hover:bg-surface2"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
          <fieldset className="flex items-center gap-1">
            <legend className="sr-only">Text size</legend>
            <button
              type="button"
              onClick={() => setStep(-1)}
              disabled={settings.textStep === 0}
              aria-label="Decrease text size"
              className="rounded border-2 border-line px-2 py-0.5 font-bold hover:bg-surface2 disabled:opacity-40"
            >
              A−
            </button>
            <button
              type="button"
              onClick={() => setStep(null)}
              aria-label={`Reset text size, currently ${Math.round(
                TEXT_STEPS[settings.textStep] * 100,
              )} percent`}
              className="rounded border-2 border-line px-2 py-0.5 font-bold hover:bg-surface2"
            >
              A
            </button>
            <button
              type="button"
              onClick={() => setStep(1)}
              disabled={settings.textStep === TEXT_STEPS.length - 1}
              aria-label="Increase text size"
              className="rounded border-2 border-line px-2 py-0.5 font-bold hover:bg-surface2 disabled:opacity-40"
            >
              A+
            </button>
          </fieldset>

          <button
            type="button"
            aria-pressed={settings.dyslexicFont}
            onClick={() => {
              update({ dyslexicFont: !settings.dyslexicFont });
              announce(
                settings.dyslexicFont
                  ? "Dyslexia-friendly font off"
                  : "Dyslexia-friendly font on",
              );
            }}
            className={`rounded border-2 px-2 py-0.5 font-bold ${
              settings.dyslexicFont
                ? "border-accent bg-accent-soft text-accent"
                : "border-line hover:bg-surface2"
            }`}
          >
            Aa font
          </button>

          <label className="flex items-center gap-2 font-semibold">
            Theme
            <select
              value={settings.theme}
              onChange={(e) => update({ theme: e.target.value as Theme })}
              className="rounded border-2 border-line bg-bg px-2 py-0.5"
            >
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="hc">High contrast</option>
            </select>
          </label>
        </div>
      </div>
    </header>
  );
}
