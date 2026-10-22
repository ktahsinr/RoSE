import AxeBuilder from "@axe-core/playwright";
import { test, expect, type Page } from "@playwright/test";

/** axe-core over every page, in the light, dark and high-contrast themes.
 *  WCAG 2.2 AA is the gate here; text contrast is designed to AAA and the
 *  token pairs are verified numerically in the design (see globals.css). */

const PAGES = ["/", "/explorer", "/pdf", "/benchmarks", "/about"];
const THEMES = ["light", "dark", "hc"] as const;

async function analyze(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  return results.violations;
}

for (const path of PAGES) {
  for (const theme of THEMES) {
    test(`axe: ${path} (${theme})`, async ({ page }) => {
      await page.goto(path);
      await page.evaluate(
        (t) => document.documentElement.setAttribute("data-theme", t),
        theme,
      );
      // Let the explorer finish its data fetches before scanning.
      await page.waitForLoadState("networkidle");
      const violations = await analyze(page);
      expect(
        violations.map((v) => ({
          id: v.id,
          impact: v.impact,
          nodes: v.nodes.map((n) => n.target.join(" ")).slice(0, 5),
        })),
      ).toEqual([]);
    });
  }
}

test("axe: explorer with a stored result shown", async ({ page }) => {
  await page.goto("/explorer");
  await page.getByRole("option", { name: /revenue peak/ }).click();
  await page.getByRole("button", { name: /Show stored result/ }).click();
  await page.getByRole("region", { name: "Answer" }).waitFor();
  const violations = await analyze(page);
  expect(violations.map((v) => v.id)).toEqual([]);
});
