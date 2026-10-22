import { test, expect, type Page } from "@playwright/test";

/** The Explorer completed with the keyboard alone: skip link → search →
 *  listbox navigation → selection → stored-result replay → answer verified
 *  against the gold — no pointer input anywhere. */

async function tabTo(page: Page, matcher: () => Promise<boolean>, max = 40) {
  for (let i = 0; i < max; i++) {
    if (await matcher()) return;
    await page.keyboard.press("Tab");
  }
  throw new Error("could not reach the target with Tab");
}

test("explorer: keyboard-only from landing to verified answer", async ({
  page,
}) => {
  await page.goto("/explorer");

  // 1 — the skip link is the first tab stop and moves focus to main.
  await page.keyboard.press("Tab");
  const skip = page.locator(".skip-link");
  await expect(skip).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main$/);

  // 2 — reach the search box and search.
  const search = page.getByRole("searchbox", { name: "Search questions" });
  await tabTo(page, () =>
    search.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.type("revenue");
  const option = page.getByRole("option", { name: /revenue peak/ });
  await expect(option).toBeVisible();

  // 3 — into the listbox: arrow keys + Enter select the question.
  await tabTo(page, () =>
    page.evaluate(() => document.activeElement?.getAttribute("role") === "option"),
  );
  await page.keyboard.press("Enter");
  await expect(option).toHaveAttribute("aria-selected", "true");
  await expect(
    page.getByRole("img", { name: /bar chart titled/i }),
  ).toBeVisible();

  // 4 — demo mode: Run live is disabled and the banner says why.
  await expect(page.getByText(/Demo mode/)).toBeVisible();
  await expect(page.getByRole("button", { name: "▶ Run live" })).toBeDisabled();

  // 5 — replay the stored result from the keyboard.
  const stored = page.getByRole("button", { name: /Show stored result/ });
  await tabTo(page, () =>
    stored.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.press("Enter");

  // 6 — the answer is verifiable against the gold on the same screen.
  const answer = page.getByRole("region", { name: "Answer" });
  await expect(answer).toBeVisible();
  await expect(answer).toContainText("2019");
  await expect(answer).toContainText(/correct per the scorer/);
  await expect(answer).toContainText(/confidence/);

  // 7 — the trace shows the planners, the vote and the paths.
  const trace = page.getByRole("region", { name: "Reasoning trace" });
  await expect(trace).toContainText("Plan — Gemma 3 4B");
  await expect(trace).toContainText("Plan — Qwen3.5 4B");
  await expect(trace).toContainText("Vote — choosing the plan");
  await expect(trace).toContainText(/reasoning paths \(3/);

  // 8 — the live region announced the replay.
  await expect(page.locator("[aria-live='polite']")).toContainText(
    /stored result/i,
  );

  // 9 — the chart description doubles as the text alternative.
  await expect(
    page.getByRole("heading", { name: "Chart description", exact: true }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("region", { name: "Chart" })
      .getByText(/vertical bar chart titled/),
  ).toBeVisible();
});

test("explorer: chart viewer zooms and pans from the keyboard", async ({
  page,
}) => {
  await page.goto("/explorer");
  await page.getByRole("option", { name: /revenue peak/ }).click();
  const viewer = page.getByRole("group", { name: /Chart for question/ });
  await viewer.focus();
  await page.keyboard.press("+");
  await expect(viewer).toHaveAccessibleName(/Zoom 150 percent/);
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("0");
  await expect(viewer).toHaveAccessibleName(/Zoom 100 percent/);
});

test("explorer: text size control scales the page and persists", async ({
  page,
}) => {
  await page.goto("/explorer");
  const bigger = page.getByRole("button", { name: "Increase text size" });
  await bigger.click();
  await bigger.click();
  const size = await page.evaluate(
    () => getComputedStyle(document.documentElement).fontSize,
  );
  expect(parseFloat(size)).toBeCloseTo(18 * 1.25, 1);
  await page.reload();
  const after = await page.evaluate(
    () => getComputedStyle(document.documentElement).fontSize,
  );
  expect(parseFloat(after)).toBeCloseTo(18 * 1.25, 1);
});
