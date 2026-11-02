import { test, expect } from "@playwright/test";

/** The FigureQA and PlotQA sections: each lists its own dataset's questions,
 *  shows the chart on the right pane, and pins the selected (and, once run,
 *  answered) question directly under the chart image. */

test("figureqa: own section, chart on screen, question pinned under it", async ({
  page,
}) => {
  await page.goto("/figureqa");
  await expect(
    page.getByRole("heading", { name: "FigureQA Explorer" }),
  ).toBeVisible();

  // Only figureqa-* questions are listed (3 in the demo sample data).
  const list = page.getByRole("listbox");
  await expect(list.getByRole("option").first()).toContainText("figureqa-");
  await expect(list.getByRole("option", { name: /factoid-/ })).toHaveCount(0);

  // Selecting a question shows its chart…
  await page.getByRole("option", { name: /Dark Red the maximum/ }).click();
  await expect(page.getByRole("group", { name: /Chart for question/ })).toBeVisible();

  // …and pins the question under the chart, marked as not yet run.
  const pinned = page.getByRole("region", { name: "Selected question" });
  await expect(pinned).toBeVisible();
  await expect(pinned).toContainText("Is Dark Red the maximum?");
  await expect(pinned).toContainText("not run yet");

  // After the stored run replays, the pinned question says it was answered.
  await page.getByRole("button", { name: /Show stored result/ }).click();
  await expect(pinned).toContainText("Question — answered above");
  await expect(page.getByRole("region", { name: "Answer" })).toContainText("Yes");
});

test("plotqa: own section with its dataset's questions and stored replay", async ({
  page,
}) => {
  await page.goto("/plotqa");
  await expect(
    page.getByRole("heading", { name: "PlotQA Explorer" }),
  ).toBeVisible();

  const list = page.getByRole("listbox");
  await expect(list.getByRole("option").first()).toContainText("plotqa-");

  await page.getByRole("option", { name: /Brazil in 2010/ }).click();
  const pinned = page.getByRole("region", { name: "Selected question" });
  await expect(pinned).toContainText("What is the value of Brazil in 2010?");

  await page.getByRole("button", { name: /Show stored result/ }).click();
  await expect(page.getByRole("region", { name: "Answer" })).toContainText("37");
  await expect(pinned).toContainText("Question — answered above");
});

test("explorer: chartqapro section lists only chartqapro questions", async ({
  page,
}) => {
  await page.goto("/explorer");
  const list = page.getByRole("listbox");
  await expect(list.getByRole("option").first()).toContainText("factoid-");
  await expect(list.getByRole("option", { name: /figureqa-/ })).toHaveCount(0);
  await expect(list.getByRole("option", { name: /plotqa-/ })).toHaveCount(0);
});
