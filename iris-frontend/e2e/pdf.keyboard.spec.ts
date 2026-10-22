import { test, expect, type Page } from "@playwright/test";
import path from "node:path";

/** The PDF flow completed with the keyboard alone: upload (the file input
 *  is a real tab stop; the file itself is set programmatically, as a file
 *  dialog cannot be driven in automation), page choice, candidate crop,
 *  keyboard adjustment of the box, confirm, and the question form. Demo
 *  mode, so the run itself is (visibly) disabled. */

const FIXTURE = path.join(__dirname, "fixtures", "sample.pdf");

async function tabTo(page: Page, matcher: () => Promise<boolean>, max = 40) {
  for (let i = 0; i < max; i++) {
    if (await matcher()) return;
    await page.keyboard.press("Tab");
  }
  throw new Error("could not reach the target with Tab");
}

test("pdf: keyboard-only upload → crop → confirm → question form", async ({
  page,
}) => {
  await page.goto("/pdf");

  // 1 — the hidden-but-focusable file input is reachable by Tab.
  const input = page.locator("input[type=file]");
  await tabTo(page, () =>
    input.evaluate((el) => el === document.activeElement),
  );
  await input.setInputFiles(FIXTURE);
  await expect(page.getByText(/Uploaded: 1 page/)).toBeVisible();

  // 2 — pick the page from the keyboard.
  const pageBtn = page.getByRole("button", { name: /^Page 1/ });
  await expect(pageBtn).toBeVisible();
  await tabTo(page, () =>
    pageBtn.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.press("Enter");

  // 3 — the drawn bar chart was detected; select the candidate.
  const candidate = page.getByRole("button", { name: /Candidate 1/ });
  await tabTo(page, () =>
    candidate.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.press("Enter");

  // 4 — the crop box is focusable and the arrows move it.
  const box = page.getByRole("group", { name: /Crop box at/ });
  await box.focus();
  const before = await box.getAttribute("aria-label");
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(async () => box.getAttribute("aria-label"))
    .not.toBe(before);

  // The coordinates are also plain number inputs.
  const left = page.getByRole("spinbutton", { name: "Left" });
  await expect(left).toBeEnabled();

  // 5 — confirm the chart.
  const confirm = page.getByRole("button", { name: /Confirm this chart/ });
  await tabTo(page, () =>
    confirm.evaluate((el) => el === document.activeElement),
  );
  await page.keyboard.press("Enter");

  // 6 — the question form appears with factoid guidance; the confirmed
  // crop is in the viewer; demo mode keeps Ask disabled, visibly.
  const question = page.getByRole("textbox", { name: "Your question" });
  await expect(question).toBeVisible();
  await question.focus();
  await page.keyboard.type("What is the tallest bar?");
  await expect(page.getByText(/factoid questions best/)).toBeVisible();
  await expect(page.getByRole("button", { name: "▶ Ask IRIS" })).toBeDisabled();
  await expect(page.getByText(/Demo mode/)).toBeVisible();
  await expect(
    page.getByRole("group", { name: /Your confirmed chart/ }),
  ).toBeVisible();

  // 7 — the live region announced the confirmed chart.
  await expect(page.locator("[aria-live='polite']")).toContainText(
    /Chart confirmed/i,
  );
});
