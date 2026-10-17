// QA: drive login -> LLM selection -> every non-Data page, capturing runtime errors.
const { test, expect } = require("@playwright/test");

const BASE = process.env.ARIA_BASE || "http://localhost:8000";

const PAGES = [
  { id: "dashboard", nav: "Dashboard" },
  { id: "analytics", nav: "AI Analytics" },
  { id: "insights", nav: "Insights" },
  { id: "predictive", nav: "Forecast" },
  { id: "decision-center", nav: "Decision Center" },
  // No standalone prescription/hypothesis pages (removed): prescription is
  // verified via Decision Center recommendations; hypothesis via Insights
  // "Investigate why" -> AI Analytics Possible drivers (see
  // qa-connected-flow.spec.cjs for the contextual coverage).
  // "data" intentionally skipped (out of scope)
  { id: "documents", nav: "Documents" },
  { id: "copilot", nav: "Copilot" },
  { id: "history", nav: "History" },
  { id: "settings", nav: "Settings" },
];

const BENIGN = [
  /favicon/i,
  /Download the React DevTools/i,
  /Session\/status probe/i,
];

function attach(page, sink) {
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (BENIGN.some((re) => re.test(text))) return;
    sink.consoleErrors.push(text);
  });
  page.on("pageerror", (err) => {
    sink.pageErrors.push(err.message);
  });
  page.on("response", (res) => {
    const url = res.url();
    const status = res.status();
    if (status >= 400 && !BENIGN.some((re) => re.test(url))) {
      sink.failedResponses.push(`${status} ${res.request().method()} ${url}`);
    }
  });
}

async function registerAndLogin(page, email, password) {
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });

  // Switch to register mode and create the account.
  await page.getByRole("button", { name: /create account/i }).first().click();
  await page.locator("#login-email").fill(email);
  await page.locator("#login-password").fill(password);
  await page.locator("#login-confirm").fill(password);
  await page.getByRole("button", { name: /create account/i }).last().click();
  await page.waitForTimeout(800);

  // Now sign in (register flips back to login mode).
  await page.locator("#login-email").fill(email);
  await page.locator("#login-password").fill(password);
  await page.getByRole("button", { name: /access intelligence hub/i }).click();
}

async function chooseLLM(page, label) {
  await expect(page.getByText(/choose your ai engine/i)).toBeVisible({ timeout: 15000 });
  await page.getByRole("button", { name: new RegExp(label, "i") }).first().click();
  await page.getByRole("button", { name: /^continue$/i }).click();
  await expect(page.getByRole("button", { name: /dashboard/i }).first()).toBeVisible({ timeout: 15000 });
}

test.describe("ARIA QA - non-Data pages", () => {
  test("login -> llm -> every page -> refresh -> back", async ({ page }) => {
    const sink = { consoleErrors: [], pageErrors: [], failedResponses: [] };
    attach(page, sink);

    const email = `qa_${Date.now()}@aria.test`;
    const password = "QaPass123!";

    await registerAndLogin(page, email, password);
    await chooseLLM(page, "Cloud AI");

    const perPage = {};

    for (const p of PAGES) {
      sink.consoleErrors.length = 0;
      sink.pageErrors.length = 0;
      sink.failedResponses.length = 0;

      await page.getByRole("button", { name: new RegExp(p.nav, "i") }).first().click();
      await page.waitForTimeout(900);

      const root = page.locator("#root");
      const html = await root.innerHTML();
      const blank = !html || html.length < 200;

      perPage[p.id] = {
        blank,
        consoleErrors: [...sink.consoleErrors],
        pageErrors: [...sink.pageErrors],
        failedResponses: [...sink.failedResponses],
      };

      expect(blank, `page ${p.id} rendered empty`).toBe(false);
    }

    // Refresh on a mid page (History) — must not blank out.
    await page.getByRole("button", { name: /history/i }).first().click();
    await page.waitForTimeout(500);
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForTimeout(1200);
    const afterReload = await page.locator("#root").innerHTML();
    expect(afterReload.length, "blank after refresh").toBeGreaterThan(200);

    // Browser back must not crash.
    await page.goBack();
    await page.waitForTimeout(600);
    await page.goForward();
    await page.waitForTimeout(600);

    // Report everything found.
    const problems = Object.entries(perPage).filter(
      ([, v]) => v.consoleErrors.length || v.pageErrors.length || v.failedResponses.length
    );
    if (problems.length) {
      console.log("QA_PAGE_PROBLEMS=" + JSON.stringify(problems, null, 2));
    }
    expect(sink.pageErrors, `uncaught page errors: ${sink.pageErrors.join(" | ")}`).toEqual([]);
  });
});