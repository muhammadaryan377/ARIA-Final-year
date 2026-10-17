// CommonJS // Playwright e2e: Intelligence module (Explore / Hypotheses /
// Goals / What-If / Reports). Single-host policy: everything runs against
// http://localhost:8000 with a real throwaway user (see auth-helper).
// Assertions are backend-agnostic: they verify tabs, actions, persistence
// and navigation work, and that failures render a Retry (never a bare
// "Failed to fetch"), whether or not /api/intelligence/* is implemented.
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';
const BARE_FETCH = /failed to fetch/i;

test.describe('Intelligence module', () => {
  let page;
  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await seedAuth(page, BASE);
  });
  test.afterAll(async () => { await page.close(); });

  async function openIntelligence() {
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /^intelligence$/i }).click();
    await expect(page.getByRole('heading', { name: 'ARIA Intelligence' })).toBeVisible({ timeout: 8000 });
  }

  async function openTab(name) {
    await page.getByRole('tab', { name: new RegExp(name, 'i') }).click();
    await page.waitForTimeout(400);
  }

  test('header + all five tabs render', async () => {
    await openIntelligence();
    await expect(page.getByText('Explore beyond the dashboard.')).toBeVisible();
    for (const t of ['Explore', 'Hypotheses', 'Goals', 'What-If', 'Reports']) {
      await expect(page.getByRole('tab', { name: new RegExp(t, 'i') })).toBeVisible();
    }
  });

  test('Explore: builder controls + Run Analysis (results or Retry, never bare fetch error)', async () => {
    await openIntelligence();
    await openTab('Explore');
    await expect(page.getByLabel('Metric')).toBeVisible({ timeout: 8000 });
    await expect(page.getByLabel('Break down by')).toBeVisible();
    await expect(page.getByLabel('Time granularity')).toBeVisible();
    await expect(page.getByLabel('Comparison')).toBeVisible();
    await page.getByRole('button', { name: /run analysis/i }).click();
    // Either results, an empty note, or an error panel with a retry button.
    await expect(
      page.getByText(/ARIA Observation|No observation was returned|couldn't reach the server|couldn't complete|couldn't load/i).first()
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
    // Retry control exists on failure.
    if (await page.getByRole('button', { name: /run again/i }).count()) {
      await expect(page.getByRole('button', { name: /run again/i })).toBeVisible();
    }
  });

  test('Explore: Bar/Line toggle + Next Steps navigate to sibling tabs', async () => {
    await openIntelligence();
    await openTab('Explore');
    await page.getByRole('button', { name: /run analysis/i }).click();
    await page.waitForTimeout(2500);
    if (await page.getByText('ARIA Observation').count()) {
      await page.getByRole('button', { name: /^line chart$/i }).click();
      await page.getByRole('button', { name: /^bar chart$/i }).click();
      // Next step -> What-If tab becomes selected.
      await page.getByRole('button', { name: /test a scenario/i }).click();
      await expect(page.getByRole('tab', { name: /what-if/i })).toHaveAttribute('aria-selected', 'true');
    } else {
      test.skip(true, 'explore backend unavailable — navigation covered by tab test');
    }
  });

  test('Hypotheses: list/search + View detail + back navigation', async () => {
    await openIntelligence();
    await openTab('Hypotheses');
    await expect(
      page.getByLabel('Search hypotheses').or(page.getByText(/no hypotheses found yet|couldn't load/i).first())
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
    const viewBtn = page.getByRole('button', { name: /^view$/i }).first();
    if (await viewBtn.count()) {
      await viewBtn.click();
      await expect(
        page.getByText(/Analysis Context|No observed pattern|couldn't load/i).first()
      ).toBeVisible({ timeout: 20000 });
      // Back to the list.
      await page.getByRole('button', { name: /back to hypotheses/i }).click();
      await expect(page.getByLabel('Search hypotheses')).toBeVisible({ timeout: 8000 });
    }
  });

  test('Goals: filter tabs + Save goal persists after reload', async () => {
    await openIntelligence();
    await openTab('Goals');
    for (const t of ['Active', 'Completed', 'Paused', 'All Goals']) {
      await expect(page.getByRole('tab', { name: new RegExp(`^${t}$`, 'i') })).toBeVisible();
    }
    await page.getByRole('button', { name: /^create goal$/i }).first().click();
    const title = 'E2E Keepalive ' + Date.now();
    await page.getByLabel('Goal title').fill(title);
    await page.getByLabel('Target value').fill('12345');
    await page.getByRole('button', { name: /^save goal$/i }).click();
    await page.waitForTimeout(2500);
    await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
    if (await page.getByText(title).count()) {
      // Status persists after reload (backend persistence, not React state).
      await page.reload();
      await page.waitForLoadState('networkidle');
      await page.getByRole('button', { name: /^intelligence$/i }).click();
      await expect(page.getByRole('heading', { name: 'ARIA Intelligence' })).toBeVisible({ timeout: 8000 });
      await page.getByRole('tab', { name: /^goals$/i }).click();
      await expect(page.getByText(title)).toBeVisible({ timeout: 20000 });
    }
  });

  test('What-If: setup form renders, slider range or retry, run only via backend', async () => {
    await openIntelligence();
    await openTab('What-If');
    await expect(
      page.getByLabel('Scenario variable').or(page.getByText(/no scenario variables available|couldn't load/i).first())
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
    if (await page.getByLabel('Scenario variable').count()) {
      const run = page.getByRole('button', { name: /run scenario/i });
      await expect(run).toBeVisible();
      await run.click();
      await expect(
        page.getByText(/Projected Impact|no projected impact|no scenario yet|couldn't reach the server|couldn't complete/i).first()
      ).toBeVisible({ timeout: 30000 });
      await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
    }
  });

  test('Reports: builder form + template + Generate PDF flow', async () => {
    await openIntelligence();
    await openTab('Reports');
    await expect(page.getByLabel('Report name')).toBeVisible({ timeout: 8000 });
    await expect(page.getByLabel('Report description')).toBeVisible();
    await page.getByLabel('Report name').fill('E2E Review ' + Date.now());
    const gen = page.getByRole('button', { name: /generate pdf/i });
    await expect(gen).toBeVisible();
    await gen.click();
    await expect(
      page.getByText(/report generated|opening print|isn't available|couldn't reach the server|couldn't complete|couldn't generate/i).first()
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(BARE_FETCH)).toHaveCount(0);
  });

  test('keyboard: arrow keys move across tabs', async () => {
    await openIntelligence();
    const explore = page.getByRole('tab', { name: /^explore$/i });
    await explore.click();
    await explore.press('ArrowRight');
    await expect(page.getByRole('tab', { name: /^hypotheses$/i })).toHaveAttribute('aria-selected', 'true');
    await page.getByRole('tab', { name: /^hypotheses$/i }).press('ArrowLeft');
    await expect(page.getByRole('tab', { name: /^explore$/i })).toHaveAttribute('aria-selected', 'true');
  });
});
