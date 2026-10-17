// CommonJS // Playwright e2e: All figma pages load without errors
// Requires user to be logged in (seed auth token in localStorage before running,
// or skip if already authenticated).
// Canonical nav contract (App.tsx VALID_PAGES): dashboard, analytics,
// insights, predictive (Forecast), decision-center, data, documents,
// copilot, history, settings. Standalone Prescription/Hypothesis pages were
// removed: prescription lives in Decision Center recommendations, hypothesis
// is an embedded investigation (Possible drivers / Investigate why).
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';

test.describe('Figma pages - all load', () => {
  // One login per file (serial worker): the auth rate limiter is 30/5min.
  let page;
  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await seedAuth(page, BASE);
  });
  test.afterAll(async () => { await page.close(); });
  for (const { name, nav, heading, selector } of [
    { name: 'Dashboard',    nav: 'Dashboard',     heading: 'Dashboard',         selector: 'text=Welcome' },
    { name: 'AIAnalytics', nav: 'AI Analytics', heading: 'AI Analytics',     selector: 'text=Ask' },
    { name: 'Insights',     nav: 'Insights',      heading: 'Insights',         selector: 'text=Insight' },
    { name: 'Predictive', nav: 'Forecast', heading: 'Forecast',         selector: 'text=Forecast' },
    { name: 'DecisionCenter', nav: 'Decision Center', heading: 'Decision Center', selector: 'text=Decision' },
    { name: 'Data',        nav: 'Data',           heading: 'Data',             selector: 'text=Connect' },
    { name: 'Documents',   nav: 'Documents',     heading: 'Documents',        selector: 'text=Documents' },
    { name: 'Copilot',     nav: 'Copilot',       heading: 'Copilot',          selector: 'textarea, input[type="text"]' },
    { name: 'History',     nav: 'History',       heading: 'History',          selector: 'text=History' },
    { name: 'Settings',    nav: 'Settings',      heading: 'Settings',         selector: '.nav-item' },
  ]) {
    test(`${name} page loads`, async () => {
      await page.waitForLoadState('networkidle');
      // Navigate to page via sidebar
      await page.getByRole('button', { name: new RegExp(nav, 'i') }).click();
      await page.waitForTimeout(500);
      // Check page heading or key element exists
      const el = page.locator(`h1, h2`).filter({ hasText: new RegExp(heading, 'i') }).first();
      await expect(el).toBeVisible({ timeout: 5000 });
      // No console errors
      const errors = [];
      page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
      await page.waitForTimeout(500);
      const realErrors = errors.filter(e => !e.includes('favicon') && !e.includes('404'));
      expect(realErrors).toHaveLength(0);
    });
  }

  test('legacy #/prescription aliases to Decision Center', async () => {
    await page.goto(`${BASE}#/prescription`);
    await page.waitForTimeout(800);
    const el = page.locator('h1, h2').filter({ hasText: /Decision Center/i }).first();
    await expect(el).toBeVisible({ timeout: 5000 });
  });

  test('legacy #/hypothesis aliases to AI Analytics', async () => {
    await page.goto(`${BASE}#/hypothesis`);
    await page.waitForTimeout(800);
    const el = page.locator('h1, h2').filter({ hasText: /AI Analytics|Analytics/i }).first();
    await expect(el).toBeVisible({ timeout: 5000 });
  });
});
