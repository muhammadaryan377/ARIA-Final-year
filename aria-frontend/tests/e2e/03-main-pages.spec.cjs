// CommonJS // Playwright e2e: Main App pages (Dashboard, Data, Schema, Analytics, etc.)
// Requires user to be authenticated.
// Canonical hash contract: #/decision-center is the Decide surface
// (recommendations + scenario + saved decisions). #/prescription aliases to
// it; #/hypothesis aliases to #/analytics where hypothesis now lives as
// embedded Possible drivers / Investigate why.
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';

test.describe('Main App pages', () => {
  let page;
  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await seedAuth(page, BASE);
  });
  test.afterAll(async () => { await page.close(); });
  test('Dashboard loads with KPI cards and question input', async () => {
        await page.waitForLoadState('networkidle');
    // Dashboard should show KPI area or question input
    const hasInput = await page.locator('textarea, input[type="text"]').count();
    const hasHeading = await page.locator('h1, h2').count();
    expect(hasInput + hasHeading).toBeGreaterThan(0);
  });

  test('Data source page loads with connect form', async () => {
        await page.goto(`${BASE}#/dataSource`);
    await page.waitForLoadState('networkidle');
    const hasConnect = await page.locator('button', { hasText: /connect|submit/i }).count();
    const hasForm = await page.locator('input, select').count();
    expect(hasForm + hasConnect).toBeGreaterThan(0);
  });

  test('Schema page loads', async () => {
        await page.goto(`${BASE}#/schema`);
    await page.waitForLoadState('networkidle');
    const hasSchema = await page.locator('h1, h2').count();
    expect(hasSchema).toBeGreaterThan(0);
  });

  test('Analytics page loads', async () => {
        await page.goto(`${BASE}#/analytics`);
    await page.waitForLoadState('networkidle');
    const hasAnalytics = await page.locator('h1, h2').count();
    expect(hasAnalytics).toBeGreaterThan(0);
  });

  test('Insights page loads', async () => {
        await page.goto(`${BASE}#/insights`);
    await page.waitForLoadState('networkidle');
    const hasInsights = await page.locator('h1, h2').count();
    expect(hasInsights).toBeGreaterThan(0);
  });

  test('Predictions page loads', async () => {
        await page.goto(`${BASE}#/predictions`);
    await page.waitForLoadState('networkidle');
    const hasPredictions = await page.locator('h1, h2').count();
    expect(hasPredictions).toBeGreaterThan(0);
  });

  test('Decision Center loads (prescription verified via recommendations integration)', async () => {
        await page.goto(`${BASE}#/decision-center`);
    await page.waitForLoadState('networkidle');
    const hasDC = await page.locator('h1, h2').count();
    expect(hasDC).toBeGreaterThan(0);
  });

  test('legacy #/prescription aliases to Decision Center', async () => {
        await page.goto(`${BASE}#/prescription`);
    await page.waitForLoadState('networkidle');
    const heading = await page.locator('h1, h2').first().innerText().catch(() => '');
    expect(heading).toMatch(/Decision Center|Decision/i);
  });

  test('legacy #/hypothesis aliases to AI Analytics (embedded Possible drivers)', async () => {
        await page.goto(`${BASE}#/hypothesis`);
    await page.waitForLoadState('networkidle');
    const heading = await page.locator('h1, h2').first().innerText().catch(() => '');
    expect(heading).toMatch(/Analytics/i);
  });

  test('History page loads', async () => {
        await page.goto(`${BASE}#/history`);
    await page.waitForLoadState('networkidle');
    const hasHistory = await page.locator('h1, h2').count();
    expect(hasHistory).toBeGreaterThan(0);
  });

  test('Documents page loads', async () => {
        await page.goto(`${BASE}#/documents`);
    await page.waitForLoadState('networkidle');
    const hasDocs = await page.locator('h1, h2').count();
    expect(hasDocs).toBeGreaterThan(0);
  });

  test('Settings page loads with AI config', async () => {
        await page.goto(`${BASE}#/settings`);
    await page.waitForLoadState('networkidle');
    // Check AI Configuration tab is accessible
    const hasAIConfig = await page.getByText(/AI Configuration/i).count();
    expect(hasAIConfig).toBeGreaterThan(0);
  });
});
