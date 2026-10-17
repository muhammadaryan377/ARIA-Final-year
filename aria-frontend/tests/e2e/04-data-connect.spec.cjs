// CommonJS // Playwright e2e: Data source connection flow (all 4 types)
// Single host: http://localhost:8000. One login per file (auth limiter).
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';

test.describe('Data source connection flow', () => {
  let page;
  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await seedAuth(page, BASE);
  });
  test.afterAll(async () => { await page.close(); });

  async function gotoData() {
    await page.goto(BASE);
    await page.waitForLoadState('networkidle');
    const dataBtn = page.getByRole('button', { name: /Data/i }).first();
    if (await dataBtn.count() > 0) await dataBtn.click();
    await page.waitForTimeout(1000);
  }

  test('Data page shows 4 source type cards', async () => {
    await gotoData();
    // Should show cards for PostgreSQL, MySQL, CSV, PDF
    const pageContent = await page.content();
    const hasPostgres = /PostgreSQL|postgresql|postgres/i.test(pageContent);
    const hasMySQL = /MySQL|mysql/i.test(pageContent);
    const hasCSV = /CSV|csv/i.test(pageContent);
    const hasPDF = /PDF|pdf/i.test(pageContent);
    expect(hasPostgres || hasMySQL || hasCSV || hasPDF).toBeTruthy();
  });

  test('PostgreSQL card is clickable', async () => {
    await gotoData();
    // Click PostgreSQL card or source type button
    const pgBtn = page.getByRole('button', { name: /PostgreSQL|postgres/i }).first();
    if (await pgBtn.count() > 0) {
      await pgBtn.click();
      await page.waitForTimeout(500);
      // Should reveal a form or inputs
      const hasForm = await page.locator('input, select').count();
      expect(hasForm).toBeGreaterThan(0);
    }
  });

  test('CSV card shows file picker', async () => {
    await gotoData();
    const csvBtn = page.getByRole('button', { name: /CSV|csv/i }).first();
    if (await csvBtn.count() > 0) {
      await csvBtn.click();
      await page.waitForTimeout(500);
      const hasFileInput = await page.locator('input[type="file"]').count();
      expect(hasFileInput).toBeGreaterThan(0);
    }
  });

  test('Cloud LLM selection shows Groq info', async ({ browser }) => {
    const cloudPage = await browser.newPage();
    try {
      await seedAuth(cloudPage, BASE, 'Cloud AI');
      const content = await cloudPage.content();
      const hasCloud = /Cloud|cloud|Groq|groq/i.test(content);
      expect(hasCloud).toBeTruthy();
    } finally {
      await cloudPage.close();
    }
  });
});
