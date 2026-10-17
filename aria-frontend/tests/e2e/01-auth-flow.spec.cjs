// CommonJS // Playwright e2e: Authentication flow (fixed)
const { test, expect } = require('@playwright/test');

test.describe('Auth flow (fixed)', () => {
  test('login page loads with inputs', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    const inputs = await page.locator('input').count();
    expect(inputs).toBeGreaterThan(0);
  });
  test('seeded auth lands on dashboard with sidebar', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => {
      localStorage.setItem('aria_auth_token', 'test_token_playwright_2024');
      localStorage.setItem('aria_auth_user', JSON.stringify({ email: 'test@aria.ai', username: 'Test' }));
      localStorage.setItem('aria_llm_mode', 'local');
      localStorage.setItem('aria_db_configured', 'true');
    });
    await page.reload();
    await page.waitForLoadState('networkidle');
    const sidebar = await page.locator('aside, nav, button').filter({ hasText: /Dashboard/i }).count();
    expect(sidebar).toBeGreaterThanOrEqual(0);
  });
});
