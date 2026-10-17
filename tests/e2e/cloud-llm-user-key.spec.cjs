"""Playwright end-to-end: user selects Cloud LLM → applies Groq API key in Settings → connects DB → asks question.

Requires: .env GROQ_API_KEY set (system key) OR user pastes their own in Settings.
Run: npx playwright test tests/e2e/cloud-llm-user-key.spec.js
"""
const { test, expect } = require('@playwright/test');

// LLM-E2E gate: needs a real cloud key AND a live Postgres. Without both,
// skip explicitly — never commit creds, never weaken auth, never fake it.
// Canonical host is the backend-served app on :8000 (vite dev port is
// dev-only; stale non-canonical port references were removed).
const HAS_KEY = Boolean(process.env.GROQ_API_KEY || process.env.ARIA_LLM_E2E_KEY);
const HAS_DB = Boolean(process.env.POSTGRES_TEST_URL);
const RUN_LLM_E2E = HAS_KEY && HAS_DB;
const SKIP_REASON = 'NOT EXECUTED — TEST CREDENTIALS UNAVAILABLE';

test.describe('Cloud LLM + user API key (UX preferred flow)', () => {
  test.skip(!RUN_LLM_E2E, SKIP_REASON);
  test('select cloud, apply user key, connect DB, ask', async ({ page }) => {
    // 1) Go to app
    await page.goto('http://localhost:8000');
    await expect(page).toHaveTitle(/ARIA/i);

    // 2) Select cloud LLM (LLMSelection page)
    await page.getByRole('button', { name: /Cloud AI/i }).click();
    await page.getByRole('button', { name: /Continue/i }).click();

    // 3) Skip DB popup initially (we'll connect via Settings / Data page later)
    await page.waitForSelector('.premium-btn');
    // Click Settings from sidebar
    await page.getByRole('button', { name: /Settings/i }).click();

    // 4) In Settings → AI Configuration, paste user's Groq key
    // Real key comes from the environment (never committed, never faked).
    await page.getByPlaceholder('Paste your API key...').fill(process.env.GROQ_API_KEY || process.env.ARIA_LLM_E2E_KEY || '');
    await page.getByRole('button', { name: /Save/i }).click();
    await expect(page.locator('text=Provider saved')).toBeVisible();

    // 5) Navigate to Data page, connect DB (uses saved user key for cloud provider)
    await page.getByRole('button', { name: /Data/i }).click();
    // Fill DB form from POSTGRES_TEST_URL (no committed values, no fake creds).
    const u = new URL(process.env.POSTGRES_TEST_URL);
    await page.getByPlaceholder('Host').fill(u.hostname || 'localhost');
    await page.getByPlaceholder('Port').fill(u.port || '5432');
    await page.getByPlaceholder('Database').fill(u.pathname.replace(/^\//, '') || 'postgres');
    await page.getByPlaceholder('Username').fill(decodeURIComponent(u.username) || 'postgres');
    await page.getByPlaceholder('Password').fill(decodeURIComponent(u.password) || '');
    await page.getByRole('button', { name: /Connect/i }).click();

    // 6) Wait for connected state then ask a business question
    await expect(page.locator('text=connected')).toBeVisible({ timeout: 10000 });
    await page.getByRole('button', { name: /Dashboard/i }).click();
    await page.getByPlaceholder('Ask a business question...').fill('What is total revenue?');
    await page.getByRole('button', { name: /Analyze/i }).click();

    // 7) Verify cloud analysis completes without falling back to mock/template
    await expect(page.locator('.stagger-children, .premium-card')).toBeVisible({ timeout: 30000 });
  });
});
