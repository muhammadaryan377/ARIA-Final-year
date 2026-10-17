// Shared Playwright helper: real backend auth + LLM selection.
// Single-host policy: everything runs against http://localhost:8000.
// No fake tokens: registers a throwaway user via the real API, signs in
// through the real login form, then picks the requested AI engine.
async function seedAuth(page, base = 'http://localhost:8000', mode = 'Local AI') {
  const user = 'pw' + Date.now() + Math.floor(Math.random() * 1e6);
  const email = `${user}@aria.test`;
  const pass = 'password123';
  // NOTE: the login form derives the backend username by stripping invalid
  // chars from the email (see toBackendUsername), so register that exact name.
  const res = await page.request.post(base + '/api/auth/register', {
    data: { username: user + 'aria.test', password: pass },
  });
  if (!res.ok()) {
    throw new Error('register failed: ' + (await res.text()).slice(0, 200));
  }
  await page.goto(base);
  await page.locator('#login-email').fill(email);
  await page.locator('#login-password').fill(pass);
  await page.locator('.login-button').click();
  await page.getByText('Choose your AI engine').waitFor({ timeout: 20000 });
  await page.getByText(mode, { exact: false }).first().click();
  await page.getByRole('button', { name: /continue/i }).click();
  await page.waitForLoadState('networkidle');
}

module.exports = { seedAuth };
