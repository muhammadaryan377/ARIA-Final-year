// Playwright e2e: View Details chain (hyp.txt section 38).
//
// Open Decision Center
//   -> Find a valid Decision
//   -> Click View Details
//   -> URL contains actual Decision ID (never a title slug / generic page)
//   -> Detail page loads
//   -> Backend detail request succeeds
//   -> Title matches selected Decision
//   -> Evidence renders
//   -> What To Consider renders when available
// Plus: direct navigation to the detail URL + detail-failure contract
// (section 39: "Unable to load this decision." with retry + back).
//
// Mocked-contract fixtures: deterministic payloads, no live analysis data.
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';
const DECISION_ID = 'dc_chain_001';
const DECISION_TITLE = 'Chain decision — freight review';
const CONSIDERATION = 'Consider focusing attention on the weakest ship region first.';

function chainDecision() {
  return {
    id: DECISION_ID,
    title: DECISION_TITLE,
    description: 'Freight differs across ship regions in saved evidence.',
    metric: 'freight',
    driver: 'ship_region',
    table: 'orders',
    status: 'Saved',
    priority: 'Medium',
    confidence: 'Moderate',
    evidence_quality: 'Moderate',
    finding: 'Freight differs across ship regions in saved evidence.',
    what_to_consider: CONSIDERATION,
    narrative: {
      executive_summary: 'Freight differs across ship regions.',
      story: 'Freight differs across ship regions in saved evidence.',
      what_found: ['Freight differs across ship regions in saved evidence.'],
      why_matters: 'Getting freight right moves cost in the intended direction.',
      status: 'deterministic',
    },
    evidence: [{ n: 60, effect_r: 0.55, p_value: 0.004, driver: 'ship_region', metric: 'freight' }],
    possible_drivers: [],
    associated_factors: [],
    recommendation: { available: true, text: 'Focus attention on the weakest ship region first.' },
    impact: { baseline: 120, scenario: 100, delta: -20, rel_pct: -16.7 },
    forecast: null,
    evidence_viz: null,
    visual: null,
    chart_ref: null,
    scenario_state: {
      availability: 'insufficient_data',
      message: "What-if modeling isn't available for this decision yet.",
    },
    monitoring_state: {
      availability: 'not_supported',
      message: "Monitoring isn't set up for this decision yet.",
      watch: null,
      watch_suggestion: null,
    },
  };
}

async function mockSessionConnected(page) {
  await page.route('**/api/session/status', async (route) => {
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        connection_status: 'connected',
        display_state: 'ready',
        db_name: 'mockdb',
        source_type: 'relational',
      }),
    });
  });
  await page.route('**/api/insights', async (route) => {
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, success: true, insights: [] }),
    });
  });
}

async function mockDecisionCenter(page, { saved, detailStatus = 200 } = {}) {
  const decision = chainDecision();
  const list = saved === undefined ? [decision] : saved;
  await page.route('**/api/decision-center/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    if (path === '/api/decision-center/overview') {
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({
          ok: true, success: true,
          connection: { status: 'connected', db_name: 'mockdb' },
          priorities: [], aria_view: 'Mock view for view-details chain.',
          limitations: [], analyzed_at: new Date().toISOString(),
        }),
      });
    }
    if (path === '/api/decision-center/decisions' && method === 'GET') {
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ ok: true, success: true, saved: list }),
      });
    }
    if (path.startsWith('/api/decision-center/recommendations')) {
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ ok: true, success: true, prescriptions: [] }),
      });
    }
    const m = path.match(/^\/api\/decision-center\/decision[s]?\/([^/]+)(\/outcome)?$/);
    if (m) {
      const id = decodeURIComponent(m[1]);
      if (m[2] === '/outcome') {
        return route.fulfill({
          status: 200, contentType: 'application/json',
          body: JSON.stringify({
            ok: true, success: true, decision_id: id,
            monitoring_available: false, reason: 'Mock: no current measurement.',
            expected: null, actual: null, metric: decision.metric,
            note: 'Mock outcome — no fresh measurement yet.',
          }),
        });
      }
      const found = list.find((d) => d.id === id);
      if (!found) {
        return route.fulfill({
          status: detailStatus, contentType: 'application/json',
          body: JSON.stringify({
            ok: false, success: false, code: 'NOT_FOUND',
            detail: 'That decision is no longer saved.',
          }),
        });
      }
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({
          ok: true, success: true,
          decision: found,
          evidence_gap: null,
          narrative: found.narrative || null,
          visual: found.visual || null,
          visual_note: null,
          scenario_state: found.scenario_state || null,
          monitoring_state: found.monitoring_state || null,
        }),
      });
    }
    return route.continue();
  });
}

test.describe('View Details chain (sections 38-40)', () => {
  test('open queue item -> URL has real ID -> detail loads with matching fields', async ({ page }) => {
    await seedAuth(page, BASE);
    await mockSessionConnected(page);
    await mockDecisionCenter(page);
    await page.goto(`${BASE}#/decision-center`);
    await page.waitForLoadState('networkidle');

    // Find a valid Decision: its card carries a View Details button.
    const viewBtn = page.getByRole('button', { name: `View details for ${DECISION_TITLE}` });
    await expect(viewBtn).toBeVisible({ timeout: 20000 });
    await viewBtn.click();

    // URL contains the actual persisted Decision ID (never a title slug).
    await expect.poll(() => page.url(), { timeout: 10000 })
      .toContain(`#/decision-center/${DECISION_ID}`);
    expect(page.url()).not.toMatch(/freight-review|revenue-decline/i);

    // Detail page loads and the backend detail request succeeds.
    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 15000 });

    // Title matches the selected Decision.
    await expect(panel).toContainText(DECISION_TITLE);

    // Evidence renders.
    await expect(page.getByTestId('dc-detail-evidence')).toBeVisible();

    // What To Consider renders when available.
    const consider = page.getByTestId('dc-detail-recommendation');
    await expect(consider).toBeVisible();
    await expect(consider).toContainText(CONSIDERATION.slice(0, 40));

    // The button opened the SELECTED decision, not a generic page.
    await expect(page.getByTestId('dc-detail-back')).toBeVisible();
    await page.getByTestId('dc-detail-back').click();
    await expect(panel).toBeHidden({ timeout: 5000 });
  });

  test('direct navigation to the detail URL loads the decision (no blank page)', async ({ page }) => {
    await seedAuth(page, BASE);
    await mockSessionConnected(page);
    await mockDecisionCenter(page);
    await page.goto(`${BASE}#/decision-center/${DECISION_ID}`);
    await page.waitForLoadState('networkidle');

    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 20000 });
    await expect(panel).toContainText(DECISION_TITLE);
    await expect(page.getByTestId('dc-detail-evidence')).toBeVisible();
    // Queue underneath is intact (Back returns to it).
    await page.getByTestId('dc-detail-back').click();
    await expect(panel).toBeHidden({ timeout: 5000 });
    await expect(page.locator('#root')).toContainText(/Decision Center/i);
  });

  test('detail failure shows Unable to load this decision with retry + back', async ({ page }) => {
    await seedAuth(page, BASE);
    await mockSessionConnected(page);
    await mockDecisionCenter(page, { saved: [], detailStatus: 404 });
    await page.goto(`${BASE}#/decision-center/dc_missing_9`);
    await page.waitForLoadState('networkidle');

    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 20000 });
    // Exact contract copy (section 39) — never blank, never "Failed to fetch".
    await expect(panel).toContainText('Unable to load this decision.');
    const body = await panel.innerText();
    expect(body).not.toMatch(/Failed to fetch|fetch failed/i);

    const retry = page.getByRole('button', { name: /retry/i });
    await expect(retry).toBeVisible();
    await retry.click();
    // Still failing backend -> the contract message persists (no blank).
    await expect(panel).toContainText('Unable to load this decision.');

    const back = page.getByRole('button', { name: /back to decision center/i });
    await expect(back).toBeVisible();
    await back.click();
    await expect(panel).toBeHidden({ timeout: 5000 });
  });
});
