// CommonJS // Playwright e2e: opened-decision workspace flow.
// Covers the new detail experience: open → story → chart → evidence →
// why → recommendation → back. Contract-based (no invented data):
// - with queued decisions: walks the full workspace and returns cleanly.
// - without data / disconnected: asserts the honest empty/connect states.
//
// Conditional-section rework (no filler, no duplicates, no null-tables):
// - weak decision renders story + evidence but NO drivers/rec/scenario/watch blocks
// - evidence never renders the Metric/Records null-table or raw internals
// - a recommendation duplicating the story renders exactly once
// - a finding-less decision renders ONE page-level insufficient state only
const { test, expect } = require('@playwright/test');
const { seedAuth } = require('./auth-helper');

const BASE = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Mocked-contract fixtures: deterministic detail payloads exercising the
// conditional-section contract without depending on live analysis data.
// ---------------------------------------------------------------------------

const WEAK_STORY = 'Total Amount shows a pattern worth deciding on across recent orders.';

function weakDecision() {
  return {
    id: 'dc_weak_001',
    title: 'Weak decision — needs validation',
    description: 'Weak handoff for conditional-section test.',
    metric: 'total_amt',
    driver: 'region',
    table: 'orders',
    status: 'Saved',
    priority: 'Low',
    confidence: 'Limited',
    evidence_quality: 'Limited',
    finding: WEAK_STORY,
    narrative: {
      executive_summary: 'Total Amount shows a pattern.',
      story: WEAK_STORY,
      what_found: [WEAK_STORY],
      why_matters: null,
      status: 'deterministic',
    },
    evidence: [{ n: 12, effect_r: 0.15, p_value: 0.2, driver: 'region', metric: 'total_amt' }],
    possible_drivers: [],
    associated_factors: [],
    recommendation: '',
    impact: {},
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

const DUP_STORY = 'Revenue is estimated to move from 100 to 120. Getting this right moves revenue forward.';
const DUP_REC = 'Revenue is estimated to move from 100 to 120.';

function duplicateDecision() {
  return {
    id: 'dc_dup_001',
    title: 'Duplicate recommendation decision',
    description: 'Recommendation duplicates the story sentence.',
    metric: 'revenue',
    driver: 'ad_spend',
    table: 'sales',
    status: 'Saved',
    priority: 'Medium',
    confidence: 'Moderate',
    evidence_quality: 'Moderate',
    finding: DUP_STORY,
    narrative: {
      executive_summary: 'Revenue linked to ad spend — worth validating.',
      story: DUP_STORY,
      what_found: [DUP_STORY],
      why_matters: 'Getting this right moves revenue in the intended direction.',
      status: 'deterministic',
    },
    evidence: [{ n: 50, effect_r: 0.6, p_value: 0.01, driver: 'ad_spend', metric: 'revenue' }],
    possible_drivers: [{ label: 'Ad Spend', driver: 'ad_spend', association: 'association +0.60 over 50 observations' }],
    associated_factors: [{ label: 'Ad Spend', driver: 'ad_spend', association: 'association +0.60 over 50 observations' }],
    recommendation: { available: true, text: DUP_REC },
    impact: { baseline: 100, scenario: 120, delta: 20, rel_pct: 20 },
    forecast: { summary: 'Forecast expects continued movement.' },
    evidence_viz: null,
    visual: {
      type: 'bar', x: 'name', y: 'value', series: [],
      title: 'Observed vs modeled',
      data: [{ name: 'Observed level', value: 100 }, { name: 'Modeled level', value: 120 }],
    },
    scenario_state: { availability: 'available', message: 'What-if modeling is available.' },
    monitoring_state: {
      availability: 'available',
      message: 'Watching against the saved reference.',
      watch: { metric: 'revenue', current: null, reference: 100, review_period: '2 weeks', status: 'watching' },
      watch_suggestion: null,
    },
  };
}

function emptyDecision() {
  return {
    id: 'dc_empty_001',
    title: 'Empty decision — no finding yet',
    description: '',
    metric: null,
    driver: null,
    table: null,
    status: 'Saved',
    priority: 'Low',
    confidence: 'Limited',
    evidence_quality: 'Insufficient',
    finding: null,
    narrative: null,
    evidence: [],
    possible_drivers: [],
    associated_factors: [],
    recommendation: '',
    impact: {},
    forecast: null,
    evidence_viz: null,
    visual: null,
    scenario_state: { availability: 'insufficient_data', message: "What-if modeling isn't available for this decision yet." },
    monitoring_state: { availability: 'not_supported', message: "Monitoring isn't set up for this decision yet.", watch: null, watch_suggestion: null },
  };
}

async function mockDecisionCenter(page, decision) {
  await page.route('**/api/decision-center/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    if (path === '/api/decision-center/overview') {
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({
          ok: true, success: true,
          connection: { status: 'connected', db_name: 'mock' },
          priorities: [], aria_view: 'Mock view for conditional tests.',
          limitations: [], analyzed_at: new Date().toISOString(),
        }),
      });
    }
    if (path === '/api/decision-center/decisions' && method === 'GET') {
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ ok: true, success: true, saved: [decision] }),
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
            expected: null, actual: null, metric: decision.metric || null,
            note: 'Mock outcome — no fresh measurement yet.',
          }),
        });
      }
      return route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({
          ok: true, success: true,
          decision,
          evidence_gap: null,
          narrative: decision.narrative || null,
          visual: decision.visual || null,
          visual_note: null,
          scenario_state: decision.scenario_state || null,
          monitoring_state: decision.monitoring_state || null,
        }),
      });
    }
    return route.continue();
  });
}

async function openMockedDecision(page, decision) {
  await mockDecisionCenter(page, decision);
  await page.goto(`${BASE}#/decision-center`);
  await page.waitForLoadState('networkidle');
  const openBtns = page.getByRole('button', { name: /^Open\b/ });
  await expect(openBtns.first()).toBeVisible({ timeout: 15000 });
  await openBtns.first().click();
  const panel = page.getByTestId('dc-detail-panel');
  await expect(panel).toBeVisible({ timeout: 15000 });
  return panel;
}

test.describe('Decision detail workspace', () => {
  let page;
  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await seedAuth(page, BASE);
  });
  test.afterAll(async () => { await page.close(); });

  test.beforeEach(async () => {
    await page.goto(`${BASE}#/decision-center`);
    await page.waitForLoadState('networkidle');
  });

  test('queue to workspace: open → story → chart → evidence → why → recommendation → back', async () => {
    const openBtns = page.getByRole('button', { name: /^Open\b/ });
    const openCount = await openBtns.count();
    if (openCount === 0) {
      // No stored decisions: the queue must say so honestly (never blank).
      const body = await page.locator('#root').innerHTML();
      expect(body).toMatch(/No decisions yet|No strong recommendations|Connect a data source|Analyzing|Insufficient evidence|Decision Center unavailable/i);
      return;
    }

    // OPEN: first queue card opens the workspace dialog.
    await openBtns.first().click();
    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 15000 });

    // STORY: the narrative section is the first thing rendered.
    await expect(page.getByTestId('dc-detail-story')).toBeVisible();

    // CHART: required whenever evidence supports it — assert the contract,
    // not data (absent chart = insufficient evidence, which is valid).
    const chartCount = await page.getByTestId('dc-detail-chart').count();
    if (chartCount > 0) {
      await expect(page.getByTestId('dc-detail-chart')).toContainText(/Observed|vs|values/i);
    }

    // EVIDENCE: renders when a finding exists (full or thin state).
    const evidence = page.getByTestId('dc-detail-evidence');
    await expect(evidence).toBeVisible();
    const evidenceText = await evidence.innerText();
    expect(evidenceText.length).toBeGreaterThan(0);

    // WHY: renders only with real content — never codes/raw JSON.
    const whyCount = await page.getByTestId('dc-detail-why').count();
    if (whyCount > 0) {
      const whyText = await page.getByTestId('dc-detail-why').innerText();
      expect(whyText).not.toMatch(/SCENARIO_UNSUPPORTED|missing baseline|missing table|missing column/i);
    }

    // RECOMMENDATION: real phrasing when available; otherwise the single
    // honest line lives in the footer (no per-section filler repeats).
    const recCount = await page.getByTestId('dc-detail-recommendation').count();
    if (recCount > 0) {
      const recText = await page.getByTestId('dc-detail-recommendation').innerText();
      expect(recText).toMatch(/ARIA|evidence|recommend|consider|estimate|should|review/i);
    } else {
      await expect(page.getByTestId('dc-detail-next')).toContainText(/No recommendation yet|Track this decision|Not enough evidence/i);
    }

    // WHAT TO DO NEXT: bottom bar with investigate + ask actions.
    const next = page.getByTestId('dc-detail-next');
    await expect(next).toBeVisible();
    await expect(next).toContainText(/What to do next/i);

    // BACK: keyboard Escape or the Back button returns to the queue cleanly.
    await page.getByTestId('dc-detail-back').click();
    await expect(panel).toBeHidden({ timeout: 5000 });
    await expect(page.locator('#root')).toContainText(/Decision Center/i);
  });

  test('workspace closes with Escape and preserves the queue', async () => {
    const openBtns = page.getByRole('button', { name: /^Open\b/ });
    if ((await openBtns.count()) === 0) {
      const body = await page.locator('#root').innerHTML();
      expect(body).toMatch(/No decisions yet|No strong recommendations|Connect a data source|Analyzing|Insufficient evidence|Decision Center unavailable/i);
      return;
    }
    await openBtns.first().click();
    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 15000 });
    await page.keyboard.press('Escape');
    await expect(panel).toBeHidden({ timeout: 5000 });
    // Queue state intact: filter group + cards/empty state still present.
    const body = await page.locator('#root').innerHTML();
    expect(body).toMatch(/Decision Center/i);
  });

  test('workspace never shows technical leakage', async () => {
    const openBtns = page.getByRole('button', { name: /^Open\b/ });
    if ((await openBtns.count()) === 0) return;
    await openBtns.first().click();
    const panel = page.getByTestId('dc-detail-panel');
    await expect(panel).toBeVisible({ timeout: 15000 });
    const text = await panel.innerText();
    expect(text).not.toMatch(/SELECT\s.+\sFROM|traceback|Traceback|SCENARIO_UNSUPPORTED|NOT_A_DECISION|missing baseline|missing table|missing column/i);
    await page.getByTestId('dc-detail-back').click();
    await expect(panel).toBeHidden({ timeout: 5000 });
  });
});

test.describe('Decision detail conditional sections (mocked contract)', () => {
  test('weak decision renders story + evidence but NO drivers/rec/scenario/watch blocks', async ({ page }) => {
    await seedAuth(page, BASE);
    const panel = await openMockedDecision(page, weakDecision());
    await expect(page.getByTestId('dc-detail-story')).toBeVisible();
    await expect(page.getByTestId('dc-detail-story')).toContainText(WEAK_STORY.slice(0, 40));
    await expect(page.getByTestId('dc-detail-evidence')).toBeVisible();
    await expect(page.getByTestId('dc-detail-drivers')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-recommendation')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-scenario')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-watch')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-chart')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-forecast')).toHaveCount(0);
    // Capability notes appear exactly once each (compact, no section fillers).
    await expect(page.getByTestId('dc-detail-scenario-note')).toHaveCount(1);
    await expect(page.getByTestId('dc-detail-watch-note')).toHaveCount(1);
    await expect(panel).toContainText(/Decision Center|What ARIA found/i);
  });

  test('evidence never renders the null-table or raw internals', async ({ page }) => {
    await seedAuth(page, BASE);
    await openMockedDecision(page, weakDecision());
    const evidence = page.getByTestId('dc-detail-evidence');
    await expect(evidence).toBeVisible();
    const text = await evidence.innerText();
    expect(text).not.toMatch(/Insufficient evidence/i);
    expect(text).not.toMatch(/total_amt/);
    expect(text).not.toMatch(/finding_id|evidence_id|baseline_required/);
    expect(text).not.toMatch(/^Metric$/m);
    // Humanized primary label + present-only prose survive the redesign.
    const panelText = await page.getByTestId('dc-detail-panel').innerText();
    expect(panelText).toMatch(/Total Amount/);
    expect(panelText).not.toMatch(/total_amt/);
  });

  test('recommendation duplicating the story renders exactly once (no duplicate)', async ({ page }) => {
    await seedAuth(page, BASE);
    const panel = await openMockedDecision(page, duplicateDecision());
    await expect(page.getByTestId('dc-detail-story')).toBeVisible();
    // Duplicated rec section is omitted — the sentence lives in the story only.
    await expect(page.getByTestId('dc-detail-recommendation')).toHaveCount(0);
    const text = await panel.innerText();
    const occurrences = text.split(DUP_REC.slice(0, 30)).length - 1;
    expect(occurrences).toBe(1);
    // Single Watch section only — backend emits XOR watch/watch_suggestion.
    await expect(page.getByTestId('dc-detail-watch')).toHaveCount(1);
    expect(text).not.toMatch(/Outcome to watch/i);
  });

  test('finding-less decision renders ONE page-level insufficient state only', async ({ page }) => {
    await seedAuth(page, BASE);
    const panel = await openMockedDecision(page, emptyDecision());
    await expect(page.getByTestId('dc-detail-insufficient')).toBeVisible();
    await expect(page.getByTestId('dc-detail-insufficient')).toHaveCount(1);
    await expect(page.getByTestId('dc-detail-story')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-chart')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-evidence')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-drivers')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-recommendation')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-scenario')).toHaveCount(0);
    await expect(page.getByTestId('dc-detail-watch')).toHaveCount(0);
    const text = await panel.innerText();
    expect(text).toMatch(/Not enough evidence yet|Insufficient evidence/i);
  });
});
