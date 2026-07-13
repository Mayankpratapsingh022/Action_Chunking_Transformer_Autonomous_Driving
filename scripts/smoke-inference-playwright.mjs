import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const simUrl = process.env.SIM_URL ?? 'http://127.0.0.1:5173/';
const timeoutMs = Number(process.env.ACT_SMOKE_TIMEOUT_MS ?? 45_000);
const movementTimeoutMs = Number(process.env.ACT_MOVEMENT_TIMEOUT_MS ?? 5_000);
const requireMovement = process.env.ACT_REQUIRE_MOVEMENT === '1';

await mkdir('smoke-artifacts', { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(simUrl, { waitUntil: 'networkidle' });
  await page.waitForSelector('#viewport');
  await page.selectOption('#intent-select', 'continue_straight_intersection');

  const started = Date.now();
  await page.keyboard.press('i');
  await page.waitForFunction(
    () => document.querySelector('[data-status="status"]')?.textContent?.includes('Connected'),
    undefined,
    { timeout: 10_000 },
  );
  await page.waitForFunction(() => window.__VLA_DEBUG__?.getInferenceStats().latest !== null, undefined, {
    timeout: timeoutMs,
  });
  const firstActionMs = Date.now() - started;
  let movementError = null;
  try {
    await page.waitForFunction(
      () => Number(document.querySelector('[data-status="speed"]')?.textContent) > 0.05,
      undefined,
      { timeout: movementTimeoutMs },
    );
  } catch (error) {
    movementError = error;
  }
  await page.waitForTimeout(2_000);

  const result = await page.evaluate(() => ({
    speed: Number(document.querySelector('[data-status="speed"]')?.textContent),
    steering: Number(document.querySelector('[data-status="steering"]')?.textContent),
    progress: document.querySelector('[data-status="progress"]')?.textContent,
    status: document.querySelector('[data-status="status"]')?.textContent,
    actions: Object.fromEntries(['forward', 'backward', 'left', 'right'].map((name) => [
      name,
      Number(document.querySelector(`[data-action-value="${name}"]`)?.textContent),
    ])),
    inference: window.__VLA_DEBUG__?.getInferenceStats(),
  }));
  await page.screenshot({ path: 'smoke-artifacts/act-inference.png' });
  if (pageErrors.length > 0) throw new Error(`Page errors: ${pageErrors.join('\n')}`);
  if (movementError && requireMovement) {
    throw new Error(`ACT returned a prediction but the car did not move: ${JSON.stringify(result)}`);
  }
  console.log(JSON.stringify({ firstActionMs, moved: movementError === null, ...result }, null, 2));
} finally {
  await browser.close();
}
