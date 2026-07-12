import { chromium } from 'playwright';

const url = process.env.SIM_URL ?? 'http://localhost:5174/';
const browser = await chromium.launch({ channel: 'chrome', headless: true });

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function overlap(page, first, second) {
  return page.evaluate(([a, b]) => {
    const firstElement = document.querySelector(a);
    const secondElement = document.querySelector(b);
    if (!(firstElement instanceof HTMLElement) || !(secondElement instanceof HTMLElement)) return false;
    const firstStyle = getComputedStyle(firstElement);
    const secondStyle = getComputedStyle(secondElement);
    if (firstStyle.display === 'none' || secondStyle.display === 'none') return false;
    const x = firstElement.getBoundingClientRect();
    const y = secondElement.getBoundingClientRect();
    return x.left < y.right && x.right > y.left && x.top < y.bottom && x.bottom > y.top;
  }, [first, second]);
}

try {
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const desktopErrors = [];
  desktop.on('pageerror', (error) => desktopErrors.push(error.message));
  await desktop.goto(url, { waitUntil: 'networkidle' });
  await desktop.waitForTimeout(1800);

  const initial = await desktop.evaluate(() => window.__VLA_DEBUG__.getRenderStats());
  assert(Number(await desktop.textContent('[data-status="fps"]')) >= 55, 'desktop FPS fell below 55');
  assert(!await overlap(desktop, '.right-panel', '.action-panel'), 'telemetry overlaps action panel');
  assert(!await overlap(desktop, '.action-panel', '#model-view-label'), 'action panel overlaps model preview label');
  assert(!await overlap(desktop, '#sensor-preview', '.record-dock'), 'model preview overlaps command dock');

  const saturatedRoutePixels = await desktop.evaluate(() => {
    const canvas = document.querySelector('#sensor-preview');
    if (!(canvas instanceof HTMLCanvasElement)) return Infinity;
    const context = canvas.getContext('2d');
    if (!context) return Infinity;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index + 2] > 135 && pixels[index + 2] > pixels[index + 1] + 32 && pixels[index + 2] > pixels[index] + 45) count++;
    }
    return count;
  });
  assert(saturatedRoutePixels < 500, `presentation route leaked into model camera (${saturatedRoutePixels} pixels)`);

  for (let index = 0; index < 50; index++) {
    await desktop.fill('#seed-input', String(1000 + index));
    await desktop.dispatchEvent('#seed-input', 'change');
    await desktop.waitForTimeout(24);
  }
  await desktop.waitForTimeout(1000);
  const finalStats = await desktop.evaluate(() => window.__VLA_DEBUG__.getRenderStats());
  assert(finalStats.geometries <= initial.geometries + 140, `geometry count grew after reloads: ${initial.geometries} -> ${finalStats.geometries}`);
  assert(finalStats.textures <= initial.textures + 4, `texture count grew after reloads: ${initial.textures} -> ${finalStats.textures}`);
  assert(desktopErrors.length === 0, `desktop errors: ${desktopErrors.join('\n')}`);
  await desktop.close();

  const mobile = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    deviceScaleFactor: 2,
  });
  const mobileErrors = [];
  mobile.on('pageerror', (error) => mobileErrors.push(error.message));
  await mobile.goto(url, { waitUntil: 'networkidle' });
  await mobile.waitForTimeout(1800);
  assert(await mobile.locator('#hud-root').evaluate((element) => element.classList.contains('inspector-closed')), 'mobile inspector did not start collapsed');
  assert(Number(await mobile.textContent('[data-status="fps"]')) >= 28, 'mobile FPS fell below 28');
  assert(!await overlap(mobile, '#sensor-preview', '#touch-go'), 'model preview overlaps accelerator');
  assert(!await overlap(mobile, '#sensor-preview', '#touch-brake'), 'model preview overlaps brake');
  assert(!await overlap(mobile, '.record-dock', '#touch-go'), 'command dock overlaps accelerator');
  assert(!await overlap(mobile, '.record-dock', '#touch-left'), 'command dock overlaps steering control');
  const accelerator = await mobile.locator('#touch-go').boundingBox();
  assert(accelerator, 'mobile accelerator is not visible');
  await mobile.mouse.move(accelerator.x + accelerator.width / 2, accelerator.y + accelerator.height / 2);
  await mobile.mouse.down();
  await mobile.waitForTimeout(650);
  await mobile.mouse.up();
  await mobile.waitForTimeout(150);
  assert(Number(await mobile.textContent('[data-status="speed"]')) > 0.05, 'mobile accelerator did not move the ego vehicle');
  assert(mobileErrors.length === 0, `mobile errors: ${mobileErrors.join('\n')}`);

  const mobileStats = await mobile.evaluate(() => window.__VLA_DEBUG__.getRenderStats());
  console.log(JSON.stringify({
    desktop: { initial, after50Reloads: finalStats, modelRoutePixels: saturatedRoutePixels },
    mobile: mobileStats,
  }, null, 2));
  await mobile.close();
} finally {
  await browser.close();
}
