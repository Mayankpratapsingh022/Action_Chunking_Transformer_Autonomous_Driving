import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const url = process.env.SIM_URL ?? 'http://localhost:5174/';
const viewports = [
  { name: 'desktop', width: 1440, height: 900, isMobile: false },
  { name: 'mobile', width: 390, height: 844, isMobile: true },
];

await mkdir('smoke-artifacts', { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: true });

try {
  for (const viewport of viewports) {
    const page = await browser.newPage({
      viewport: { width: viewport.width, height: viewport.height },
      isMobile: viewport.isMobile,
      deviceScaleFactor: viewport.isMobile ? 2 : 1,
    });
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForSelector('#viewport');
    await page.waitForTimeout(2500);
    const initialSpeed = Number(await page.textContent('[data-status="speed"]'));
    if (viewport.name === 'desktop' && initialSpeed > 0.05) {
      throw new Error(`ego vehicle moved before user input: speed=${initialSpeed}`);
    }
    await page.screenshot({ path: `smoke-artifacts/${viewport.name}.png` });

    const canvasShot = await page.locator('#viewport').screenshot();
    const stats = { screenshotBytes: canvasShot.byteLength };
    if (canvasShot.byteLength < 12_000) {
      throw new Error(`${viewport.name} canvas appears blank: ${JSON.stringify(stats)}`);
    }
    if (viewport.name === 'desktop') {
      await page.selectOption('#intent-select', 'detour_blocked_lane');
      await page.waitForFunction(() => document.querySelector('#scenario-select')?.value === 'blocked_lane_detour');
      const scenarioAfterIntent = await page.inputValue('#scenario-select');
      const instructionAfterIntent = await page.textContent('[data-status="intent"]');
      if (scenarioAfterIntent !== 'blocked_lane_detour' || !instructionAfterIntent.includes('Detour')) {
        throw new Error(`language intent did not update scenario/instruction: ${scenarioAfterIntent} ${instructionAfterIntent}`);
      }
      await page.selectOption('#intent-select', 'obey_traffic_lights');
      await page.waitForFunction(() => document.querySelector('#scenario-select')?.value === 'traffic_light_stop_go');
      await page.waitForTimeout(700);
      const trafficScenario = await page.inputValue('#scenario-select');
      const trafficInstruction = await page.textContent('[data-status="intent"]');
      if (trafficScenario !== 'traffic_light_stop_go' || !trafficInstruction.includes('traffic lights')) {
        throw new Error(`traffic light intent did not update scenario/instruction: ${trafficScenario} ${trafficInstruction}`);
      }
      const signalPixels = await page.evaluate(() => {
        const canvas = document.querySelector('#sensor-preview');
        if (!(canvas instanceof HTMLCanvasElement)) throw new Error('Sensor preview canvas not found');
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Sensor preview 2D context unavailable');
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        const minX = Math.floor(canvas.width * 0.38);
        const maxX = Math.ceil(canvas.width * 0.62);
        const minY = Math.floor(canvas.height * 0.08);
        const maxY = Math.ceil(canvas.height * 0.52);
        for (let y = minY; y < maxY; y++) {
          for (let x = minX; x < maxX; x++) {
            const i = (y * canvas.width + x) * 4;
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const green = g > 120 && g > r + 18 && g > b + 10;
            const red = r > 145 && r > g + 25 && r > b + 20;
            const yellow = r > 145 && g > 110 && b < 115;
            if (green || red || yellow) count++;
          }
        }
        return count;
      });
      if (signalPixels < 4) {
        throw new Error(`traffic signal is not visible in model-view canvas: ${signalPixels} signal pixels`);
      }
      await page.screenshot({ path: 'smoke-artifacts/traffic-light.png' });
      await page.selectOption('#intent-select', 'follow_curved_loop');
      await page.waitForFunction(() => document.querySelector('#scenario-select')?.value === 'curved_loop_drive');
      await page.waitForTimeout(2500);
      const loopScenario = await page.inputValue('#scenario-select');
      const loopInstruction = await page.textContent('[data-status="intent"]');
      if (loopScenario !== 'curved_loop_drive' || !loopInstruction.includes('curved loop')) {
        throw new Error(`curved loop intent did not update scenario/instruction: ${loopScenario} ${loopInstruction}`);
      }
      await page.screenshot({ path: 'smoke-artifacts/curved-loop.png' });
      await page.selectOption('#intent-select', 'turn_left_intersection');
      await page.selectOption('#resolution-select', '256');
      const resolutionValue = await page.inputValue('#resolution-select');
      if (resolutionValue !== '256') {
        throw new Error(`capture resolution did not update: ${resolutionValue}`);
      }
      await page.click('[data-mode="inference"]');
      await page.click('[data-mode="training"]');
      await page.click('#record-button');
      await page.waitForTimeout(700);
      const pausedSamples = Number(await page.textContent('[data-status="samples"]'));
      if (pausedSamples !== 0) {
        throw new Error(`recording captured idle samples before input: ${pausedSamples}`);
      }
      await page.keyboard.down('w');
      await page.waitForTimeout(400);
      const liveForward = Number(await page.textContent('[data-action-value="forward"]'));
      if (!Number.isFinite(liveForward) || liveForward <= 0.1) {
        throw new Error(`live action plot did not show forward action: ${liveForward}`);
      }
      await page.waitForTimeout(400);
      await page.keyboard.up('w');
      const drivenSpeed = Number(await page.textContent('[data-status="speed"]'));
      const samples = Number(await page.textContent('[data-status="samples"]'));
      const videoLabel = await page.textContent('#video-button');
      if (!Number.isFinite(drivenSpeed) || drivenSpeed <= 0.05) {
        throw new Error(`ego vehicle did not start after key input: speed=${drivenSpeed}`);
      }
      if (!Number.isFinite(samples) || samples < 1) {
        throw new Error(`recording did not capture driving samples: ${samples}`);
      }
      if (!videoLabel.includes('Recording')) {
        throw new Error(`episode video did not start after key input: ${videoLabel}`);
      }
    }
    if (pageErrors.length > 0) {
      throw new Error(`${viewport.name} page errors: ${pageErrors.join('\n')}`);
    }
    console.log(`${viewport.name}: canvas OK`, stats);
    await page.close();
  }
} finally {
  await browser.close();
}
