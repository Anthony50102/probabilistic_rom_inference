// Render every panel of public/data/figure/manifest.json with NiiVue in headless Chrome (software WebGL2)
// and write public/data/figure/panels/<id>.png plus the colormap tables used by the figure's colour bar.
// Usage (after `npm run build`): npm run render-figure   [CHROME=/path/to/chrome PORT=5180 CDP_PORT=9230]
// RENDER_OVERRIDES='{"azimuth":120}' previews other camera/colour settings; the settings used are saved
// with the panels (panels/render.json) and read by the figure composer.
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createLocalServer } from '../server.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const figure = path.join(root, 'public/data/figure');
const chrome = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = Number(process.env.PORT ?? 5180), cdpPort = Number(process.env.CDP_PORT ?? 9230);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const manifest = JSON.parse(await readFile(path.join(figure, 'manifest.json'), 'utf8'));
const overrides = JSON.parse(process.env.RENDER_OVERRIDES ?? '{}');
const settings = { ...manifest.render, ...overrides };
const { width, height } = settings;
const server = createLocalServer();
await new Promise((resolve, reject) => server.once('error', reject).listen(port, '127.0.0.1', resolve));
const profile = await mkdtemp(path.join(os.tmpdir(), 'niivue-figure-'));
const browser = spawn(chrome, ['--headless=new', `--user-data-dir=${profile}`, '--remote-debugging-address=127.0.0.1',
  `--remote-debugging-port=${cdpPort}`, '--enable-unsafe-swiftshader', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-component-update', '--disable-sync', 'about:blank'],
{ stdio: 'ignore' });
let socket;
try {
  let page;
  for (let attempt = 0; attempt < 100 && !page; attempt++) {
    try {
      page = (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(item => item.type === 'page');
    } catch { /* Chrome is still starting */ }
    if (!page) await sleep(200);
  }
  if (!page) throw new Error('Headless Chrome did not expose a page');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
  let sequence = 0;
  const waiting = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const pending = waiting.get(message.id);
    waiting.delete(message.id);
    if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
    else pending.resolve(message.result);
  });
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    waiting.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const response = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  };
  await command('Runtime.enable');
  await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 2, mobile: false });
  await command('Page.navigate', { url: `http://127.0.0.1:${port}/figure.html` });
  for (let attempt = 0; attempt < 150; attempt++) {
    const state = await evaluate("document.getElementById('status')?.dataset.state");
    if (state === 'ready') break;
    if (state === 'error') throw new Error(await evaluate("document.getElementById('status').textContent"));
    if (attempt === 149) throw new Error('Timed out waiting for the figure page');
    await sleep(200);
  }
  await mkdir(path.join(figure, 'panels'), { recursive: true });
  for (const panel of manifest.panels) {
    const info = await evaluate(`window.renderPanel(${JSON.stringify(panel.id)}, ${JSON.stringify(overrides)})`);
    await sleep(300);
    const shot = await command('Page.captureScreenshot', { format: 'png',
      clip: { x: 0, y: 0, width, height, scale: 1 } });
    await writeFile(path.join(figure, 'panels', `${panel.id}.png`), Buffer.from(shot.data, 'base64'));
    console.log(`${panel.id}: grid ${info.dims.join('x')}, colour range ${info.cal_min}-${info.cal_max}`);
  }
  const names = [...new Set(manifest.panels.map(panel => settings.colormap ?? panel.colormap))];
  const tables = await evaluate(`window.colormapTables(${JSON.stringify(names)})`);
  await writeFile(path.join(figure, 'panels', 'colormaps.json'), JSON.stringify(tables, null, 2));
  await writeFile(path.join(figure, 'panels', 'render.json'), JSON.stringify(settings, null, 2));
  console.log(`Rendered ${manifest.panels.length} panels to ${path.join(figure, 'panels')}`);
} finally {
  socket?.close();
  browser.kill();
  server.close();
  await sleep(300);
  await rm(profile, { recursive: true, force: true });
}
