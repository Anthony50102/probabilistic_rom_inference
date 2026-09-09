// Optional integration check against an already-running local server and Chrome CDP.
import assert from 'node:assert/strict';

const base = `http://127.0.0.1:${process.env.PORT ?? 5173}`;
const cdp = `http://127.0.0.1:${process.env.CDP_PORT ?? 9224}`;
const pages = await (await fetch(`${cdp}/json/list`)).json();
const page = pages.find(item => item.type === 'page');
assert.ok(page, 'Start headless Chrome with a local remote-debugging port');
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
let sequence = 0;
const waiting = new Map(), errors = [], requests = [];
let manifestOverride = null;
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const pending = waiting.get(message.id);
    waiting.delete(message.id);
    if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
    else pending.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') {
    errors.push(message.params.exceptionDetails);
  } else if (message.method === 'Network.requestWillBeSent') {
    requests.push(message.params.request.url);
  } else if (message.method === 'Fetch.requestPaused') {
    const requestId = message.params.requestId;
    if (manifestOverride !== null) {
      void command('Fetch.fulfillRequest', { requestId, responseCode: 200,
        responseHeaders: [{ name: 'Content-Type', value: 'application/json' }],
        body: Buffer.from(JSON.stringify(manifestOverride)).toString('base64') });
    } else void command('Fetch.continueRequest', { requestId });
  }
});
function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    waiting.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const response = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function ready(expected = 'ready') {
  for (let attempt = 0; attempt < 120; attempt++) {
    const state = await evaluate("document.getElementById('status')?.dataset.state");
    if (state === expected) return;
    if (state === 'error') throw new Error(await evaluate("document.getElementById('status').textContent"));
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error('Timed out waiting for WebGL volumes');
}

try {
  await command('Runtime.enable');
  await command('Network.enable');
  await command('Page.enable');
  await command('Page.navigate', { url: base });
  await ready();
  assert.equal(await evaluate("document.querySelectorAll('canvas').length"), 4);
  console.log(await evaluate("document.getElementById('status').textContent"));
  await evaluate("document.getElementById('time').value = document.getElementById('time').max; document.getElementById('time').dispatchEvent(new Event('input'))");
  assert.match(await evaluate("document.getElementById('day').textContent"), /forecast/);
  await evaluate("document.getElementById('range').value = 'full'; document.getElementById('range').dispatchEvent(new Event('change'))");
  for (const slice of ['0', '4', '3']) {
    await evaluate(`document.getElementById('slice').value = '${slice}'; document.getElementById('slice').dispatchEvent(new Event('change'))`);
  }
  const doseCount = await evaluate("document.getElementById('dose').options.length");
  if (doseCount > 1) {
    await evaluate("document.getElementById('dose').selectedIndex = 1; document.getElementById('dose').dispatchEvent(new Event('change'))");
    await ready();
  }
  for (const selector of ['schema', 'method']) {
    if (await evaluate(`document.getElementById('${selector}').options.length > 1`)) {
      await evaluate(`document.getElementById('${selector}').selectedIndex = 1; document.getElementById('${selector}').dispatchEvent(new Event('change'))`);
      await ready();
    }
  }
  const original = await (await fetch(`${base}/data/manifest.json`)).json();
  await command('Fetch.enable', { patterns: [{ urlPattern: `${base}/data/manifest.json` }] });
  const first = original.cases[0];
  manifestOverride = { ...original, cases: [{ ...first, status: 'no_stable_solves',
    metadata: { ...first.metadata, n_stable: 0 }, volumes: { truth: first.volumes.truth } }] };
  await command('Page.navigate', { url: base });
  await ready();
  assert.match(await evaluate("document.getElementById('status').textContent"), /No stable solves/);
  assert.equal(await evaluate("document.querySelectorAll('#panels article:not([hidden])').length"), 1);
  manifestOverride = { ...original, cases: [] };
  await command('Page.navigate', { url: base });
  await ready('error');
  assert.match(await evaluate("document.getElementById('status').textContent"), /No completed results/);
  manifestOverride = { ...original, cases: [{ ...first, volumes: { ...first.volumes,
    truth: { ...first.volumes.truth, file: `chemo-niivue-truth-${'0'.repeat(64)}.nii.gz` } } }] };
  await command('Page.navigate', { url: base });
  await ready('error');
  assert.equal(await evaluate("document.querySelectorAll('#panels article:not([hidden])').length"), 0);
  manifestOverride = null;
  await command('Fetch.disable');
  await command('Page.navigate', { url: base });
  await ready();
  assert.deepEqual(errors, [], 'No uncaught browser errors');
  assert.deepEqual(requests.filter(url => !url.startsWith(base) && !/^(data:|blob:)/.test(url)), [],
    'Every page network request remains local');
  console.log(`Browser smoke passed: WebGL load, time/dose controls, color ranges, slice/3D modes, zero-stable/empty/missing-data states; ${requests.length} local requests.`);
} finally {
  socket.close();
}
