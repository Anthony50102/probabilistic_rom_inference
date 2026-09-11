import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { frameLabel, localVolumeURL, methodLabel, validateManifest } from '../src/contract.js';
import { createLocalServer } from '../server.js';

const volume = { file: `chemo-niivue-truth-${'a'.repeat(64)}.nii.gz`, min: 0, max: 1 };
const item = { id: 'test', schema: 'dense_low_noise', method: '04_unified_chemo', dose_scale: 1,
  times_days: [5, 70, 110], training_end: 70, status: 'no_stable_solves', volumes: { truth: volume } };

test('input-aware results are explicitly labeled experimental', () => {
  assert.equal(methodLabel('04_unified_chemo', [item]), 'Bayesian OpInf');
  assert.equal(methodLabel('04_unified_chemo', [
    { ...item, metadata: { inference_profile: 'input-aware' } },
  ]), 'Bayesian OpInf (experimental input-aware)');
  assert.equal(methodLabel('05_neural_ode_chemo', [item]), 'Neural ODE');
});

test('manifest supports truth-only zero-stable cases and nonuniform actual days', () => {
  assert.equal(validateManifest({ format_version: 1, cases: [item] }).cases.length, 1);
  assert.equal(frameLabel(item, 1), 'day 70.000 · fit interval');
  assert.equal(frameLabel(item, 2), 'day 110.000 · forecast');
  assert.deepEqual(validateManifest({ format_version: 1, cases: [] }).cases, []);
});

test('reject external URLs, traversal, incomplete predictions and fabricated zero-stable fields', () => {
  for (const file of ['https://example.org/a.nii.gz', '../a.nii.gz', '//example.org/a', 'a.nii.gz']) {
    assert.throws(() => localVolumeURL(file));
  }
  assert.throws(() => validateManifest({ format_version: 1, cases: [{ ...item, status: 'ready' }] }));
  assert.throws(() => validateManifest({ format_version: 1, cases: [
    { ...item, volumes: { truth: volume, prediction: volume } },
  ] }));
  assert.throws(() => validateManifest({ format_version: 1, cases: [{ ...item, times_days: [5, 5] }] }));
});

test('read-only server is loopback bound, same-origin, and confines files to public roots', async t => {
  const server = createLocalServer({ dist: new URL('..', import.meta.url).pathname });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const url = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(url);
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-security-policy'), /connect-src 'self'/);
  assert.equal(server.address().address, '127.0.0.1');
  assert.equal((await fetch(`${url}/.gitignore`)).status, 403);
  assert.equal((await fetch(`${url}/%2e%2e%2fsecret`)).status, 403);
  assert.equal((await fetch(url, { method: 'POST' })).status, 405);
  assert.equal((await fetch(url, { headers: { Origin: 'https://example.org' } })).status, 403);
  const invalidHostStatus = await new Promise((resolve, reject) => {
    http.get(url, { headers: { Host: 'evil.example' } }, response => {
      response.resume();
      resolve(response.statusCode);
    }).on('error', reject);
  });
  assert.equal(invalidHostStatus, 403);
  assert.equal((await fetch(`${url}/missing`)).status, 404);
  assert.equal((await fetch(`${url}/%zz`)).status, 400);
});

test('unexpected filesystem errors surface as server errors, not missing data', async t => {
  const diagnostics = t.mock.method(console, 'error', () => {});
  const server = createLocalServer({ dist: '\0' });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  assert.equal((await fetch(`http://127.0.0.1:${server.address().port}`)).status, 500);
  assert.equal(diagnostics.mock.callCount(), 1);
  assert.ok(diagnostics.mock.calls[0].arguments[0] instanceof Error);
});
