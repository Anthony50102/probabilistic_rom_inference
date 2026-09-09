import { Niivue } from '@niivue/niivue';
import { frameLabel, kinds, localVolumeURL, methodLabels, validateManifest } from './contract.js';
import './style.css';

const $ = id => document.getElementById(id);
const viewers = new Map();
const titles = { truth: 'FOM truth', prediction: 'ROM reduced-median reconstruction',
  error: 'Absolute error', uncertainty: 'Voxelwise 90% interval width' };
let manifest, current, loading = false;

function options(id, values, label = value => value) {
  const previous = $(id).value;
  $(id).replaceChildren(...values.map(value => new Option(label(value), value)));
  if (values.map(String).includes(previous)) $(id).value = previous;
}

function refreshMethods() {
  const rows = manifest.cases.filter(item => item.schema === $('schema').value);
  options('method', [...new Set(rows.map(item => item.method))], method => methodLabels[method]);
  refreshDoses();
}

function refreshDoses() {
  const rows = manifest.cases.filter(item => item.schema === $('schema').value && item.method === $('method').value);
  options('dose', [...new Set(rows.map(item => item.dose_scale))].sort(), dose => `${dose}×`);
}

function setDisabled(disabled) {
  for (const id of ['schema', 'method', 'dose', 'slice', 'range', 'time']) $(id).disabled = disabled;
}

function showError(error) {
  $('status').textContent = `Cannot display data: ${error.message}. Export completed results, then reload this page.`;
  $('status').dataset.state = 'error';
}

function updateFrame() {
  if (!current) return;
  const index = Number($('time').value);
  $('day').textContent = frameLabel(current, index);
  for (const [kind, viewer] of viewers) {
    if (current.volumes[kind] && viewer.volumes[0]) {
      // Niivue 0.68.2 uses the volume UUID, not its array index.
      viewer.setFrame4D(viewer.volumes[0].id, index);
    }
  }
}

function updateRange() {
  if (!current) return;
  const truth = current.volumes.truth, prediction = current.volumes.prediction;
  const full = $('range').value === 'full';
  for (const [kind, viewer] of viewers) {
    const info = current.volumes[kind];
    if (!info || !viewer.volumes[0]) continue;
    const shared = kind === 'truth' || kind === 'prediction';
    const low = shared ? Math.min(0, truth.min, full && prediction ? prediction.min : 0) : 0;
    const high = shared ? Math.max(truth.max, full && prediction ? prediction.max : truth.max) : info.max;
    viewer.volumes[0].cal_min = low;
    viewer.volumes[0].cal_max = high > low ? high : low + 1;
    viewer.updateGLVolume();
    $(`${kind}-range`).textContent = `Color range: ${low.toPrecision(4)} … ${(high > low ? high : low + 1).toPrecision(4)}; actual: ${info.min.toPrecision(4)} … ${info.max.toPrecision(4)}`;
  }
}

async function loadCase() {
  if (loading) return;
  loading = true;
  setDisabled(true);
  current = manifest.cases.find(item => item.schema === $('schema').value &&
    item.method === $('method').value && item.dose_scale === Number($('dose').value));
  $('status').dataset.state = 'loading';
  $('status').textContent = 'Loading local NIfTI volumes…';
  try {
    if (!current) throw new Error('No completed case matches these selections');
    for (const kind of kinds) {
      const info = current.volumes[kind];
      $(`${kind}-panel`).hidden = !info;
      const viewer = viewers.get(kind);
      // Await every load (including failures) before controls can initiate another case.
      if (info) {
        await viewer.loadVolumes([{ url: localVolumeURL(info.file), name: `${kind}.nii.gz`,
          colormap: kind === 'error' || kind === 'uncertainty' ? 'hot' : 'viridis',
          cal_min: info.min, cal_max: Math.max(info.max, info.min + 1e-9) }]);
        if (viewer.volumes[0]?.nFrame4D !== current.times_days.length) {
          throw new Error(`${kind} NIfTI frame count differs from manifest`);
        }
        viewer.setSliceType(Number($('slice').value));
      } else {
        while (viewer.volumes.length) viewer.removeVolume(viewer.volumes[0]);
      }
    }
    $('time').max = current.times_days.length - 1;
    $('time').value = Math.min(Number($('time').value), current.times_days.length - 1);
    updateRange();
    updateFrame();
    $('metadata').textContent = JSON.stringify({
      ...current.metadata, sources: current.sources, basis_fingerprint: current.basis_fingerprint,
      times_days: current.times_days, coordinates: current.coordinates,
    }, null, 2);
    $('status').textContent = current.status === 'no_stable_solves'
      ? 'No stable solves: truth only. Prediction, error and uncertainty are unavailable, not zero.'
      : `${current.metadata.n_stable}/${current.metadata.n_total} stable solves. Fit ends day ${current.training_end}; nominal-dose training for all doses.`;
    $('status').dataset.state = 'ready';
  } catch (error) {
    for (const kind of kinds) $(`${kind}-panel`).hidden = true;
    $('metadata').textContent = 'No valid dataset loaded.';
    current = null;
    showError(error);
  } finally {
    loading = false;
    setDisabled(false);
    if (!current) for (const id of ['slice', 'range', 'time']) $(id).disabled = true;
  }
}

async function init() {
  const response = await fetch('/data/manifest.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('Local manifest missing; run the exporter');
  manifest = validateManifest(await response.json());
  $('warnings').textContent = (manifest.warnings ?? []).join('\n') || 'None';
  if (!manifest.cases.length) throw new Error('No completed results in manifest (see skipped results)');
  for (const kind of kinds) {
    const panel = document.createElement('article');
    panel.id = `${kind}-panel`;
    const title = document.createElement('h2');
    title.textContent = titles[kind];
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', titles[kind]);
    canvas.id = `${kind}-canvas`;
    const range = document.createElement('p');
    range.id = `${kind}-range`;
    panel.append(title, canvas, range);
    $('panels').append(panel);
    const viewer = new Niivue({ isColorbar: true, isOrientCube: false, isOrientationTextVisible: false,
      showAllOrientationMarkers: false, dragAndDropEnabled: false, isResizeCanvas: true,
      backColor: [0.035, 0.045, 0.06, 1] });
    await viewer.attachToCanvas(canvas);
    viewers.set(kind, viewer);
  }
  for (const viewer of viewers.values()) {
    viewer.broadcastTo([...viewers.values()].filter(other => other !== viewer), { '2d': true, '3d': true });
  }
  options('schema', [...new Set(manifest.cases.map(item => item.schema))],
    schema => schema.replace('dense_', '').replace('_noise', ' noise'));
  refreshMethods();
  $('schema').addEventListener('change', () => { refreshMethods(); void loadCase(); });
  $('method').addEventListener('change', () => { refreshDoses(); void loadCase(); });
  $('dose').addEventListener('change', () => void loadCase());
  $('time').addEventListener('input', updateFrame);
  $('range').addEventListener('change', updateRange);
  $('slice').addEventListener('change', () => {
    for (const viewer of viewers.values()) viewer.setSliceType(Number($('slice').value));
  });
  await loadCase();
}

init().catch(showError);
