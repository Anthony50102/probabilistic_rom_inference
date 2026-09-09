export const methodLabels = { '04_unified_chemo': 'Bayesian OpInf', '05_neural_ode_chemo': 'Neural ODE' };
export const kinds = ['truth', 'prediction', 'error', 'uncertainty'];

export function localVolumeURL(file) {
  if (typeof file !== 'string' || !/^chemo-niivue-[a-z]+-[a-f0-9]{64}\.nii\.gz$/.test(file)) {
    throw new Error('Manifest contains an invalid local volume filename');
  }
  return `/data/${file}`;
}

export function validateManifest(manifest) {
  if (manifest?.format_version !== 1 || !Array.isArray(manifest.cases)) throw new Error('Unsupported manifest format');
  const ids = new Set();
  for (const item of manifest.cases) {
    if (typeof item.id !== 'string' || ids.has(item.id)) throw new Error('Invalid / duplicate case ID');
    ids.add(item.id);
    if (!Array.isArray(item.times_days) || !item.times_days.length ||
        item.times_days.some((t, i, times) => !Number.isFinite(t) || (i > 0 && t <= times[i - 1]))) {
      throw new Error('Invalid saved prediction times');
    }
    if (!['ready', 'no_stable_solves'].includes(item.status) ||
        !methodLabels[item.method] || !Number.isFinite(item.dose_scale) ||
        !Number.isFinite(item.training_end) || typeof item.schema !== 'string') throw new Error('Invalid case metadata');
    const required = item.status === 'ready' ? kinds : ['truth'];
    for (const kind of required) {
      const volume = item.volumes?.[kind];
      localVolumeURL(volume?.file);
      if (!Number.isFinite(volume.min) || !Number.isFinite(volume.max) || volume.min > volume.max) {
        throw new Error('Invalid volume intensity range');
      }
    }
    if (item.status === 'no_stable_solves' && kinds.slice(1).some(k => item.volumes[k])) {
      throw new Error('Unstable case must not contain fabricated predictions');
    }
  }
  return manifest;
}

export function frameLabel(item, index) {
  const day = item.times_days[index];
  return `day ${day.toFixed(3)} · ${day <= item.training_end ? 'fit interval' : 'forecast'}`;
}
