import { Niivue } from '@niivue/niivue';

// Publication panels: a single canvas, fixed camera and colour range, no widgets or text.
const status = document.getElementById('status');
const frame = document.getElementById('frame');
const canvas = document.getElementById('gl');

async function init() {
  const response = await fetch('/data/figure/manifest.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('Figure manifest missing; run niivue_benchmark_figure.py export');
  const manifest = await response.json();
  const render = manifest.render;
  document.body.style.cssText = 'margin:0;overflow:hidden;background:transparent';
  // NiiVue sizes the canvas to its parent (and paints the parent black), so the frame fixes the panel size.
  frame.style.cssText = `width:${render.width}px;height:${render.height}px`;
  const nv = new Niivue({
    backColor: render.background, isColorbar: false, isOrientCube: false, show3Dcrosshair: false,
    isOrientationTextVisible: false, showAllOrientationMarkers: false, dragAndDropEnabled: false,
    isResizeCanvas: true, crosshairWidth: 0, logLevel: 'error',
  });
  await nv.attachToCanvas(canvas);
  window.renderPanel = async (id, overrides = {}) => {
    const settings = { ...render, ...overrides };
    const panel = manifest.panels.find(item => item.id === id);
    if (!panel) throw new Error(`Unknown panel ${id}`);
    nv.opts.backColor = settings.background;
    await nv.loadVolumes([{ url: `/data/figure/${panel.file}`, colormap: settings.colormap ?? panel.colormap,
      cal_min: settings.cal_min ?? panel.cal_min, cal_max: panel.cal_max }]);
    nv.setSliceType(nv.sliceTypeRender);
    await nv.setVolumeRenderIllumination(settings.illumination);
    nv.setRenderAzimuthElevation(settings.azimuth, settings.elevation);
    nv.volScaleMultiplier = settings.zoom;
    // The cut face shows the interior; the plane itself is not drawn.
    nv.setClipPlaneColor([0, 0, 0, 0]);
    nv.setClipPlane(settings.clip ?? [2, 0, 0]);
    nv.drawScene();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const volume = nv.volumes[0];
    return { dims: volume.dims.slice(1, 4), cal_min: volume.cal_min, cal_max: volume.cal_max };
  };
  window.colormapTables = names => Object.fromEntries(names.map(name => [name,
    { name, ...Object.fromEntries(['R', 'G', 'B', 'A', 'I'].map(key => [key, [...nv.colormapFromKey(name)[key]]])) }]));
  status.dataset.state = 'ready';
}

init().catch(error => {
  status.textContent = error.message;
  status.dataset.state = 'error';
});
