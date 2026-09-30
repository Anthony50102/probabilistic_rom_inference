import { Niivue } from '@niivue/niivue';

// Publication panels: a single canvas, fixed camera and colour range, no widgets or text.
// Volume style: a clipped volume rendering of one field. MRI style: the anatomy's in-plane slice with one
// field as a translucent overlay coloured from 0 whose alpha fades as (u / threshold)^2 below the threshold.
const status = document.getElementById('status');
const frame = document.getElementById('frame');
const canvas = document.getElementById('gl');
const ZERO_TO_MAX_TRANSPARENT_BELOW_MIN = 1;
const painted = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

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
  const find = id => {
    const panel = manifest.panels.find(item => item.id === id);
    if (!panel) throw new Error(`Unknown panel ${id}`);
    return panel;
  };
  const volumePanel = async (id, overrides = {}) => {
    const settings = { ...render, ...overrides };
    const panel = find(id);
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
    await painted();
    const volume = nv.volumes[0];
    return { dims: volume.dims.slice(1, 4), cal_min: volume.cal_min, cal_max: volume.cal_max };
  };
  const slicePanel = async (id, overrides = {}) => {
    const settings = { ...render, ...overrides };
    const panel = find(id);
    const { anatomy } = manifest;
    nv.opts.backColor = settings.background;
    await nv.loadVolumes([
      { url: `/data/figure/${anatomy.file}`, colormap: 'gray', cal_min: anatomy.cal_min, cal_max: anatomy.cal_max },
      { url: `/data/figure/${panel.file}`, colormap: settings.colormap, cal_min: settings.threshold,
        cal_max: panel.cal_max },
    ]);
    const overlay = nv.volumes[1];
    // This type's per-voxel alpha, (u / cal_min)^2, saturates above cal_min, so a volume opacity would only
    // thin the fringe; the global overlay alpha makes the whole overlay translucent.
    overlay.colormapType = ZERO_TO_MAX_TRANSPARENT_BELOW_MIN;
    nv.overlayAlphaShader = settings.opacity;
    nv.overlayOutlineWidth = settings.outline_width ?? 0;
    nv.updateGLVolume();
    nv.setSliceType(nv.sliceTypeAxial);
    nv.scene.crosshairPos = nv.vox2frac(settings.slice_voxel);
    nv.drawScene();
    await painted();
    // The composer crops each screenshot to the slice's tile (canvas pixels).
    return { dims: overlay.hdr.dims.slice(1, 4), cal_min: overlay.cal_min, cal_max: overlay.cal_max,
      tile: [...nv.screenSlices[0].leftTopWidthHeight], canvas: [canvas.width, canvas.height] };
  };
  window.renderPanel = manifest.style === 'mri' ? slicePanel : volumePanel;
  window.colormapTables = names => Object.fromEntries(names.map(name => [name,
    { name, ...Object.fromEntries(['R', 'G', 'B', 'A', 'I'].map(key => [key, [...nv.colormapFromKey(name)[key]]])) }]));
  status.dataset.state = 'ready';
}

init().catch(error => {
  status.textContent = error.message;
  status.dataset.state = 'error';
});
