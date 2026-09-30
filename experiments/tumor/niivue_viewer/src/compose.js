// Lays out the MRI-style publication figure from the NiiVue panels written by scripts/render-figure.js:
// column and row labels, per-panel errors, a scale bar and a colour bar from NiiVue's own colormap table.
const status = document.getElementById('status');
const figure = document.getElementById('figure');

const node = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};

async function json(name) {
  const response = await fetch(`/data/figure/${name}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${name} missing; run niivue_benchmark_figure.py export, then render-figure`);
  return response.json();
}

function picture(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Cannot load ${url}`));
    image.src = url;
  });
}

// NiiVue's 256-entry lookup table: RGB interpolated linearly between the colormap nodes I.
function lookup(table) {
  return Array.from({ length: 256 }, (_, index) => {
    const upper = Math.max(1, table.I.findIndex(position => position >= index));
    const lower = upper - 1;
    const weight = (index - table.I[lower]) / (table.I[upper] - table.I[lower]);
    return ['R', 'G', 'B'].map(key => Math.round(table[key][lower] + weight * (table[key][upper] - table[key][lower])));
  });
}

// The overlay as drawn over black: colours run from 0 to max (NiiVue's zero-to-max overlay type) and the alpha
// fades as (u / threshold)^2 below the threshold. The panels' global overlay opacity is not applied here.
function colorbar(spec, threshold, table) {
  const colours = lookup(table);
  const bar = node('div', 'bar');
  const gradient = node('canvas');
  gradient.width = 1024;
  gradient.height = 1;
  const context = gradient.getContext('2d');
  const pixels = context.createImageData(gradient.width, 1);
  for (let x = 0; x < gradient.width; x++) {
    const value = (x + .5) / gradient.width * spec.max;
    const alpha = Math.min(1, (value / threshold) ** 2);
    const colour = colours[Math.min(255, Math.round(255 * value / spec.max))].map(c => Math.round(alpha * c));
    pixels.data.set([...colour, 255], 4 * x);
  }
  context.putImageData(pixels, 0, 0);
  bar.append(gradient);
  const ticks = node('div', 'ticks');
  for (const tick of spec.ticks) {
    const mark = node('span', '', String(Number(tick.toFixed(3))));
    mark.style.left = `${100 * tick / spec.max}%`;
    ticks.append(mark);
  }
  // Odd parts of the label are italic (the symbol).
  const label = node('div', 'label');
  spec.label.forEach((part, index) => label.append(index % 2 ? node('i', '', part) : document.createTextNode(part)));
  label.append(`; the overlay fades out below ${threshold}`);
  const wrapper = node('div', 'colorbar');
  wrapper.append(bar, ticks, label);
  return wrapper;
}

async function init() {
  const [manifest, render, tables] = await Promise.all(
    ['manifest.json', 'panels/render.json', 'panels/colormaps.json'].map(json));
  if (manifest.style !== 'mri' || !render.tiles) throw new Error('Render the MRI-style manifest first');
  const layout = manifest.figure;
  const panels = new Map(manifest.panels.map(panel => [panel.id, panel]));
  figure.style.width = `${layout.width_px}px`;
  figure.append(node('div', 'title', layout.title));
  const grid = node('div', 'grid');
  grid.style.setProperty('--columns', layout.columns.length);
  grid.append(node('div'), ...layout.columns.map(column => node('div', 'column', column.label)));
  const cells = [];
  for (const [row, label] of manifest.rows) {
    grid.append(node('div', 'row', label.replace(/\s+/g, ' ')));
    for (const column of layout.columns) {
      const id = column.panels[row];
      const cell = node('div', 'cell');
      const canvas = node('canvas');
      canvas.style.aspectRatio = `${render.width} / ${render.height}`;
      cell.append(canvas);
      const error = panels.get(id).relative_field_error_percent;
      if (error !== null) cell.append(node('span', 'error', `${error.toFixed(1)}%`));
      grid.append(cell);
      cells.push({ id, cell, canvas });
    }
  }
  figure.append(grid, colorbar(layout.colorbar, render.threshold, tables[render.colormap]));
  const [first] = cells;
  const scale = node('div', 'scalebar');
  scale.style.width = `${first.cell.clientWidth * layout.scale_bar_mm / render.fov_width_mm}px`;
  scale.append(node('span', '', `${layout.scale_bar_mm} mm`));
  first.cell.append(node('span', 'corner', layout.corner_label), scale);
  await Promise.all(cells.map(async ({ id, canvas }) => {
    const image = await picture(`/data/figure/panels/${id}.png`);
    const { tile, canvas: size } = render.tiles[id];
    const [sx, sy] = [image.naturalWidth / size[0], image.naturalHeight / size[1]];
    canvas.width = Math.round(canvas.clientWidth * devicePixelRatio);
    canvas.height = Math.round(canvas.clientHeight * devicePixelRatio);
    const context = canvas.getContext('2d');
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, tile[0] * sx, tile[1] * sy, tile[2] * sx, tile[3] * sy, 0, 0, canvas.width, canvas.height);
  }));
  await document.fonts.ready;
  status.dataset.state = 'ready';
}

init().catch(error => {
  status.textContent = error.message;
  status.dataset.state = 'error';
});
