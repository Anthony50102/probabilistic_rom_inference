# Local TNBC Niivue viewer

Four linked views show synthetic tumor FOM truth, ROM prediction, absolute error,
and voxelwise 90% uncertainty width. Select observation noise, method, dose, and
actual saved prediction time. Missing result combinations are not selectable;
zero-stable-solve combinations show **truth only**.

## Export and run

From the repository root, using the existing `prob_rom` environment:

```bash
conda run -n prob_rom python experiments/tumor/export_chemo_niivue.py
cd experiments/tumor/niivue_viewer
npm ci
npm run build
npm start
```

Open **http://127.0.0.1:5173** (not `localhost`). Stop with Ctrl-C.
Node >=22.12 is required. `PORT=5174 npm start` changes the port.
The exporter needs NumPy, SciPy and nibabel (already available in `prob_rom`).
No training, fitting, or FOM simulation is performed.

For an initial subset, from the repository root:

```bash
conda run -n prob_rom python experiments/tumor/export_chemo_niivue.py \
  --schema dense_low_noise --method 04_unified_chemo --dose 1 \
  --times 5 20 40 60 70 80 90 100 110
```

Run export again as results arrive, then refresh the page; rebuilding is unnecessary
for data changes. **Each export replaces the manifest selection**, not merges it.
The default results root is
`experiments/tumor/results/chemo_matched_80_5_70_110_v1`.
`--results-root`, `--output-dir`, `--schema`, `--method`, `--dose`, `--times`,
`--chunk-voxels` and `--strict` are supported; see `--help`.
Default exports go in `public/data`. To view a custom output directory, export
there for archival purposes, then export the desired selection to `public/data`.

Incomplete JSON/NPZ pairs, absent corresponding fit/protocol files, or inconsistent
provenance produce explicit warnings and are omitted (including warnings in the
manifest). `--strict` fails instead, leaving the previous manifest intact.
An empty manifest is valid and explains why nothing can be viewed.

## Scientific meaning and resource use

- Coordinates are a **simulation grid; not registered patient coordinates**.
  Source arrays have no anatomical orientation, origin, or patient-space affine.
  NIfTI axes follow NumPy `reshape(grid_shape)`; the affine is spacing-scaled
  with zero grid origin. The sform stores this display transform (code 2);
  it is not evidence of anatomical registration. Anatomical labels are hidden.
- Times are nearest **actual saved** `t_pred` values, de-duplicated and sorted,
  normally nine frames. They are nonuniform: read `times_days` in the manifest,
  not the NIfTI fourth-axis spacing. Day 70 marks the training end.
- Truth uses the evaluator's bounded cubic time interpolation. Spatial chunks
  prevent allocating a whole-grid cubic spline; interpolated truth is reused
  across methods/noise at the same dose/times.
- Prediction is `basis @ median_comp + basis_shift`, **not** the voxelwise
  median. Error is the absolute difference between truth and this displayed field.
- Uncertainty is P95−P5 across **reconstructed stable-sample fields at each voxel**.
  It is never computed by reconstructing componentwise reduced quantiles.
  Stable-sample uncertainty excludes failed trajectories; stable does not imply
  physically plausible or accurate. Counts and evaluation metrics are shown.
- Original signs and magnitudes are retained (no clipping / masking).
  Default truth/prediction colors share the truth range, so poor predictions may
  saturate. Select “Combined full range” to include all prediction values.
  Each panel states its color range and actual global range. Error and width use
  their own full ranges. Ranges are global over the exported frames.
- Full FOM NPZ decompression still requires roughly 0.6 GB for the default grid.
  Ensembles are reconstructed in voxel chunks (8192 by default), one time at a
  time. Float32 volumes reduce RAM/disk usage; percentile computation uses all
  saved stable samples, without ensemble downsampling.
- The manifest records source filenames and SHA256s, model/data/basis
  fingerprints, actual days, metadata, full metrics, and volume data fingerprints.
  Model and data IDs must match the corresponding fit's JSON sidecar or
  scalar `model_id` / `data_fingerprint` fields embedded in its NPZ. If both
  are present, both must match. FOM source IDs are checked when present.

## Local-only and files

Niivue is installed from npm and bundled by Vite. No CDN, telemetry, external
image/model upload, remote fonts, or third-party runtime resources are used.
Package installation requires internet access; after installation/build the viewer
works offline. A restrictive Content Security Policy prohibits off-origin
connections. The read-only Node server binds **only 127.0.0.1**, checks Host/Origin,
and serves only `dist` and `public/data`, rejecting traversal, hidden paths, and
escaping symlinks. Do not proxy/expose this server publicly.

Generated NIfTI files, manifests, `dist`, and `node_modules` are ignored by Git.
Use an empty export directory initially. An ownership marker prevents overwriting
unrelated directories. Volumes use content-addressed filenames; repeated exports
reuse matching files and never delete old exports. Reclaim disk space manually
only after confirming which generated files are no longer needed.

## Validation

From this directory:

```bash
npm test
npm run build
conda run -n prob_rom python -m unittest discover -s tests -p 'test_*.py'
```

Tests use built-in Node and Python runners. They check local server boundaries,
manifest failure modes, actual time selection, reconstructed voxel quantiles,
cubic interpolation, and NIfTI data/shape/affine. Browser rendering requires WebGL2.
The installed Niivue 0.68.2 declarations specify
`setFrame4D(volume.id, frameIndex)`; the viewer uses that UUID-based API.

Optional real-browser check: with `npm start` running and a separate Chrome
instance launched with a project-local profile and a loopback CDP port:

```bash
mkdir -p .browser-test
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --user-data-dir="$PWD/.browser-test" \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port=9224 \
  --disable-background-networking --disable-component-update --disable-sync \
  --no-first-run --no-default-browser-check --disable-default-apps \
  --disable-features=OptimizationHints,MediaRouter \
  --enable-unsafe-swiftshader about:blank
```

In another terminal in this directory, run `node tests/browser-smoke.js`.
Change the Chrome executable for your platform. The script verifies actual
WebGL loading, dose/time controls, slice/3D views, empty/missing/zero-stable
states (via in-memory request overrides, not data edits), and that all page
requests remain local. Stop both validation processes afterward. Software
WebGL is only enabled for this isolated smoke-test browser.
