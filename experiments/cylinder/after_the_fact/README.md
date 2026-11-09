# After-the-fact scripts (not part of the declared design)
Written after the declared fits had started, and disclosed in `../DESIGN.md` (Amendment and Results).
- `centre_diag.py`: noise-free least squares on centred vs uncentred POD (refuted the centring hypothesis).
- `draw_diag.py`: why posterior draws blow up (spectra of the draws vs the posterior mean).
- `rescore_mean.py RUN_DIR...`: posterior-mean operator forecast from a saved `04_unified.npz` (no refit). Writes
  `rescore_mean.{json,npz}` next to it.
- `make_figures.py RESULTS_DIR OUT_DIR`: figures 01-05 and `table.txt` from the pulled `summary.json`,
  `plotdata.npz` and `rescore_mean.*` files.
- `results_table.txt`: the table for all runs (9 Nov 2026).
