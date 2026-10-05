# Chemotherapy on the TumorTwin tutorial schedule (exploratory, production only)

## Status and scope

**Exploratory; production (Bayesian OpInf) only.** The Neural ODE has not been
run on this design, and nothing here is in the manuscript. The two cases,
`standard-regimen-chemo` and `alternative-regimen-chemo`, are not run by
default: the runners default to the three reported cases
(`benchmark_cases.REPORTED_CASES`), whose recipes, fingerprints, and results
are unchanged.

The question was whether a more realistic chemotherapy task is worth moving
the paper to: let the tumour grow untreated, then treat it on the patient's
own schedule, as in TumorTwin's TNBC tutorial. Production was run first, so
that no Neural-ODE ensembles are trained on a task the method does badly on.

**Outcome.** Production fits the training scans closely (2-3% error) and
always predicts the right direction of a dose change, but it forecasts too
much shrinkage. Median forecast errors are 11-35% across doses, against
9.6-18.9% on the reported chemo tasks. It captures only 40-50% of the true
difference between doses, against 79-80% on the reported multi-dose task. A
noise-free least-squares check shows that the model form can represent the
forecasts (3-7% error when fitted with hindsight). The limit is how little the
52-day training window determines about the future; inference does not cause
it. The 2x arm looks best only because two errors cancel there (see "Why 2x
looks best").

## Design

| | Reported chemo cases | Tutorial schedule (this record) |
|---|---|---|
| Synthetic truth | `k=.025`, `d=.05`, sensitivity .5, training strength .5 | Tutorial's calibrated twin: `k=.0521`, `d=.0965`, sensitivity .23 (`TUTORIAL_TWIN`), training strength 1 |
| Doses | Days 20, 40, 60, 80, 100 | Patient's record: weekly, days 35-112 (12 doses; `PATIENT_DOSE_DAYS`) |
| Training | Days 5-70 (3 doses), 120 segmented scans, 1% noise | Days 5-57 (4 doses, up to the second MRI), 120 segmented scans, 1% noise |
| Forecast | Days 70-110 (2 doses) | Days 57-119 (8 doses, to the third MRI) |
| Future doses | Single: unchanged. Multi: 0.25/0.5/0.75/1x | Standard: unchanged (1x). Alternative: 0.5/1/1.5/2x |
| Seeds | Development 45; multi 48-50, single 51-53 | Development 45; alternative 48-50, standard 51-53 |

Everything else is shared with the reported cases: anatomy `TNBC_demo_001`,
capacity θ=1, drug decay .7/day, the segmented observation model, the
standardized production settings (tempered dynamics rows), and the Galerkin
operator rule (`cABN` requested; `cAN` on the uncentred basis).

The twin parameters are the tutorial's Levenberg-Marquardt fit to the first
two MRIs. Its total tumour cell count is within 2.1% (day 57) and 1.0% (day
119) of the patient's measured MRIs, and day 119 was not used in the
calibration. Under the standard schedule the synthetic tumour grows untreated
to a peak at the first dose (day 35). The four training doses cut it by about
40% by day 57. Over the forecast it declines only about 20% more, dipping
after each dose and partly regrowing in between.

**Dose information.** The patient record (`TNBC_demo_001.json`) lists twelve
taxol (paclitaxel) administrations, weekly on days 35-112 after the first
MRI, each with a normalised dose of 1 ("mg"). It does not record the clinical
amount. In the model, each dose adds `sensitivity × strength` to the kill
rate α(t) of the term -α(t)u, and α then decays at .7/day. Strength is 1 in
training and 0.5-2 in the alternative futures
(`BenchmarkCase.pulse_coefficients`).

## Running

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py alternative-regimen-chemo --seeds 45   # development
conda run -n prob_rom python 04_unified_benchmark.py alternative-regimen-chemo              # 48-50
conda run -n prob_rom python 04_unified_benchmark.py standard-regimen-chemo                 # 51-53
conda run -n prob_rom python tutorial_chemo_structure_check.py                              # clean-data check (about a minute)
conda run -n prob_rom python tutorial_chemo_rate_decomposition.py                           # regrowth vs drug split (seconds)
```

On first use, preparation generates the TumorTwin sources. These are the
untreated-then-weekly history plus one future branch per strength, about 20 s
each, cached under names containing `s0.23` and `days1fe6e04f`. A seed then
takes about 10 minutes (alternative) or 3-5 minutes (standard). Results go to
`results/benchmarks/<case>/segmented_observed_none_r4/seed<N>/`. The
development candidates add `--pod-rank 3` and/or `--pod-centering mean`.

## Basis selection (rule declared before any result)

The reported seeds use the development-45 candidate with the lowest mean
production error over days 57-119 at the four alternative strengths. The
candidates are mean-centred or uncentred, rank 3 or 4, among those the
Gavish-Donoho noise cap allows (it allowed all four). The Neural-ODE stage of
the earlier rule does not apply yet. The standard case uses the same basis,
as single-dose uses the multi-dose basis.

| Candidate (seed 45) | Mean full-field error, days 57-119 |
|---|---|
| **uncentred rank 4** | **18.17%** |
| uncentred rank 3 | 22.46% |
| mean-centred rank 4 | 206.10% |
| mean-centred rank 3 | 213.25% |

Uncentred rank 4 was selected. It is also the case default, so nothing
changed before seeds 48-53 were fitted, and each was run once.

## Results

Relative full-field forecast error over days 57-119 (%), including the basis
residual. Every fit completed its 12,000 updates, and all 64 posterior draws
completed in every cell.

| Case, future dose | Per seed | Median | Days 57-88 | Days 88-119 | Training fit (5-57) | Basis floor |
|---|---|---|---|---|---|---|
| Alternative, 0.5x | 31.3 / 35.1 / 35.1 | 35.1 | 19.4 | 41.7 | 2.1 | 9.4 |
| Alternative, 1x | 24.3 / 25.2 / 24.6 | 24.6 | 13.7 | 36.4 | 2.1 | 5.8 |
| Alternative, 1.5x | 14.4 / 14.4 / 13.5 | 14.4 | 8.6 | 31.7 | 2.1 | 3.7 |
| Alternative, 2x | 9.6 / 12.2 / 11.1 | 11.1 | 5.8 | 44.9 | 2.1 | 2.6 |
| Standard, 1x (51-53) | 29.1 / 16.4 / 32.6 | 29.1 | 14.8 | 43.9 | 3.1 | 5.7 |

Per-window values are medians. The basis floor is the error of the best
rank-4 approximation of the truth. The 88-119 window is relative to a
tumour that has shrunk at high doses, so its percentages are large even
where absolute errors are small.

**Too much shrinkage.** The table below gives the tumour burden on day 119 as
a fraction of the true day-57 burden.

| Future dose | Truth | Production (per seed) |
|---|---|---|
| 0.5x | 2.12 (regrows) | 1.02 / 0.93 / 0.93 |
| 1x | 0.79 | 0.36 / 0.36 / 0.37 |
| 1.5x | 0.26 | 0.13 / 0.14 / 0.15 |
| 2x | 0.07 | 0.04 / 0.06 / 0.06 |
| Standard 1x (51-53) | 0.79 | 0.30 / 0.48 / 0.28 |

Under the patient's own schedule the true tumour keeps about 80% of its
day-57 burden at day 119; production predicts about a third. At half dose the
true tumour doubles, while production predicts it roughly holds steady.

**Dose effects: right direction, half the size.** The predicted difference
from the 1x arm has the correct sign on every seed at every forecast time
after the first changed dose (196 of 196 times). Its size on day 119 is
0.42-0.50 of the truth at 0.5x, 0.40-0.44 at 1.5x and 0.41-0.45 at 2x. On the
reported multi-dose task the median is 0.79-0.80.

**Posterior draws.** Of the 64 draws per seed, those within 25% full-field
error number 9-17 at 0.5x, 30 at 1x, 44-47 at 1.5x, 40-52 at 2x, and 17-43
for the standard case. The draw burden band (5-95%) contains the true burden
at 96-100% of forecast times at 1.5x and 2x. At 0.5x and 1x it does so at only
8-20%, and for the standard case at 8-81%. These are empirical draw bands, not
calibrated uncertainty.

## Why: noise-free least-squares check

`tutorial_chemo_structure_check.py` removes noise, Gaussian processes, and
Bayesian inference. It fits the operators by plain least squares, in the same
integral form, to the noise-free truth projected onto each acquisition's
basis, then forecasts every arm from day 5. Errors are reduced-coordinate,
days 57-119, and ranges cover acquisitions 45 and 48-50.

| Fitted on | Form | 1x | 0.5x | 1.5x | 2x |
|---|---|---|---|---|---|
| Training window (1x, days 5-57) | `cAN` (reported) | 14-26 | 95-109 | 205-229 | 291-323 |
| Training window (1x, days 5-57) | `cAHN` (+ logistic term) | diverges | diverges | diverges | diverges |
| Hindsight: all arms, days 5-119 | `cAN` | 7.1-7.2 | 2.9-3.0 | 3.6-3.7 | 6.6-6.7 |
| Hindsight: all arms, days 5-119 | `cAHN` | 29-42 | 21-32 | 34-48 | 36-50 |
| Production, noisy scans (for reference) | `cAN` | 21-25 | 28-34 | 12-14 | 9-12 |

Three conclusions:

1. **The model form is adequate.** With hindsight, the same `cAN` form
   forecasts every dose to within 3-7%.
2. **The training window does not determine the future.** Fitted only to what
   the method sees, but without noise, least squares misses the changed doses
   by 95-323%. Production, which sees the same window with noise, does far
   better at the changed doses (9-34%) and about as well at 1x. So the
   Bayesian inference is not what loses accuracy. The information is missing
   from days 5-57. Training sees the steep first response: about 40% lost in
   three weeks from the peak. It does not see the slower decline that
   follows. In the true (logistic) model, thinned tissue regrows faster per
   cell between doses. Production's fitted regrowth does not speed up, so it
   keeps shrinking the tumour too fast (measured in the next section).
3. **Adding the logistic term does not help.** `cAHN` fitted to the training
   window diverges before day 119 on every arm. Even with hindsight, its
   rollouts (21-50%) are worse than those of `cAN`. This matches the
   manuscript's choice to omit `H` as unidentifiable from one training
   trajectory.

## Why 2x looks best: two errors cancel

On the burden plots, production looks accurate at 2x and poor at the other
doses. `tutorial_chemo_rate_decomposition.py` explains this by splitting the
burden's growth rate into regrowth and drug kill. In the truth, the drug
removes exactly α(t) of the cells per day (the term -α(t)u), and regrowth per
cell rises as the drug thins the tumour (the logistic term). The script
evaluates production's operators on the true states, so the differences come
from the operators. Means over days 57-119 for seeds 48-50, in %/day:

| Future dose | True regrowth | Production regrowth | Production kill per unit dose (truth 1) | Regrowth error | Drug error | Net error |
|---|---|---|---|---|---|---|
| 0.5x | 3.6 | 1.6-1.9 | 0.74-0.77 | -2.0 to -1.7 | +0.5 to +0.6 | -1.4 to -1.2 |
| 1x | 4.1 | 1.7-2.0 | 0.74-0.78 | -2.4 to -2.2 | +1.0 to +1.2 | -1.3 to -1.2 |
| 1.5x | 4.4 | 1.8-2.1 | 0.75-0.79 | -2.6 to -2.3 | +1.4 to +1.7 | -1.0 to -0.8 |
| 2x | 4.5 | 2.1-2.6 | 0.75-0.79 | -2.4 to -2.0 | +1.8 to +2.2 | -0.4 to +0.2 |

- **Regrowth too slow at every dose.** Before treatment, production's
  regrowth matches the truth (2.4-2.5 against 2.5%/day). Once the drug thins
  the tumour, the true regrowth climbs to 3.6-4.5%/day, but production's
  stays near 2%/day. This error is similar at every dose and shrinks the
  tumour too fast.
- **Drug too weak, more so at higher doses.** Production's drug kills
  0.74-0.79 per unit dose instead of 1. This error grows with the dose. At 2x
  it cancels most of the regrowth error; at 0.5x and 1x, less than half.
- **Training cannot tell the two apart.** Over days 35-57, production's
  regrowth is already 0.8-1.1%/day too slow and its drug 15-21% too weak. The
  two cancel to within 0.05%/day, so the training fit is accurate either way.
  Noise-free least squares splits the training window correctly (regrowth 3.3
  against 3.1%/day, kill 1.06), but its rates run away in the forecast (kill
  2.5-4.6 per unit dose). With hindsight, its kill is 1.01-1.06.
- **Scale flatters 2x.** The true 2x tumour ends at 917 cells, 7% of its
  day-57 burden, so on a linear axis every curve sits near zero. Production's
  day-119 gap is 120-420 cells at 2x, against 13,500-14,700 at 0.5x and
  5,200-5,300 at 1x. As a fraction of the truth, production's day-119 burden
  is 0.44-0.59 at 0.5-1.5x and 0.54-0.87 at 2x.

So the 2x result is not evidence that production extrapolates dose increases
well. It depends on how the training fit split the net shrinkage between
regrowth and drug, which the window does not determine.

## Open decisions

The following are not done and await a decision: the Neural-ODE ensembles on
this design (about 2-3 hours per seed), tutorial-style figures, and moving the
manuscript's tumour section and dose description to this design. If the
design is kept, the levers are the information in training: a longer window,
or a scan after a recovery period. The reported cases remain the manuscript's
chemo results until then.

Run notes, the pre-declared rule, logs, and plots
(`figures/reported_burden.png`, `figures/dev45_burden.png`,
`figures/why_2x_looks_best.png`) are local session
artifacts, not bundled with the repository:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-tutorial-runs/`.
