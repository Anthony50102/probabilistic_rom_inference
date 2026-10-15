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

**Follow-up: mid-course re-planning works.** The `midcourse-*-chemo` cases
train through day 84 (eight treated weeks) and forecast the last four doses
at 0.5-2x. Their selection rule was declared before the fits. Under it, the
chosen model forecasts the evaluation seeds within 3.7-5.8% (median per
dose), close to the basis floor. It has the projected PDE's form and an
input-aware Gaussian process. Its main remaining error is too little
regrowth when doses are lowered (day-119 burden 0.88-0.91 of the truth at
0.5x). See "Mid-course re-planning with the projected PDE's form".

An after-the-fact check finds that the model change, not the split, does
most of the work. On the earlier days 5-57 split the same model forecasts
within 3.5-11.4% (median per dose), against 11.1-35.1% before. This revises
the outcome above. The default smoother contributed to the failure by
rounding off the drop at each dose, and the projected form leaves less for
the 52-day window to determine. See "After-the-fact check: the chosen model
on the earlier split".

**The free ROM on realistic dose changes.** The protocol never raises the
weekly dose; it cuts it to 75% or 50% for nerve damage, or stops. The
reported free ROM was given only the input-aware Gaussian process and tested
on those futures under a rule declared before the fits. It passes the
18.9% bar everywhere:
- the planned course from day 57: 16.1% (median);
- cuts from day 57: 12.0-16.6%;
- cuts from day 84: 6.7-9.1%.

It forecasts too little regrowth after a cut, and its errors are 1.2-2.4
times the projected model's. See "Dose modifications with the free ROM".

Its draw band misses the truth after cuts to 0.5x (from day 84, after about
three weeks). The band does not include the cells the rank-4 basis cannot
draw as the tumour spreads, and every model on this basis loses them. A
backtest on the training scans predicts that loss. Adding it to the band was
tried after the fact; it is not part of any declared rule. See "Why the draw
band misses after cuts".

Smoothing the reported maps by 2 mm before the POD, also after the fact,
wins most of those cells back. With 5 modes, from day 84, the band then
holds the truth at 0.5x on seeds 48-50, and the forecast error falls. The
model then reproduces blurred fields inside the training window, unless
only the modes are smoothed. From day 57 it helps less, because the free
model's regrowth after a cut is too slow.

Blurring only the four production modes, with the reported maps kept as
data, lowers the forecast error at every arm from day 84. The bands are a
little wider, and the 0.5x band holds the truth 76-100% of the time. The
price is a blurred first scan: 14% error on day 5, against 2%. See "Can
preprocessing win the cells back?".

**Realistic scan schedules fail.** Every result above trains on 120 scans,
one every 0.4-0.7 days. On realistic schedules, from four-weekly to weekly
(3-12 scans), the free ROM learns little or no drug effect. It fails the
18.9% bar on every such schedule, from day 57 and from day 84; its worst
scenario arm is 32-76% off. 120 evenly spaced scans pass, so the number of
scans matters, not the even spacing. After the fact, on one seed, it comes
back under the bar with scans about every 2 days (to day 84) or daily (to
day 57). See "Realistic scan schedules".

**Fixing sparse scans: both declared fixes fail.** Two fixes kept the scan
count at weekly or below. One told the Gaussian process the scanner's known
noise; the other scanned every 8 days, so the scans fall at different points
of the dose cycle. Neither passes on either split, nor do both together or
two modes instead of three or four. Together they do worst: 33-108% off.
The dense control still passes. After the fact, the cause is the Gaussian
process's trend: its prior is too tight for 8-12 scans, so the process
treats the dose cycle as noise or bends sharply between scans. See "Fixing
sparse scans".

**Learning the trend's prior scale: one realistic schedule passes.** The
trend's prior scale became a learned Gaussian-process setting, declared
before the fits. Scans every 8 days to day 84 (11 scans) then pass: 13.6,
12.8 and 17.8% at 1, 0.75 and 0.5x. Weekly scans to day 84 fail at 0.5x
only (21.1%), and both schedules to day 57 still fail (worst arm 25-32%).
The errors fall by 33-71%. The fits still learn about half
the growth and kill rates, so they regrow too little after a cut, and their
bands are 4-10 times as wide as with 120 scans. The dense control does not
change. See "Learning the Gaussian process's trend scale".

**Past patients: a population prior and partial pooling, two of four
schedules at best.** Both use a virtual cohort of eight past patients on
the same anatomy, scanned through their whole course, and both were
declared before their fits.
- A population prior (the target centred on one operator fitted to the
  cohort) passes weekly scans to day 57. It makes both every-8-days
  schedules much worse.
- Partial pooling gives every patient their own operator around a shared
  mean, all fitted together. It passes both weekly schedules, to day 84 for
  the first time (7.2, 8.4 and 10.8%), and its bands always hold the truth.
  It fails both every-8-days schedules (worst arm 21-22%).
- In both, the target takes the cohort's whole-tumour growth and kill. That
  suits it with the weekly cohort and not with the every-8-days cohort.

No version passes all four realistic schedules. See "A population prior
from past patients" and "Partial pooling of past patients".

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

## Mid-course re-planning with the projected PDE's form (pre-registered)

**Why.** The sections above show two problems. Days 5-57 cannot separate
regrowth from drug kill, and the free reduced model cannot speed up regrowth
as the drug thins the tumour. This follow-up changes two things. Both were
chosen from a noise-free least-squares screen before any production fit of
the new models.

1. **The question: a mid-course re-plan.** Training sees the first eight
   weekly doses: days 5-84, still 120 segmented scans at 1% noise. The
   forecast covers days 84-119, with the last four doses (days 91-112) given
   at 0.5, 1, 1.5 or 2x. Clinically, this asks whether to change the dose for
   the rest of the course once the response has been seen. The truth is the
   same TumorTwin run as above up to day 84, and the four futures branch from
   day 84.
2. **The model: the form the PDE projects to.** Take the twin's PDE,
   du/dt = div(d grad u) + k u (1 - u) - α(t) u, and project it onto an
   uncentred orthonormal POD basis u ≈ Vq. The result is
   dq/dt = A q - k T(q, q) - α(t) q, where T_ijk = Σ_x V_xi V_xj V_xk.
   - **No constant term:** with no tumour, nothing grows.
   - **No direct drug term:** B = 0 on an uncentred basis.
   - **Drug acts on each mode separately:** N is diagonal, because VᵀV = I.

   The structured fits impose these zeros and learn the rest:
   - A, freely.
   - One drug coefficient per mode, n_i. It would be -1 if the projection
     were exact.
   - In the logistic variant only, one crowding rate per mode, h_i, which
     multiplies the fixed shape T_i(q, q). It would be -k = -0.052 if the
     projection were exact.

   Every learned coefficient gets the same zero-centred prior as the free
   model's operator columns, so no true value enters the fit. Implementation:
   `core.weakform_opinf.structured_rows` and `build_model(row_maps=...)`, with
   the maps built and checked by `benchmark_models.galerkin_row_maps`.

| Case | Reduced model | Operators fitted |
|---|---|---|
| `midcourse-regimen-chemo` | Free, as reported (control) | `cAN` |
| `midcourse-galerkin-chemo` | Projected form, no crowding | `AN`: A free, N diagonal |
| `midcourse-logistic-chemo` | Projected form with crowding | `AHN`: A free, N diagonal, H row i = h_i T_i |

Everything else is unchanged:
- the twin, anatomy, schedule and doses;
- the segmented observations;
- the uncentred rank-4 POD, carried over from `alternative-regimen-chemo` and
  not reselected;
- the production settings: 12,000 updates, 500 draws, tempered dynamics rows.

**Screen behind the choice.** These numbers are local session artifacts, not
repository results.
- Noise-free least squares on the mid-course split, reduced-coordinate errors
  over days 84-119 across the four doses:
  - projected form with crowding: 0.6-1.6%;
  - projected form without crowding: 3.6-16.5%;
  - with the free constant term kept: 1-17%.
- With noise, a smoothing-spline stand-in for the Gaussian process biased the
  learned drug coefficient to about 0.8 of the truth, giving 5-21%.
- Production's free model, already run on development seed 45, gave these
  full-field errors over days 84-119: 29.3% at 0.5x, 16.7% at 1x, 10.8% at
  1.5x and 15.8% at 2x (mean 18.2%). The basis floor is 3.2-5.1%.

How the production Gaussian process handles the noise is the open question.

**Rule, declared before any development fit of the structured models.**
1. Run all three cases on development seeds 45-47 with the default settings.
2. **Selection.** Of the two structured variants, choose the one with the
   lower mean full-field error over days 84-119 (`forecast_84_119`, point
   forecast), averaged over the four doses and three seeds. The free model is
   reported alongside as the control and is not a candidate.
3. **Go/no-go.** Use the evaluation seeds only if the chosen variant's
   development mean is below the free control's. Otherwise stop and report the
   development results.
4. **At most one fix.** If the learned drug coefficients are clearly biased,
   as the spline stand-in's were, one fix may be tried: the input-aware
   Gaussian-process trend (`gp_input_trend`). It is tried on development seeds
   only, judged by the same metric, and recorded here before step 5.
5. **Evaluation.** Run evaluation seeds 48-50 once, for the chosen variant and
   the free control, and report them whatever they show.
6. **Success bar (for reporting, not selection).** At every dose, the median
   `forecast_84_119` error is no worse than the worst arm of the reported
   chemo tasks (18.9%).
7. **Neural ODE.** It stays gated until these results are reviewed. If it is
   run, it gets the same known drug term (a hybrid), so the comparison is like
   for like.

```bash
cd experiments/tumor
for c in midcourse-regimen-chemo midcourse-galerkin-chemo midcourse-logistic-chemo; do
  conda run -n prob_rom python 04_unified_benchmark.py $c --seeds 45 46 47      # development
done
conda run -n prob_rom python 04_unified_benchmark.py <chosen variant>           # evaluation 48-50, once
conda run -n prob_rom python 04_unified_benchmark.py midcourse-regimen-chemo    # evaluation 48-50, once
```

The four futures from day 84 are generated on first use (about 20 s each,
with names ending `_from84`). The cached exposure table now carries the
training and forecast spans in its name
(`inputs_<history>_train5-84_to119.npz`). The mid-course cases share the
alternative-regimen history but not its fit grid. The existing tables
regenerate bit-identically under the new names.

### Development results (seeds 45-47, default settings)

Point-forecast full-field error over days 84-119 (%). Each cell lists seeds
45 / 46 / 47 (median). All nine fits completed their 12,000 updates.

| Model | 0.5x | 1x | 1.5x | 2x | Mean | Training fit |
|---|---|---|---|---|---|---|
| free (control) | 29.3 / 36.3 / 36.6 (36.3) | 16.7 / 25.9 / 27.3 (25.9) | 10.8 / 18.9 / 20.8 (18.9) | 15.8 / 17.8 / 19.0 (17.8) | 22.93 | 7.8 |
| galerkin | 38.0 / 44.6 / 40.1 (40.1) | 28.9 / 36.3 / 33.0 (33.0) | 22.4 / 30.0 / 27.1 (27.1) | 21.2 / 27.4 / 23.8 (23.8) | 31.06 | 9.1 |
| logistic | 27.4 / 32.7 / 29.7 (29.7) | 16.8 / 23.1 / 21.9 (21.9) | 13.5 / 18.3 / 16.2 (16.2) | 20.0 / 21.3 / 14.0 (20.0) | 21.25 | 6.7 |

The basis floor is 5.1, 4.3, 3.6 and 3.2% at 0.5, 1, 1.5 and 2x. Every model
predicts the right direction of every dose change (294 of 294 times), but
only 0.29-0.55 of its size on day 119.

**Rule, steps 2-3.** Logistic is chosen (21.25% against 31.06% for
galerkin). It beats the free control (22.93%), so the evaluation seeds may be
used. The margin is small, and the success bar is missed at 0.5x, 1x and 2x.

**Learned coefficients: clearly biased.** The drug coefficients should be
about -1.
- Mode 1 learns -0.68 to -0.74 in every model; it carries most of the burden.
- Mode 2 learns -0.003 to -0.018 on seeds 45-46 in the structured models
  (-0.07 to -0.12 for free).
- Modes 3-4 learn -0.24 to -0.78.

The crowding rates, -0.023 to -0.082 against -0.052, have the right size.
Because the bias is clear, step 4 applies before any evaluation seed is used.

**Step 4, declared before these fits.** The fix is the input-aware Gaussian
process. It adds the cumulative drug exposure to each mode's Gaussian-process
mean, so the smoother can follow the drop after each dose instead of
flattening it. Smoothing those drops away is the mechanism behind the
spline stand-in's 0.8 bias.
- It is fitted on development seeds 45-47 for the chosen logistic model
  (`midcourse-logistic-trend-chemo`) and for the free control
  (`midcourse-regimen-trend-chemo`). Nothing else changes.
- It is adopted if it lowers logistic's development mean below 21.25%.
- Evaluation then compares logistic with the free control under the same
  smoothing. Go/no-go (step 3) is re-applied to that pair. If the fix is not
  adopted, the default pair is evaluated as above.

**Step 4 result (seeds 45-47).** Each cell lists seeds 45 / 46 / 47 (median).

| Model | 0.5x | 1x | 1.5x | 2x | Mean | Training fit |
|---|---|---|---|---|---|---|
| free + input-aware GP | 6.7 / 7.2 / 6.1 (6.7) | 9.8 / 9.4 / 8.8 (9.4) | 17.5 / 16.3 / 19.8 (17.5) | 24.7 / 23.2 / 30.0 (24.7) | 14.96 | 1.2 |
| logistic + input-aware GP | 6.2 / 9.1 / 7.7 (7.7) | 4.7 / 7.5 / 6.0 (6.0) | 4.0 / 6.5 / 5.1 (5.1) | 3.8 / 5.9 / 4.8 (4.8) | 5.95 | 1.5 |

- **Fix adopted.** Logistic's development mean falls from 21.25% to 5.95%,
  within 1.5-2.6 points of the basis floor at every dose.
- **Go/no-go passes.** 5.95% is below the free control's 14.96% under the
  same smoothing, so the evaluation seeds are run for
  `midcourse-logistic-trend-chemo` and `midcourse-regimen-trend-chemo`.
- **The bias is gone where the data inform it.**
  - Mode 1's drug coefficient is -0.99 to -1.00 (truth -1). Mode 2's is
    -0.94 to -1.00, and modes 3-4 are -0.53 to -0.76.
  - Mode 1's crowding rate is -0.046 to -0.058 (truth -0.052).
  - The predicted dose effect on day 119 is 0.81-0.89 of the truth, up from
    0.36-0.55. The predicted day-119 burden is 0.84-0.99 of the truth at
    every dose.
  - All 64 draws are within 25% at every dose and seed.
- **Free + input-aware GP** also learns mode 1's drug coefficient (-0.94 to
  -0.98) and is accurate at the lower doses. It also learns a large constant
  source (c_1 = 2.1-2.4) that adds tumour whatever its size; the projected
  form has none, since with no tumour nothing grows. When high doses shrink
  the tumour, that source dominates. The day-119 burden is over-predicted at
  1.3x the truth for 1.5x doses and 1.9x for 2x doses, and only 11-40 of 64
  draws are within 25% at 2x.

### Evaluation results (seeds 48-50, run once)

Point-forecast full-field error over days 84-119 (%). Each cell lists seeds
48 / 49 / 50 (median). All six fits completed their 12,000 updates (about 6
minutes each, single-threaded), and all 64 draws completed in every cell.

| Model | 0.5x | 1x | 1.5x | 2x | Mean | Training fit |
|---|---|---|---|---|---|---|
| **logistic + input-aware GP** | 6.7 / 5.8 / 5.7 (**5.8**) | 5.5 / 4.3 / 4.3 (**4.3**) | 4.8 / 3.7 / 3.8 (**3.8**) | 4.4 / 3.5 / 3.7 (**3.7**) | **4.68** | 1.0 |
| free + input-aware GP | 6.7 / 6.6 / 7.2 (6.7) | 9.1 / 9.2 / 8.7 (9.1) | 15.9 / 16.3 / 15.7 (15.9) | 22.6 / 22.9 / 22.4 (22.6) | 13.61 | 1.2 |
| Basis floor (median) | 4.9 | 4.1 | 3.5 | 3.0 | | |

- **Success bar met.** Every median is far below 18.9%. The chosen model is
  within 0.2-0.9 points of the basis floor, i.e. of the best any rank-4 model
  could do.
- **Dose changes.** The predicted change from the 1x arm on day 119 is
  0.86-0.92 of the true change (free + input-aware GP: 0.66-0.74). Its
  direction is right at every forecast time (294 of 294 per arm). The
  day-119 burden is 0.88-1.02 of the truth at every dose. Free + input-aware
  GP over-predicts the 2x burden at 1.82-1.90 times the truth.
- **Learned rates match the PDE.**
  - Mode 1's drug coefficient is -1.022, -0.995 and -1.008 (truth -1).
  - Mode 1's crowding rate is -0.046 to -0.049 (truth -0.052).
  - Modes 2-4 learn drug coefficients of -0.64 to -0.95 and crowding rates
    of -0.020 to -0.068.
- **Posterior draws.** For the chosen model, 63-64 of 64 draws are within
  25% at every dose and seed (free + input-aware GP: 40-51 at 2x). The draw
  band (5-95%) contains the true burden at 99-100% of forecast times at 1x,
  1.5x and 2x, and at 56-85% at 0.5x. These are empirical draw bands, not
  calibrated uncertainty.
- **Where the remaining error is.** The chosen model regrows the tumour too
  slowly when doses are lowered.
  - Its day-119 burden is 0.88-0.91 of the truth at 0.5x, 0.89-0.94 at 1x,
    and 0.93-1.02 at 2x.
  - At 0.5x the band contains the true burden at only 19-73% of days 100-119.
  - Free + input-aware GP has the same low-dose shortfall (0.87-0.90 at
    0.5x).

**Comparison with the earlier tutorial-schedule design.** That design
trained on days 5-57 with the free model and default smoothing, and its
median forecast errors were 11-35%. Three changes bring them to 3.7-5.8%:
- a mid-course question (eight treated weeks seen);
- the projected PDE's form, with no source, a diagonal drug term and the
  crowding shape;
- a smoother that knows when doses were given.

The development fits separate their effects:
- With default smoothing, the structure helps little (21.25% against
  22.93%).
- With the input-aware GP alone, the free model reaches 14.96%.
- Together they reach 5.95%.

The Neural ODE has not been run on this design.

```bash
cd experiments/tumor
for c in midcourse-logistic-trend-chemo midcourse-regimen-trend-chemo; do
  conda run -n prob_rom python 04_unified_benchmark.py $c --seeds 45 46 47      # step 4 (development)
  conda run -n prob_rom python 04_unified_benchmark.py $c                       # evaluation 48-50
done
```

### What the input-aware Gaussian process changes

The Gaussian process (GP) smooths each mode's noisy scans before the
operators are fitted.

Each dose makes the tumour's rate of change drop at once: the drug's kill
rate α(t) jumps at the dose and then decays. So each mode's trajectory has a
corner at every dose. The default GP, a squared-exponential kernel, is smooth
everywhere and rounds these corners off.

The input-aware GP adds three terms to each mode's GP: a constant, time, and
the cumulative training exposure ∫α. Their coefficients are zero-mean
Gaussian and are integrated out exactly, as for explicit basis functions in
Rasmussen & Williams (2006, §2.7). The derivative of ∫α is α(t), so the
smoothed curve can have a corner at each dose, and the size of each corner
is learned from the scans. The GP uses only the training exposure table,
which is the same α(t) the reduced model takes as its input. The forecast
does not use the GP.

On mid-course development seed 45 (training days 5-84;
`figures/smoother_seed45.png`), the two logistic fits compare with the
noise-free simulation projected onto the basis as follows:
- **Rate of change.** The default GP's is off by 54% (mode 1) and 64%
  (mode 2), against 17% and 18% for the input-aware GP.
- **State.** The default GP's errors are 1.3% and 3.6%, against 0.4% and
  0.5%.
- **Mode 2.** The default GP's length scale is 12 days, longer than the week
  between doses, so it smooths over mode 2's weekly dose pattern. Mode 2's drug
  coefficient then comes out near 0 (-0.0025). With the input-aware GP it is
  -0.96, and mode 1's moves from -0.71 to -0.99 (truth -1).

**Why it is physically justified.**
- The corners are in the PDE: the term -α(t)u changes the moment α jumps.
- The GP is told only when and how the drug was given, which the reduced
  model is already told.

**Limits.**
- It inherits the drug model (instant rise, decay at .7/day). With a
  different drug curve the corners would be wrong, but so would the reduced
  model.
- Each mode's corner size is one coefficient for the whole window, while the
  true size scales with the tumour. The squared-exponential part absorbs the
  difference.
- The Neural ODE fits trajectories by integrating them, so it needs no
  smoother.

### After-the-fact check: the chosen model on the earlier split

This check was not pre-registered. The mid-course follow-up changed the
split, the model form and the smoother together. To separate them,
`alternative-regimen-logistic-trend-chemo` fits the chosen model (projected
logistic form plus the input-aware GP, same settings) to the earlier design's
acquisitions. It trains on days 5-57 and forecasts days 57-119 through eight
changed doses. Seeds 45 and 48-50 were each run once, and nothing was tuned.
Each fit took 2.8 minutes, with the four seeds running at once.

Full-field error over days 57-119 (%). Each cell lists seeds 48 / 49 / 50
(median).

| Model | 0.5x | 1x | 1.5x | 2x | Training fit |
|---|---|---|---|---|---|
| Free, default smoothing (Results above) | 31.3 / 35.1 / 35.1 (35.1) | 24.3 / 25.2 / 24.6 (24.6) | 14.4 / 14.4 / 13.5 (14.4) | 9.6 / 12.2 / 11.1 (11.1) | 1.9-2.3 |
| **Logistic + input-aware GP** | 10.6 / 11.7 / 11.4 (**11.4**) | 6.6 / 7.5 / 6.8 (**6.8**) | 4.4 / 5.1 / 4.3 (**4.4**) | 3.5 / 3.7 / 3.2 (**3.5**) | 0.7-0.9 |
| Basis floor (median) | 9.4 | 5.8 | 3.7 | 2.6 | |

- **Development seed 45:** 12.5 / 7.1 / 4.7 / 3.6, against 29.1 / 22.0 /
  12.4 / 9.2 before.
- **Dose changes.** On day 119, the predicted difference from the 1x arm is
  0.81-0.88 of the true difference (before: 0.40-0.50). The day-119 burden is
  0.79-0.90 of the truth at every dose (before: 0.44-0.87).
- **Learned rates.** Mode 1's drug coefficient is -1.021 to -1.033 (truth
  -1), and its crowding rate is -0.044 to -0.048 (truth -0.052).
- **Posterior draws.** 63-64 of 64 draws are within 25% at every dose and
  seed (before: 9-52). The draw band contains the true burden:
  - at every forecast time on seed 50;
  - at 46-100% of forecast times on seed 48;
  - at only 29-47% on seed 49.
- **Same days, 84-119.** With this model, the mid-course split's medians are
  5.8 / 4.3 / 3.8 / 3.7, against 13.6 / 9.7 / 8.1 / 8.2 for the earlier
  split. That is what 27 more days of data and a shorter horizon should give.

**Conclusion.** The model change, not the split, accounts for most of the
improvement. With it, the earlier question (data to the second MRI, forecast
to the third) is within 0.7-2.0 points of its basis floor.
`figures/what_changed_seed48.png` shows the three steps for seed 48.

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py alternative-regimen-logistic-trend-chemo --seeds 45 48 49 50
```

## Dose modifications with the free ROM (rule declared before any fit)

**Why.** The manuscript reports the free ROM. On this twin it fails mainly at
1.5x and 2x, where the tumour shrinks far below the sizes seen in training.
There its learned constant source (c_1 about 2.1-2.4) dominates. Those arms
are not clinical options: the weekly paclitaxel dose is not raised (toxicity,
mainly neuropathy, limits it). The protocol's response to neuropathy is to
hold the drug, then:
- resume at 75% (grade 2);
- resume at 50% (grade 3);
- stop if it does not resolve.

Source: University Hospital Southampton breast paclitaxel protocol, "Dose
Modifications"; checked 2026-10-06. With the input-aware GP, the free ROM's
mid-course errors were 6.7% at 0.5x and 9.1% at 1x (evaluation seeds 48-50,
above). This check asks whether the free ROM, changed only by that smoother,
forecasts these realistic futures.

**Cases.** All three use:
- the free ROM (`cAN`, as reported);
- the input-aware GP;
- the uncentred rank-4 POD, carried over;
- the production settings. Nothing else changes.

| Case | Training | Forecast | Future doses | Seeds |
|---|---|---|---|---|
| `standard-regimen-trend-chemo` | 5-57 | 57-119 | 1x (as planned) | 51-53 |
| `reduced-regimen-trend-chemo` | 5-57 | 57-119 | 1, 0.75, 0.5x; stress 0.25, 0x | 45, 48-50 |
| `midcourse-reduced-trend-chemo` | 5-84 | 84-119 | 1, 0.75, 0.5x; stress 0.25, 0x | 45, 48-50 |

**Rule.**
1. **Runs.** Each seed is run once, and nothing is tuned.
2. **Metric.** The point-forecast full-field error over the forecast window,
   as a median over seeds 48-50 (51-53 for the standard case). Seed 45 is
   reported separately.
3. **Arms.** The scenario arms are 1, 0.75 and 0.5x. The stress arms are
   0.25x and 0x (stopped): there the tumour regrows past the sizes seen in
   training. They are reported but not judged.
4. **Bar.** The free ROM works on a split if every scenario arm's median is
   at most 18.9%, the worst arm of the reported chemo tasks. Also reported:
   - the basis floor;
   - the day-119 burden as a fraction of the truth;
   - the projected logistic model, where it was run.
5. **Overlap.** This model has already been run at 0.5x and 1x on the
   mid-course split (seeds 45-50, above), and those arms should reproduce.
   Only 0.75x, 0.25x and 0x are new there. It has not been run on the 5-57
   split or on seeds 51-53.

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py reduced-regimen-trend-chemo --seeds 45 48 49 50
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo --seeds 45 48 49 50
conda run -n prob_rom python 04_unified_benchmark.py standard-regimen-trend-chemo                  # 51-53
```

### Results (each seed run once)

All 11 fits completed their 12,000 updates, and all 64 draws completed in
every arm. With seven fits running at once, the 5-57 fits took about 13.5
minutes each and the standard fits 7.5. The mid-course fits took 12.7
minutes, with four at once. The mid-course 0.5x and 1x arms reproduce the
earlier `midcourse-regimen-trend-chemo` fits exactly, with identical
operators and scores.

Full-field error over the forecast window (%). Each cell lists seeds 48 / 49
/ 50 (median).

| Training | 1x | 0.75x | 0.5x | 0.25x (stress) | 0x (stress) |
|---|---|---|---|---|---|
| Days 5-57 | 16.9 / 15.4 / 16.6 (**16.6**) | 12.3 / 11.4 / 12.0 (**12.0**) | 15.1 / 12.8 / 13.9 (**13.9**) | 18.1 / 14.3 / 16.6 (16.6) | 20.2 / 18.6 / 20.4 (20.2) |
| Basis floor | 5.8 | 7.4 | 9.4 | 11.9 | 15.0 |
| Days 5-84 | 9.1 / 9.2 / 8.7 (**9.1**) | 7.0 / 6.9 / 6.9 (**6.9**) | 6.7 / 6.6 / 7.2 (**6.7**) | 7.3 / 7.5 / 8.8 (7.5) | 8.2 / 8.5 / 10.5 (8.5) |
| Basis floor | 4.1 | 4.5 | 4.9 | 5.4 | 6.0 |

- **Standard regimen** (days 5-57, 1x as planned, seeds 51 / 52 / 53): 16.1
  / 15.9 / 16.3 (**16.1**). The basis floor is 5.7. With default smoothing
  the same model gave 29.1 / 16.4 / 32.6 (29.1).
- **Development seed 45** (1, 0.75, 0.5, 0.25, 0x):
  - days 5-57: 17.2 / 10.6 / 12.6 / 14.0 / 16.7;
  - days 5-84: 9.8 / 7.2 / 6.7 / 7.6 / 8.6.

**Verdict under the rule: the free ROM works on all three.** Every scenario
arm's median is at most 18.9%:
- days 5-57: 16.6, 12.0 and 13.9%;
- days 5-84: 9.1, 6.9 and 6.7%;
- standard regimen: 16.1%.

**How it compares.**
- **Default smoothing, same model.** On days 5-57 it gave 24.6% at 1x and
  35.1% at 0.5x. The smoother alone turns a fail into a pass.
- **Projected logistic + input-aware GP.** It gave 6.8% and 11.4% (days
  5-57) and 4.3% and 5.8% (days 5-84) at 1x and 0.5x. The free ROM's
  errors are 1.2-2.4 times as large.

**Where the error is.**
- **Too little regrowth after a dose cut.**
  - The predicted day-119 change from the 1x arm is 0.47-0.60 of the true
    change from day 57 at 0.75x and 0.5x, and 0.67-0.74 from day 84.
  - The day-119 burden at those arms is 0.70-0.84 of the truth from day 57,
    and 0.87-0.95 from day 84.
  - The learned constant source (c_1 = 2.0-2.4 on seeds 48-53, A_11 about
    0) adds a fixed number of cells per day. In the PDE, regrowth is
    proportional to the tumour. With the drug stopped, the predicted burden
    rises along a straight line while the truth curves upward.
- **The draw band misses after cuts from day 57.** It contains the true
  burden at 100% of forecast times at 1x, but at only 42-64% at 0.75x and
  29-36% at 0.5x. From day 84 it contains it at 100% at 1x and 0.75x, and at
  61-66% at 0.5x. All 64 draws are within 25% at every scenario arm. See
  "Why the draw band misses after cuts" below.
- **At 1x the error grows with the horizon.**
  - For the standard regimen the error is 6.1-6.6% over days 57-88 and
    24.8-25.5% over days 88-119.
  - The day-119 burden is 0.99-1.05 of the truth, but partly by coincidence.
    On seed 51 the rank-4 basis holds 9% fewer cells than the truth on day
    119 (8,846 against 9,753). The free ROM's mode 1 is 36% too large and
    adds back about as many cells, in the wrong places.
  - From day 84 the error is 4.6-5.1% over days 84-101.5 and 12.0-12.5% over
    days 101.5-119.
- **Learned drug coefficient.** Mode 1's is -0.96 to -0.99 on every split,
  close to the projected form's -1.

**What this means.**
- The reported free ROM can forecast the protocol's realistic dose changes
  (continue, 75% or 50%). The only change is the smoother, an implementation
  detail.
- Its limits must be disclosed:
  - it forecasts too little regrowth after dose cuts;
  - it over-predicts when the tumour shrinks far below the training sizes
    (1.5x and 2x, above).
- The choice of reduction-only futures was informed by that failure. The
  clinical reason, the protocol, is independent of it.
- The projected form remains more accurate.

`figures/free_dose_mod_burden.png` shows the burden forecasts for seed 48
from days 57 and 84, and for the standard regimen.

### Why the draw band misses after cuts (after the fact, no new fits)

This was checked after the 0.5x band was seen to miss. Nothing in this
subsection was declared beforehand.

- **What the band covers.** The 5-95% band of the 64 posterior draws covers
  uncertainty in the reduced model's coefficients on this basis. It cannot
  include cells that the rank-4 basis cannot draw. It also cannot include a
  growth law that is wrong outside the training data.
- **The basis loses cells as the tumour spreads.** The best rank-4 answer is
  the burden of the truth projected on the basis. Its shortfall from the true
  burden, on seeds 48-50:
  - from day 84: 1.4% on day 84, and on day 119 5.2-5.3% (1x), 5.8-5.9%
    (0.5x) and 6.6-6.7% (0x);
  - from day 57: 0.7% on day 57, and on day 119 9.4-9.5% (1x), 12.6-12.7%
    (0.5x) and 18.4-18.6% (0x).
  - Every model on this basis shares it. That includes the Neural ODE, which
    is trained in the same reduced coordinates.
- **More modes do not help.** On seed 48 the fifth mode is 1.04 times the
  scan-noise threshold from day 84 and 0.87 times it from day 57. Ranks 5
  and 6 cut the day-119 shortfall at 0.5x only from 5.8% to 5.1% and 5.0%
  (from day 84), and from 12.6% to 12.3% (from day 57). All 120 modes give
  5.0% and 12.2%.
- **Where the lost cells are.** Every mode is a blend of training scans, so
  it is zero wherever no training scan showed tumour. On seed 48 the modes'
  support is exactly the union of the segmented lesions. The shortfall splits
  into two parts (seeds 48-50, 0.5x, day 119):
  - cells outside that region, where no mix of the modes can put any: 2.9%
    from day 84 and 7.2-7.4% from day 57;
  - a form inside the region that no mix of the modes matches: 2.9% from
    day 84 and 5.3-5.4% from day 57.
  - On the last training day the parts are 0.7 + 0.7% (day 84) and
    0.3-0.4 + 0.3% (day 57). From day 84 the two parts are equal at every
    arm. From day 57 the outside part is 57-60% of the shortfall.
  - On seed 48 the cells outside lie close to the region: median 1.7 mm and
    99% within 7.1 mm of it from day 84; 2.0 mm and 7.9 mm from day 57.
- **The limit is the scans, not POD itself.** The same 120 training scan
  times were decomposed three ways (seed 48, 0.5x, day 119, shortfall of the
  best mix):
  - our scans (noisy, zero outside the detected outline): 5.8% at rank 4 and
    5.0% at any rank from 6 to 120 (from day 84);
  - the same scans without noise, still cut at the outline: 3.5% at rank 120
    from day 84 and 8.6% from day 57. None of the cells beyond the outline
    come back at any rank;
  - the same scans without noise or cutting (a perfect scanner): 4.5% at
    rank 4, 1.5% at rank 6 and 0.2% at rank 10 from day 84. Rank 10 puts
    back 92% of the cells beyond the outline. From day 57 rank 10 gives 2.5%
    and puts back 63%.
  - The earlier fields have a faint edge below the detection threshold. A mix
    of them can push the edge outwards. The patterns that do this are weak:
    the fifth is 0.7 times the scan-noise threshold and the tenth 0.009
    times (day 84). With this noise only four or five patterns can be read
    from the scans. The Neural ODE in the same coordinates has the same limit.
- **From day 84 the miss is all basis loss.** At 0.75x and 0.5x the band
  contains the best rank-4 answer at 100% of forecast times. The truth leaves
  the 0.5x band on day 105-107. At 1x the forecast lies above the best rank-4
  answer and close to the truth, so the two errors cancel. The band contains
  the best rank-4 answer there at 48-100% of times.
- **From day 57 the growth law adds to it.** At 0.5x the day-119 forecast is
  0.70-0.75 of the true burden, against 0.87 for the best rank-4 answer. The
  band contains the best rank-4 answer at 53-88% of times.
- **Projected logistic + input-aware GP.** Its band contains the best rank-4
  answer at 100% of times on both splits. Against the truth at 0.5x it
  reaches 56-85% from day 84 and 33-100% from day 57, from basis loss alone.
- **The reported multi-dose bands.** They contain the truth at 100% of days
  70-110, but they are 3-4 times as wide: 0.38-0.52 of the burden on day 110,
  against 0.10-0.18 from day 84 here. The Neural-ODE bands there contain it
  at 1-80%.

**Can the training scans predict the basis loss? (feasibility, no fits)**
- **Backtest.** The rank-4 POD is fitted to the noisy training scans taken up
  to 21 days before the last one. It is then used to measure the cells missed
  in the later scans. The pipeline's scans were regenerated exactly, with
  singular values matching to 1e-12.
- **Rates from day 84.** The estimate is 0.09-0.10% per day (seeds 45 and
  48-50). The true rate is 0.10-0.11% per day at 1x and 0.13% at 0.5x.
- **Rates from day 57.** The estimate is 0.14% per day. The true rate is
  0.14-0.15% per day at 1x and 0.19-0.20% at 0.5x.
- **Sensitivity.** Holding out 14 or 28 days gives 0.08-0.11 from day 84,
  but 0.09-0.21 from day 57.
- **What the estimate misses.** It starts at 0.2-0.4%, so it misses most of
  the 1.4% already lost on day 84. About half of that gap is cells the scans
  cannot see, below their detection threshold: on day 84, 0.7 of the 1.4
  points lie outside every segmented lesion.
- **Effect on the band.** The upper edge was divided by
  1 - (a + r (t - T)), with a and r the backtest's intercept and rate. Time
  inside the band, against the truth (seeds 48 / 49 / 50):
  - from day 84: 100% at 1x and 0.75x, and 92 / 82 / 89% at 0.5x (was 64 /
    61 / 66). The stress arms reach 73 / 63 / 63% at 0.25x and 82 / 59 / 56%
    at 0x. At 0.5x the day-119 width rises from 0.13 to 0.17 of the truth.
  - from day 57: 96 / 90 / 100% at 0.75x, and 48 / 60 / 65% at 0.5x.
- **Shifting both edges** instead drops 1x to 59-100%.
- **Status.** The rule was chosen after seeing these seeds. Using it would
  need declaring it first and checking it on fresh seeds.

Scripts: `band_coverage.py`, `shape_loss_backtest.py`, `allowance_coverage.py`,
`rank5_check.py`, `four_shapes.py` and `pod_limits.py`. Figures:
`figures/band_miss_seed48.png`, `figures/four_shapes_seed48.png` (the four
modes on one slice, and where the lost cells are) and
`figures/pod_limits_seed48.png` (shortfall against rank for the three kinds
of scan, and each pattern's strength against the noise).

### Can preprocessing win the cells back? (after the fact)

This asks whether a different preprocessing or POD removes the basis loss
above. The smoothing width (2 mm) and the ranks were chosen after looking at
seed 48, and seeds 48-50 had been seen before. Nothing here was declared
beforehand.

- **Idea.** The modes are zero beyond the detected outline because every
  reported map is. Each reported map is smoothed with a 2 mm Gaussian
  (normalised inside the breast) before the POD. The maps, and so the modes,
  then taper a little past the outline. Averaging neighbouring voxels also
  lowers the noise, so more patterns clear the noise threshold. This is close
  to smoothed functional PCA (Silverman 1996). The threshold is carried over:
  Gavish-Donoho for our maps, times the ratio of the largest singular values
  of pure noise in the same lesion masks with and without the smoothing. It
  allows 6 modes from day 84 and 5 from day 57 on seeds 48-50; our maps allow
  5 and 4.
- **What the smoothing is.** It is a blur, not added noise. Each voxel
  becomes a Gaussian-weighted average of its neighbours (sigma 2 mm, i.e. 2
  voxels), counting breast voxels only, so the total number of cells is
  unchanged (within 0.1%). Nothing random is added. The voxel noise drops to
  about 5% of its level. The pipeline's own detection step already blurs each
  scan by 1 mm to find the outline, but it then reports the unblurred values.
- **Is it physical? Partly** (seed 48, data to day 84):
  - For. The twin spreads cells by uniform, direction-independent diffusion
    (TumorTwin `ReactionDiffusion3D`: one scalar d inside the breast mask).
    Diffusion alone turns a point into a Gaussian of width sqrt(2 D t). A 2 mm
    blur is therefore about 21 days of the twin's own diffusion
    (D = 0.0965 mm^2/day); over the 35-day forecast it is 2.6 mm. The zero
    beyond the outline is a detection artefact: on day 84, 0.85% of the true
    cells lie outside it. The blurred map puts 1.18% there and is closer to
    the truth (full-field error 4.8% against 5.6% for the reported map).
  - Against. The 2 mm was chosen after the fact. The blur guesses the margin
    by spreading the cells seen inside the outline; it does not measure it.
    It is wrong for a small, sharp tumour. On day 5 the edge comes from the
    segmented patient MRI. On a line through the tumour the blur lowers the
    peak (0.60 against 0.65) and pushes the edge outward. It puts 2.4% of the
    cells outside the outline (truth 0.03%), and the error is 20.5% against
    2.5%. Hence the cost inside the training window below. It also matches
    the twin's even spread by construction; real tissue is not uniform, so a
    real tumour's margin need not look like a blur.
  - A defensible version would fix the width by a stated rule before any fit,
    for example the diffusion length over the forecast from a literature D
    (not the twin's), and keep the data sharp ("smooth modes" or "spread"
    modes below).
- **Basis loss.** Shortfall of the best mix of the modes from the true
  burden, at 0.5x on day 119, and full-field error of the best mix over the
  training window. Seeds 48-50 agree within 0.3 points.

  | Basis | Shortfall, from day 84 | Shortfall, from day 57 | Training window, from day 84 |
  |---|---|---|---|
  | reported maps, rank 4 (as run) | 5.8-5.9% | 12.6-12.7% | 0.7% |
  | smoothed 2 mm, rank 4 | 2.5% | 6.0-6.1% | 5.6-5.7% |
  | smoothed 2 mm, rank 5 | 1.3% | 4.5-4.6% | 3.6-3.7% |
  | smoothed 2 mm, rank 6 | 0.9-1.0% | 3.5-3.8% | 3.0-3.2% |

  - Rank 5 puts back 59-60% of the cells beyond the outline from day 84 and
    46% from day 57.
  - Smooth modes cannot draw the sharp training fields as well, hence the
    last column.
  - Smoothing by 1 mm gives 2.4% at rank 6 (day 84). Smoothing only inside
    the outline gives nothing back.
- **Fits.** The unchanged production fit (free ROM, input-aware GP, 64 draws)
  was run on these bases for seeds 48-50, in two ways:
  - *smoothed maps as data*: the POD and the reduced observations both use
    the smoothed maps;
  - *smooth modes only*: the POD uses the smoothed maps, but the reduced
    observations are the reported maps projected on those modes, so the fit
    sees no blur.
  - Nothing in the repository changed. `blur_fit.py` swaps the map,
    threshold and POD functions for its own process only.
- **From day 84** (seeds 48 / 49 / 50). Full-field error over days 84-119,
  and the share of forecast times with the true burden inside the 5-95% band:

  | | reported, rank 4 | smoothed maps, rank 4 | smoothed maps, rank 5 | smooth modes, rank 5 |
  |---|---|---|---|---|
  | 1x error | 9.1 / 9.2 / 8.7% | 6.6 / 6.4 / 6.4% | 6.2 / 6.4 / 5.9% | 8.1 / 8.0 / 7.3% |
  | 1x in band | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 97 / 89% | 100 / 52 / 53% |
  | 0.75x error | 7.0 / 6.9 / 6.9% | 4.6 / 3.8 / 4.3% | 4.1 / 3.7 / 3.7% | 5.6 / 5.2 / 4.6% |
  | 0.75x in band | 100% | 100% | 100% | 100% |
  | 0.5x error | 6.7 / 6.6 / 7.2% | 4.9 / 3.9 / 5.0% | 4.2 / 3.5 / 4.5% | 5.1 / 4.4 / 4.8% |
  | 0.5x in band | 64 / 61 / 66% | 86 / 85 / 76% | 100 / 100 / 100% | 100 / 100 / 100% |
  | 0.5x day 119, point / truth | 0.87-0.90 | 0.90-0.92 | 0.92-0.93 | 0.92-0.94 |
  | 0.5x day 119, band width / truth | 0.12-0.15 | 0.12-0.16 | 0.16-0.24 | 0.25-0.33 |
  | error inside the training window | 1.0-1.4% | 10.0-10.2% | 10.1-10.3% | 3.7-3.8% |

  - Each of these three beats the reported basis in forecast error at every
    arm, on every seed. With 5 smoothed modes the 0.5x band holds the truth
    throughout on all three seeds, partly because it is wider.
  - Smoothing alone, at the same rank 4, already helps.
  - The stress arms follow the same pattern. At 0x the error is 5.2 / 5.4 /
    8.0% with smoothed maps at rank 5 (reported: 8.2 / 8.5 / 10.5%).
- **Cost inside the training window.** With smoothed maps as data, the model
  reproduces a blurred tumour. The error is about 10% instead of about 1%.
  It is about 20% on day 5 and 4-6% on day 84 (seeds 48-50), because blur
  matters most for the small early tumour. Smooth modes keep the data sharp,
  and the error is 3.7-3.8%, which is the basis's own limit (3.6-3.7%
  above). The price is wider bands, and a 1x forecast 1.06-1.09 times the
  true burden on day 119.
- **Rank 6 is unstable.** On seed 48 it did well (0.5x: 4.1% with smoothed
  maps, 4.6% with smooth modes). On seeds 49 and 50 it overshot:
  - with smoothed maps, day-119 burden up to 1.4 times the truth and errors
    up to 31%;
  - with smooth modes, 1.3-2.1 times the truth and bands up to 18 times the
    burden.
  - Most of the error is in the first mode. The sixth pattern is only
    1.1-1.2 times the noise threshold. Why it upsets seeds 49 and 50 and not
    48 is not pinned down.
- **From day 57** (rank 5, seeds 48 / 49 / 50):

  | | reported, rank 4 | smoothed maps, rank 5 | smooth modes, rank 5 |
  |---|---|---|---|
  | 1x error | 16.9 / 15.4 / 16.6% | 13.5 / 12.6 / 13.2% | 17.8 / 15.8 / 17.1% |
  | 0.75x error | 12.3 / 11.4 / 12.0% | 10.0 / 8.5 / 9.4% | 12.8 / 9.6 / 11.4% |
  | 0.75x in band | 50 / 42 / 64% | 94 / 100 / 95% | 97 / 100 / 100% |
  | 0.5x error | 15.1 / 12.8 / 13.9% | 12.3 / 8.7 / 11.2% | 14.3 / 8.9 / 11.8% |
  | 0.5x in band | 29 / 30 / 36% | 48 / 50 / 43% | 38 / 100 / 100% |
  | 0.5x day 119, point / truth | 0.70-0.75 | 0.77-0.83 | 0.74-0.83 |
  | 0.5x day 119, band width / truth | 0.21-0.22 | 0.24-0.28 | 0.30-0.39 |
  | error inside the training window | 0.8% | 10.5% | 3.7% |

  - Smoothed maps are better at every arm, on every seed, but the 0.5x band
    still misses about half the time. From day 57 the free model's too-slow
    regrowth after a cut adds to the basis loss (above), and smoothing cannot
    fix that.
  - Smooth modes hold the truth more often than smoothed maps mainly because
    their bands are wider (0.52-0.77 of the burden at 1x on day 119). Their
    point forecast is worse than the reported basis at 1x on every seed, and
    at 0.75x and 0x on seed 48.
- **Sharp data, blurred modes, rank 4.** A third way: the four production
  modes are blurred by 2 mm, and the reported maps stay the data
  (`blur_fit.py` "blurred"). The rank rule is the production one.
  - Blurring the modes and taking the POD of blurred maps give the same
    subspace, to within 2.6 degrees at rank 4 and 6.3 degrees at rank 5
    (seeds 48-50, `blurred_modes_check.py`). "Smooth modes, rank 5" above is
    therefore the same idea with a fifth mode.
  - Blurring turns the weakest production direction by about 48 degrees
    (53 from day 57).

  From day 84 (seeds 48 / 49 / 50):

  | | reported, rank 4 | blurred, rank 4 | blurred, rank 5 |
  |---|---|---|---|
  | 1x error | 9.1 / 9.2 / 8.7% | 8.0 / 8.2 / 7.5% | 8.1 / 8.0 / 7.3% |
  | 1x in band | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 52 / 53% |
  | 1x day 119, point / truth | 1.03-1.04 | 1.04-1.06 | 1.06-1.09 |
  | 0.75x error | 7.0 / 6.9 / 6.9% | 5.7 / 5.6 / 5.6% | 5.6 / 5.2 / 4.6% |
  | 0.5x error | 6.7 / 6.6 / 7.2% | 5.6 / 5.1 / 6.3% | 5.1 / 4.4 / 4.8% |
  | 0.5x in band | 64 / 61 / 66% | 92 / 100 / 76% | 100 / 100 / 100% |
  | 0.5x day 119, point / truth | 0.87-0.90 | 0.88-0.92 | 0.92-0.94 |
  | 0.5x day 119, band width / truth | 0.12-0.15 | 0.14-0.15 | 0.25-0.33 |
  | 0.5x, best possible on the basis | 4.8-4.9% | 2.1-2.2% | 1.4% |
  | error inside the training window | 1.0-1.4% | 5.5-5.6% | 3.7-3.8% |

  - At rank 4 the forecast error is lower than the reported basis at every
    arm, including the stress arms, on every seed. The 0.75x band holds the
    truth throughout. The 0.5x band holds it more often (median 92%, lowest
    76%). The bands are a little wider: on day 119, 0.14-0.19 of the burden
    from 1x to 0.5x, against 0.10-0.18. Full dose stays inside its band.
  - The fifth mode gives full coverage at 0.5x, but the bands are about
    twice as wide, and the 1x forecast leaves its band on seeds 49 and 50.
  - The basis is no longer the main limit. At 0.5x the best possible error
    falls from about 4.9% to 2.2%, but the forecast error only falls from
    6.7% to 5.6%. The rest is the model's too-slow regrowth after the cut:
    on day 119 the 0.5x burden is still 0.88-0.92 of the truth, while the
    basis alone would lose only 2.6% of it.
  - Cost: the model draws a slightly blurred tumour. Inside the training
    window the error (rank 4, seeds 48-50) is:
    - 14% on day 5 (reported: 2%);
    - 6.5% on day 20;
    - 2-4% from day 40;
    - 1.5-2.0% on day 84, better than the reported basis (2.3-2.6%).

    Blurred modes suit the late, diffuse tumour, not the small, sharp first
    scan.
  - From day 57, at rank 4, the forecast is better at 0.5x and below on
    every seed (0.5x: 14.6 / 8.9 / 10.9%, against 15.1 / 12.8 / 13.9%).
    - The 0.5x band holds the truth 38 / 100 / 57% of the time (reported:
      29 / 30 / 36%).
    - At 0.75x it is better on seeds 49 and 50 but worse on seed 48.
    - At 1x it is no better (19.1 / 15.5 / 16.8%, against 16.9 / 15.4 /
      16.6%).
    - The best possible error at 0.5x falls from 9.4% to 4.1%, so the
      regrowth law dominates.
- **Other options (basis loss only, not fitted).**
  - Unsegmented maps inside a 6 mm margin around the lesion, then smoothed:
    0.3-0.4% at rank 6 from day 84, 3.1-3.3% from day 57. This needs maps
    reported beyond the outline. It changes the measurement, not the
    processing.
  - The four modes plus copies of them smoothed by 3 mm ("spread" modes,
    rank 8): 0.2% from day 84 and 1.1% from day 57, and the training window
    stays at 0.5-0.6%. The width was picked after the fact, and 4 mm
    overshoots. An eight-mode model was not fitted. Rank 6 above suggests
    weak extra modes can make the model unstable.
  - More modes of the reported maps: 5.0% at any rank from 6 (above).
- **Status.** Exploratory. To use it: declare the smoothing width, the
  variant and the rank rule first (for example, chosen on development seed
  45 as the original basis was), then test on fresh seeds and report the
  error inside the training window next to the forecast. The simplest
  candidate is the four production modes blurred: it keeps the production
  rank rule and adds a single setting, the width.

Scripts:
- `pod_variants.py` (basis loss for any seed);
- `pod_fix_figure.py`;
- `blur_fit.py` (fit runner);
- `blur_fit_summary.py` (tables and figure);
- `smoothing_explainer.py` (what the blur does to one scan);
- `blurred_modes_check.py` (blurred modes against the POD of blurred maps,
  and basis loss);
- `blurred_modes_summary.py` (tables and figure).

Figures:
- `figures/pod_fix_seed48.png` (the idea on one slice, and loss against
  rank);
- `figures/blur_fit_burden.png` (burden forecasts and bands, seeds 48-50);
- `figures/smoothing_profile_seed48.png` (a line through the tumour edge on
  days 84 and 5, with a zoom on the edge);
- `figures/blurred_modes_burden.png` (sharp data and blurred modes against
  the reported basis, seeds 48-50, from days 84 and 57).

## Realistic scan schedules (rule declared before any fit)

**Why.** Every case above trains on 120 scans at random times, far more than
any patient gets. This patient's record (`TNBC_demo_001.json`) has three
MRIs: days 0, 57 and 119. That is before treatment, three weeks into weekly
paclitaxel, and after the twelfth dose. It is the schedule of the I-SPY 2
imaging trial: "before treatment, at early treatment (3 weeks), at
midtreatment (12 weeks), and after treatment" (Partridge et al., Radiology
2018;289:618-627, ACRIN 6698). Before a decision on day 57 or day 84, a
real patient has two MRIs. A research protocol could scan more often, at
most weekly: the patient is in clinic weekly from day 35 for the infusions.

**Design.** The `*-scans` cases (`benchmark_cases.SCAN_SCHEDULE_CASES`)
repeat `midcourse-reduced-trend-chemo` (data to day 84) and
`reduced-regimen-trend-chemo` (data to day 57). Everything is kept: the twin,
the doses, the arms, the free ROM with the input-aware Gaussian process, the
1% noise rule and the segmentation. Only the scan days change. They are
fixed and evenly spaced, with the first and last scan at the window ends
(`regular_scan_days`: the whole number of intervals closest to the target
spacing, days rounded to whole days). The days are the same for every seed,
so a seed now changes only the scanner noise.

| Schedule | Data to day 84: scan days | Rank | Data to day 57: scan days | Rank |
|---|---|---|---|---|
| Random (reported) | 120 random | 4 | 120 random | 4 |
| Regular (control) | 120, every 0.66 days | 4 | not run | |
| Weekly | 5, 12, 19, 27, 34, 41, 48, 55, 62, 70, 77, 84 (12) | 4 | 5, 12, 20, 27, 35, 42, 50, 57 (8) | 3 |
| Fortnightly | 5, 18, 31, 45, 58, 71, 84 (7) | 4 | 5, 18, 31, 44, 57 (5) | 3 |
| Four-weekly | 5, 31, 58, 84 (4) | 3 | 5, 31, 57 (3) | 3 |

The regular control separates even spacing from fewer scans. The
four-weekly schedule to day 57 is the trial's two MRIs (the first moved from
day 0 to day 5, the window start) plus one on day 31.

**Two scans cannot be run.** A mechanical smoke run (300 updates, scores not
used) stopped at the operator-layout step, which takes second-order
differences and so needs three scans. With that step bypassed for the test,
it stopped again at the Gaussian process's lengthscale prior. That prior has
its median at the scan spacing and its 99th percentile at the window length;
with two scans they coincide, and the prior has zero width. Both could be
patched, but the data could not support the model anyway. Two scans give 2
numbers per mode, against 5 operator entries per mode at rank 2. Three scans
are the fewest the method runs on.

**Rank rule.** The rank stays 4 unless fewer modes clear the scan-noise
threshold on every acquisition (45 and 48-50). This was checked on the noisy
scans alone, before any fit (`scan_schedule_check.py`). Fewer scans mean
smaller singular values against the same noise. To day 84 the fourth mode is
1.19 times the threshold weekly, 1.05 fortnightly and 0.87 four-weekly. To day
57 it is 0.95 weekly. The noise rule takes 1% of the range of the noise-free
fields on the scanned days. That gives a noise level of 0.0086-0.0087 for
weekly to four-weekly, against 0.0088 for 120 scans.

**Rule.**
1. **Runs.** Seeds 48-50, each run once. Nothing is tuned.
2. **Metric and arms.** As in "Dose modifications with the free ROM": the
   point-forecast full-field error over the forecast window (median over
   seeds). Scenario arms are 1, 0.75 and 0.5x; 0.25x and 0x are stress arms,
   reported and not judged.
3. **Bar.** A schedule works on a split if every scenario arm's median is at
   most 18.9%.
4. **Also reported.** The basis floor, the training-window error, the
   day-119 burden as a fraction of the truth, and how often the 5-95% draw
   band holds the true burden at 0.5x and 1x.

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans   # 48-50
# likewise -fortnightly-scans, -4weekly-scans, -regular-scans,
# and reduced-regimen-trend-chemo-{weekly,fortnightly,4weekly}-scans
```

### Results (each seed run once)

All 21 fits completed their 12,000 updates, and all 64 draws completed in
every arm. With six fits at once they took 13-16 minutes each; the 120-scan
control took 18-19 minutes alongside other jobs.

**Verdict under the rule: no realistic schedule works.** Every one fails the
18.9% bar, on both splits, most by a wide margin. The 120 evenly spaced
scans pass, and do slightly better than 120 random ones. Even spacing is
not the problem; the number of scans is.

Full-field error over the forecast window (%), median of seeds 48-50. The
seeds now differ only in scanner noise and agree within 1.3 points on
these arms.
"Training" is the point forecast's error over the training window. The band
columns are for 0.5x: how often the 5-95% draw band holds the true burden,
and its width on day 119 as a fraction of the true burden.

| Data to | Schedule (scans, rank) | 1x | 0.75x | 0.5x | Training | Band holds truth | Band width | Verdict |
|---|---|---|---|---|---|---|---|---|
| Day 84 | 120 random (reported; r4) | 9.1 | 6.9 | 6.7 | 1.2 | 61-66% | 0.12-0.15 | pass |
| | 120 evenly spaced (control; r4) | 8.9 | 6.3 | 6.1 | 1.2 | 59-82% | 0.11-0.14 | **pass** |
| | Weekly (12; r4) | 22.1 | 24.1 | 32.4 | 18.2 | 100% | 1.3 | **fail** |
| | Fortnightly (7; r4) | 70.9 | 49.9 | 33.7 | 29.0 | 100% | 3.1 | **fail** |
| | Four-weekly (4; r3) | 68.9 | 48.3 | 32.3 | 28.0 | 100% | 2.9-3.2 | **fail** |
| Day 57 | 120 random (reported; r4) | 16.6 | 12.0 | 13.9 | 0.8 | 29-36% | 0.21-0.22 | pass |
| | Weekly (8; r3) | 42.4 | 49.1 | 53.9 | 3.2 | 13% | 0.78 | **fail** |
| | Fortnightly (5; r3) | 63.7 | 31.1 | 18.0 | 24.3 | 100% | 1.8 | **fail** |
| | Four-weekly (3; r3) | 75.5 | 40.6 | 19.9 | 22.3 | 100% | 2.0-2.2 | **fail** |

**What goes wrong.**
- **The fits learn almost no drug effect.**
  - Mode 1's drug coefficient is -0.02 to -0.15 from day 84 and -0.05 to
    -0.66 from day 57. With 120 scans it is -0.95 to -0.99, close to the
    projected form's -1. The growth source c_1 falls from 2.0-2.4 to -0.02
    to 0.29.
  - The forecasts barely respond to the dose. At 0.5x they reproduce 0-7%
    of the true day-119 change from the 1x arm from day 84, and 7-25% from
    day 57. With 120 scans: 68-74% and 50-60%.
- **They do not follow their own scans.**
  - The training-window error is 18-29% (3.2% for weekly scans to day 57),
    against about 1% with 120 scans.
  - The Gaussian process smooths over 9-48 days, against 2-8 with 120 scans.
  - Its estimate of the scanner noise in mode 1 is 6-11 times as large from
    day 84, and 13-20 times from day 57 (same seed, 120 random scans).
  - So the fit settles on small operators and puts most of the change between
    scans down to noise.
- **The bands are honest but useless, except weekly to day 57, where they
  are wrong.**
  - From day 84, the 0.5x band is 1.3-3.2 times the tumour wide. It holds
    the truth because it holds almost everything.
  - With weekly scans to day 57 the fit tracks the scans (3.2%). But it
    forecasts the tumour falling to 18-23% of the true day-119 burden, and
    the band holds the truth only 13-24% of the time across the scenario
    arms.
- **Weekly scans miss each dose's effect.**
  - Each weekly dose kills 16-19% of the tumour, which bottoms out 2.4-3.0
    days later. It then regrows 7-13% before the next dose (1x truth, doses
    on days 35-77).
  - To day 84, every treated weekly scan falls either 6 days after a dose
    or on a dose day, before the drug acts. Both are near the top of the
    sawtooth.
  - So the scans show a smooth decline, and nothing in them separates the
    drug from slower growth.
  - Fortnightly and four-weekly scans land at varied points of the cycle,
    but leave only 4 and 2 treated scans to day 84.
- **Too few numbers for the unknowns.** This is a rough count.
  - The free model has 36 unknown operator entries at rank 4, and 21 at
    rank 3.
  - The first scan sets the starting state. The rest give 44 numbers
    weekly, 24 fortnightly and 9 four-weekly (to day 84), and 21, 12 and 6
    (to day 57). 120 scans give 476.
  - The Gaussian process adds smoothness between scans, not information,
    and the true tumour is not smooth at the scale of a week.
  - More numbers than unknowns does not guarantee a good fit. Weekly scans
    to day 84 give 44 for 36 and still fail, because they all sit near the
    top of the dose cycle.
- **The lower rank is not the main cause.** The basis floors rise, to
  5.6-8.2% from day 84 and 9.1-14.2% from day 57. Every scenario-arm floor
  is still under the 18.9% bar, so a better fit on the same basis could
  pass.

Because the scan days are fixed, the seeds no longer test sensitivity to
scan timing. Moving the scans a day or two relative to the doses could
change these numbers; that was not tested.

### How often would we have to scan? (after the fact)

This was not declared. Session `scan_sweep_fit.py` registered the same cases
with scans about every 3.5, 2 and 1 days, for this process only. It built
them exactly as the declared ones and ran seed 48 only. Rank 4 cleared the
noise threshold in every case. Nothing in the repository changed.

| Data to | Scan every | Scans | 1x | 0.75x | 0.5x | 0.5x band holds truth |
|---|---|---|---|---|---|---|
| Day 84 | 7 days (declared) | 12 | 22.3 | 24.1 | 32.4 | 100% |
| | 3.5 days | 24 | 9.2 | 12.2 | 19.3 | 36% |
| | 2 days | 41 | 9.2 | 7.4 | 9.3 | 70% |
| | 1 day | 80 | 9.2 | 7.0 | 7.7 | 46% |
| | 0.66 days (declared control) | 120 | 8.9 | 6.4 | 6.1 | 59% |
| Day 57 | 7 days (declared) | 8 | 42.5 | 49.2 | 53.9 | 13% |
| | 3.5 days | 16 | 30.5 | 36.7 | 42.8 | 16% |
| | 2 days | 27 | 19.2 | 15.8 | 18.8 | 18% |
| | 1 day | 53 | 15.2 | 14.3 | 17.4 | 22% |
| | 120 random (reported) | 120 | 16.9 | 12.3 | 15.1 | 29% |

Seed 48 throughout.
- **Day 84.** The 1x forecast recovers at about twice-weekly scans. Lower
  doses need scans about every 2 days.
- **Day 57.** It takes daily scans to come in under the bar (2 days misses
  by 0.3 points at 1x).
- **At 0.5x the band still misses part of the time on every dense
  schedule.** It holds the truth 16-70% of the time, as with 120 scans. The
  basis and regrowth limits in "Why the draw band misses after cuts" apply
  here too.

**What this means.**
- The free ROM with the input-aware GP needs dense imaging on this twin:
  about every 2 days to day 84, and daily to day 57 (after the fact, one
  seed).
- Clinical MRI is far too sparse. On the trial's schedule a patient has two
  MRIs before a decision on day 57 or 84, and a research protocol could scan
  at most weekly.
- Every chemo result in this record assumes about 120 scans in the training
  window, so a scan every 0.4-0.7 days.
- Any model that learns all 36 entries from one patient's scans faces the
  same count, and the Neural ODE learns more. Clinical schedules would need
  fewer unknowns, for example the projected form, or prior information from
  other patients. Neither has been tested on sparse scans.

`figures/scan_schedule_burden.png` shows the burden forecasts and the scans
for seed 48 on every declared schedule. `figures/scan_schedule_error.png`
plots error against the number of scans, and `figures/scan_sweep_burden.png`
shows the after-the-fact schedules. The tables come from
`scan_schedule_summary.py` (log `logs/scan_schedule_summary.txt`).

## Fixing sparse scans (rule declared before any fit)

**Why.** On realistic schedules the free ROM learns almost no drug effect
("Realistic scan schedules"). Two of the causes found there can be fixed
without more scans:
- **The Gaussian process is told the wrong noise level.**
  - Each mode's noise prior is centred on 1% of that mode's variance
    across the scans.
  - For weekly scans to day 84 that is 42,600, 17,300, 1,150 and 159 times
    the voxel noise variance for modes 1-4. The fits then raise mode 1's
    estimate further, to about 290,000 times.
  - But the pipeline already knows the scanner noise. The noise rule sets
    it (0.0087), and segmentation and the rank threshold use it.
- **Weekly scans all land at the same point of the dose cycle.** To day 84,
  every treated weekly scan falls 6 days after a dose or on a dose day. The
  scans then cannot separate the drug from slower growth.

A third cause, too few numbers for the unknowns, is tested on the side with
two modes instead of three or four.

**Fix 1: the scans' known noise (`*-known-noise`).** Each mode's noise prior
is centred on the noise its reduced observations carry. This is computed from
the scans and the known noise level alone
(`benchmark_data.scan_noise_variances`). For mode i:

```text
nu_i = sigma^2 + mean over scans j of (n_j * sigma^2 * y_ij / s_i^2)^2
```

- **Symbols.** sigma is the voxel noise SD and n_j the number of segmented
  voxels in scan j. y_ij is the reduced observation, and s_i is mode i's
  singular value of the scan matrix (the size of y_i).
- **First term.** The scanner noise inside the outline. For a fixed
  orthonormal mode it is at most sigma^2.
- **Second term.** The noise a mode soaks up because the POD was fitted to
  the same noisy scans. It grows with the lesion and is large for weak modes.
  The earlier `04_unified_chemo.py` profile fitted its basis to clean data,
  so it needed only the first term.
- **Width.** The prior's width is unchanged: log-normal with scale 1.0, a
  factor of 2.7 either way. Nothing else changes.

Prior centre divided by the voxel noise variance. The known-noise values
agree to within 2% on acquisitions 45 and 48-50.

| Prior and scans | Mode 1 | Mode 2 | Mode 3 | Mode 4 |
|---|---|---|---|---|
| Default (1% of variance), weekly to day 84 | 42,600 | 17,300 | 1,150 | 159 |
| Known noise, weekly to day 84 | 1.4 | 18 | 244 | 1,830 |
| Known noise, every 8 days to day 84 | 1.5 | 21 | 268 | 2,100 |
| Known noise, weekly to day 57 | 1.6 | 33 | 607 | |
| Known noise, every 8 days to day 57 | 1.7 | 34 | 550 | |

The default prior is not just too large; it has the wrong shape. It treats
mode 1 as the noisiest mode and mode 4 as the cleanest, but the reverse is
true.

**Checks of the formula, before any fit.**
- **Synthetic test.** In
  `test_scan_noise_variances_predict_the_reduced_observation_error`, the
  formula predicts the weakest mode's error to within 1% on average.
- **Development acquisition 45** (actual error divided by declared):
  - Mode 1: 1.55-1.73. The formula leaves out segmentation errors, such as
    clipping and the outline.
  - Mode 2: 0.71-0.78.
  - Weaker modes: 1.00-1.07.
- **Disclosure.** While deriving the formula, I also compared it with the
  actual error on evaluation acquisitions 48-50. That covered weekly,
  fortnightly and four-weekly scans, and the ratio was 0.74-1.38. The
  comparison used their noise-free fields, which are evaluation-only. No
  forecast was computed with the new prior, and the formula has no tuned
  constant.

**Fix 2: scans every 8 days (`*-every8-scans`).** One scan every 8 days from
day 5, plus one on the last training day (`stride_scan_days`). Each scan
falls a day later in the weekly dose cycle than the one before. There are
never more scans than weekly. A research protocol would need some visits on
days without an infusion.

| Data to | Schedule | Scan days | Days after the last dose (treated scans) |
|---|---|---|---|
| Day 84 | Weekly (declared earlier) | 5, 12, 19, 27, 34, 41, 48, 55, 62, 70, 77, 84 (12) | 6, 6, 6, 6, 0, 0, 0 |
| | Every 8 days | 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 84 (11) | 2, 3, 4, 5, 6, 0, 0 |
| Day 57 | Weekly (declared earlier) | 5, 12, 20, 27, 35, 42, 50, 57 (8) | 0, 0, 1, 1 |
| | Every 8 days | 5, 13, 21, 29, 37, 45, 53, 57 (8) | 2, 3, 4, 1 |

The weekly schedules' ranks also clear the scan-noise threshold here on
acquisitions 45 and 48-50. The fourth mode is 1.18 times the threshold to
day 84, and the third is 2.4 times to day 57.

**Side check: two modes (`*-every8-scans-known-noise-r2`).** Scans every 8
days with the known noise, at rank 2. That leaves 10 unknown operator
entries, against 36 at rank 4 and 21 at rank 3. The basis floor rises, and
it is reported.

**Control: 120 evenly spaced scans with the known noise** (to day 84 only).
It checks that the new prior does not hurt dense scans.

**Design.** Nine cases (`benchmark_cases.SPARSE_FIX_CASES`). Each is built
from a declared case, and only the named settings change.

| Case suffix | Scans to day 84 / 57 | Noise prior | Rank to day 84 / 57 |
|---|---|---|---|
| `-weekly-scans-known-noise` | weekly: 12 / 8 | known | 4 / 3 |
| `-every8-scans` | every 8 days: 11 / 8 | default | 4 / 3 |
| `-every8-scans-known-noise` | every 8 days: 11 / 8 | known | 4 / 3 |
| `-every8-scans-known-noise-r2` | every 8 days: 11 / 8 | known | 2 / 2 |
| `-regular-scans-known-noise` | 120 evenly spaced / not run | known | 4 / - |

Each of the first three rows makes one paired comparison with the weekly
results:
- **First row:** only the prior changes (same scans, same noise).
- **Second row:** only the timing changes.
- **Third row:** both change.

**Rule.** As in "Realistic scan schedules":
1. **Runs.** Seeds 48-50, each run once. Nothing is tuned.
2. **Metric and arms.** The same metric. Scenario arms are 1, 0.75 and 0.5x;
   0.25x and 0x are stress arms, reported and not judged.
3. **Bar.** A case passes on its split if every scenario arm's median is at
   most 18.9%.
4. **Also reported.**
   - The basis floor, the training-window error and the day-119 burden.
   - How often the 0.5x and 1x bands hold the truth, and their widths.
   - Mode 1's drug coefficient, c_1 and the share of the true dose effect.
   - The Gaussian process's lengthscale, and its noise estimate against the
     declared value.

**Expectations, written before any fit.**
- **Known noise, weekly.**
  - To day 84, the fit should follow its scans much more closely: training
    error well under the earlier 18%. But the scans still sit at the top of
    the dose cycle, so I expect little drug effect and a fail at 0.5x.
  - To day 57, the fit already followed its scans (3.2%), so I expect little
    change.
- **Every 8 days, default prior.** Some drug effect may appear, but the
  Gaussian process still smooths over the dose cycle. I expect a fail.
- **Every 8 days with known noise.** This has the best chance.
  - To day 84, a pass would not surprise me.
  - To day 57 a pass is less likely: 8 scans, 21 unknowns and only 4 treated
    scans.
- **Two modes.** Fewer unknowns, but a higher floor. It may help to day 57.
- **Control.** Still passes, within about 2 points of 8.9, 6.3 and 6.1%.

If no realistic schedule passes, the next step is a population prior on the
tumour-wide growth and kill rates, learned from a virtual cohort. It will be
declared separately.

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-every8-scans-known-noise   # 48-50
# likewise every case in benchmark_cases.SPARSE_FIX_CASES
```

The data-only check is session `sparse_fix_check.py`, and its output is
`sparse_fix_check.json`.

### Results (each seed run once)

**Every fix fails, on both splits. The dense control passes.** All 27 fits
finished with 64 of 64 draws. Median point-forecast error (%) over seeds
48-50; the bar is 18.9% on every scenario arm:

| Data to | Case | Scans, rank | 1x | 0.75x | 0.5x | Verdict |
|---|---|---|---|---|---|---|
| Day 84 | Weekly (declared earlier) | 12, r4 | 22.1 | 24.1 | 32.4 | fail |
| | Weekly + known noise | 12, r4 | 29.3 | 30.3 | 32.9 | fail |
| | Every 8 days | 11, r4 | 47.2 | 32.9 | 27.0 | fail |
| | Every 8 days + known noise | 11, r4 | 106.6 | 97.5 | 90.8 | fail |
| | Every 8 days + known noise, 2 modes | 11, r2 | 45.2 | 44.2 | 45.2 | fail |
| | 120 even + known noise (control) | 120, r4 | 8.4 | 6.1 | 6.1 | pass |
| | 120 even (declared earlier) | 120, r4 | 8.9 | 6.3 | 6.1 | pass |
| Day 57 | Weekly (declared earlier) | 8, r3 | 42.4 | 49.1 | 53.9 | fail |
| | Weekly + known noise | 8, r3 | 30.9 | 32.8 | 36.2 | fail |
| | Every 8 days | 8, r3 | 39.2 | 47.6 | 54.4 | fail |
| | Every 8 days + known noise | 8, r3 | 107.7 | 64.1 | 33.1 | fail |
| | Every 8 days + known noise, 2 modes | 8, r2 | 45.4 | 40.8 | 50.1 | fail |

The three seeds agree to within 1.5 points everywhere. On a fixed schedule
the seeds share the scan days and differ only in scanner noise, so this
agreement shows robustness to the noise, not to a different patient or
timing.

What changed inside the fits (ranges over seeds 48-50):

| Data to | Case | Training error | Dose effect at 0.5x | Day-119 burden / truth at 1x | Mode 1: c_1, N_11 |
|---|---|---|---|---|---|
| Day 84 | 120 even (earlier) | 1.2% | 0.73-0.74 | 1.03-1.04 | 2.24, -0.95 |
| | Weekly (earlier) | 18.2% | 0.06-0.07 | 0.80-0.82 | 0.09, -0.15 to -0.14 |
| | Weekly + known noise | 10.3% | 0.37 | 0.66-0.67 | 0.53, -0.62 |
| | Every 8 days | 23.3% | 0.02 | 1.21-1.22 | -0.02, -0.04 |
| | Every 8 days + known noise | 43.2% | 0.01-0.02 | 0.33-0.34 | -0.11, -0.04 |
| | Every 8 days + known, 2 modes | 19.8% | 0.28 | 0.53 | 0.63, -0.48 |
| Day 57 | 120 random (reported) | 0.8% | 0.50-0.60 | 1.01-1.06 | 2.24 to 2.43, -0.99 to -0.96 |
| | Weekly (earlier) | 3.2% | 0.25 | 0.18 | 0.29, -0.66 |
| | Weekly + known noise | 2.1% | 0.45 | 0.37 | 0.75, -0.86 |
| | Every 8 days | 5.5% | 0.21 | 0.17-0.18 | 0.15, -0.56 |
| | Every 8 days + known noise | 19.7% | 0.02 | 2.18-2.19 | 0.04, -0.02 |
| | Every 8 days + known, 2 modes | 16.5% | 0.01 | 0.46-0.49 | -0.16, -0.03 |

- **Known noise, weekly.** The fit follows its scans more closely, and part
  of the drug effect comes back (dose effect 0.06 to 0.37 to day 84, 0.25 to
  0.45 to day 57). But the growth source c_1 stays at a quarter to a third
  of the dense fits' value, so every forecast shrinks too much. To day 84
  the errors rise; to day 57 they fall by 11-18 points but stay far above
  the bar.
- **Every 8 days, default prior.** No drug effect at all to day 84. The
  default prior still calls the dose cycle noise (mode 1's noise estimate is
  about 280,000 times the scanner's).
- **Every 8 days with known noise: the worst of all.** The Gaussian process
  can no longer call the cycle noise, so it shortens mode 1's lengthscale to
  4.7 days (day 84) or 6.5 days (day 57) and wiggles between scans. The
  training error is 43% and 20%. To day 57 the forecast more than doubles
  the tumour at 1x.
- **Two modes.** Fewer unknowns did not help. The basis floor rises to
  12.7-14.4% (day 84) and 14.1-20.5% (day 57) on the scenario arms.
- **Control.** The known noise does not hurt dense scans: 8.4, 6.1 and
  6.1%, against 8.9, 6.3 and 6.1%. Its 1x band is narrower (width 0.09-0.10
  against 0.12) and holds the truth 92-95% of the time, against 100%.

**Expectations against outcomes.**

| Case | Expected | Outcome |
|---|---|---|
| Known noise, weekly, day 84 | Training error well under 18%; little drug effect; fail at 0.5x | Training error 10.3%; dose effect 0.37; fails every arm, worse than before at 1x and 0.75x |
| Known noise, weekly, day 57 | Little change | Errors fall by 11-18 points; still fails |
| Every 8 days, default prior | Some drug effect; fail | No drug effect to day 84; fail |
| Every 8 days + known noise | Best chance; a pass to day 84 would not surprise | The worst results of all |
| Two modes | May help to day 57 | Fails; floor 14-21% |
| Control | Pass within about 2 points | Pass, 0-0.5 points better |

The two fixes I expected most from did the most damage when combined.

### Why the fixes fail (after the fact)

Two checks, made after the fits. Neither fits a ROM or computes a forecast.

**1. The Gaussian process hands the regression poor derivatives.** The
regression learns the operators from the Gaussian process's state and rate
of change between the scans. For mode 1 on seed 48, with each fit's mean
hyperparameters, I compared both with the noise-free truth over the
training window (session `gp_mode1_check.py`,
`figures/gp_mode1_seed48.png`):

| Scans and noise prior | Lengthscale (days) | Noise variance / scanner's | State error | Rate error |
|---|---|---|---|---|
| 120 even, default | 3.3 | 4,643 | 0.4% | 25.7% |
| Weekly, default | 16.9 | 333,000 | 5.4% | 75.5% |
| Weekly, known noise | 4.7 | 2.3 | 3.1% | 47.6% |
| Every 8 days, default | 12.7 | 321,000 | 5.6% | 81.9% |
| Every 8 days, known noise | 4.7 | 2.5 | 4.6% | 70.1% |
| Day 57 weekly, default | 48 | 114,000 | 3.4% | 38.4% |

- **Default prior.** The Gaussian process treats the weekly drop and
  regrowth as noise and draws a smooth curve through it. For every 8 days,
  mode 1's curve even peaks around day 29, before the first dose on day 35.
- **Known noise.** It can no longer call the cycle noise. It cannot follow
  it either, with 7-8 days between scans, so it bends sharply at each scan.

**2. The trend's prior is too tight.** The input-aware Gaussian process
already has a shape for the dose cycle. Its trend has three parts: a
constant, a straight line in time and the cumulative drug exposure. The
last one steps down at each dose, and the straight line regrows the tumour
between doses. But the prior on each part's coefficient has a fixed spread:
the mode's observed spread across the scans. With 120 scans the data pin the
coefficients and the prior hardly matters. With 8-12 scans it holds them
near zero. The Gaussian process then explains the cycle with noise or with
sharp bends.

A data-only check fits the Gaussian process alone to mode 1's scans and
multiplies the trend's prior variance by m (session `gp_trend_check.py`,
`figures/gp_trend_seed48.png`, `figures/gp_trend_seed45.png`). The other
hyperparameters are refitted by maximum marginal likelihood for each m. I
scored the weak-form targets, the numbers the regression actually fits,
against the noise-free truth:

| Seed 48, mode 1 (noise prior) | m = 1 (as fitted) | m = 10 | m = 100 | Marginal likelihood, m = 10 against 1 |
|---|---|---|---|---|
| Day 84 weekly (default) | 28.2% | 8.8% | 8.6% | +6.9 |
| Day 84 weekly (known) | 25.9% | 21.1% | 10.5% | +7.1 |
| Day 84 every 8 days (default) | 35.4% | 16.8% | 15.3% | +4.0 |
| Day 84 every 8 days (known) | 19.4% | 10.4% | 10.3% | +7.1 |
| Day 57 weekly (default) | 22.2% | 8.5% | 7.2% | +1.4 |
| Day 57 weekly (known) | 19.1% | 8.5% | 8.4% | +7.0 |
| Day 84 120 even (default) | 0.5% | 0.5% | 0.5% | +10.0 |

- **The scans themselves prefer a looser trend.** The marginal likelihood
  is higher at m = 10 in every case, by 1.4-10 log units, and falls again at
  m = 100 and 1,000.
- **The targets improve 2-3 times on every sparse schedule.** The dense
  control does not change.
- **Development acquisition 45 gives the same numbers** to within 1.5
  points. It adds day 57 every 8 days: 32.1% to 10.7% (default noise) and
  50.9% to 15.4% (known noise).
- **The remaining error** comes from the trend's constant step per dose. The
  true kill shrinks as the tumour shrinks, so the later troughs overshoot.
- **Disclosure.** Seed 48 is an evaluation acquisition. The check used its
  scans and its noise-free training-window states, no forecast. Seed 45 is
  the development acquisition, but on a fixed schedule it is a
  near-replicate of 48-50: only the scanner noise differs.

**3. Even good targets carry little information.** The tempering gives
each mode about as many effective observations as the Gaussian process has
degrees of freedom: 6-8 per mode on weekly scans to day 84. A rank-4 mode
has 9 unknowns: c_i, four A entries and four N entries. (An earlier version
said 10, with a direct drug term B_i; the free model has no B on an
uncentred basis, see `benchmark_cases.galerkin_operators`.) Part of the
answer therefore comes from the operator prior, which is centred on zero.
That fits the way c_1 and N_11 shrink towards zero in the tables above.

**What this means.** The known noise removed the wrong explanation (noise)
without making the right one (the dose cycle) available, so the Gaussian
process found a third one (sharp bends). Timing alone cannot help while the
Gaussian process smooths the cycle away. The next declared test therefore
loosens the trend's prior; see "Learning the Gaussian process's trend
scale".

## Learning the Gaussian process's trend scale (rule declared before any fit)

**Why.** "Why the fixes fail" found that the input-aware Gaussian process's
trend prior is too tight for 8-12 scans. The process then explains the dose
cycle with noise or sharp bends, and the regression gets poor targets. The
scans themselves prefer a looser trend. In the data-only check, of the
factors tried (1, 10, 100 and 1,000), the marginal likelihood was highest
at 10 times the prior variance, about 3 times the SD. Fixing a factor of 10
after seeing that check would be tuning, so the factor is learned instead,
like the other Gaussian-process settings.

**Change: one setting, `gp_trend_scale="learned"`** (`WeakFormConfig`).
- **Before (`"fixed"`, every earlier fit).** Each trend coefficient of mode
  i (the constant, the straight line and the cumulative drug exposure) has
  prior SD sd(y_i): the SD of mode i's observations across the scans.
- **Now.** The prior SD is kappa_i * sd(y_i). kappa_i is one new
  Gaussian-process hyperparameter per mode, with log kappa_i ~ N(0, 3^2):
  a factor of 20 either way at one SD. This is the width the operator
  block scales already use (`hier_tau_scale`).
- **Where it is fitted.** With the lengthscale, variance and noise of the
  same mode, in both places those are fitted:
  - the data-only fit that sets the tempering;
  - the main fit.
- **Check.** kappa_i = 1 gives back the fixed model exactly (unit tests in
  `core/weakform_opinf/tests/test_trend_scale.py`).
- **Unchanged.** Everything else: the default noise prior, the acquisitions,
  bases and ranks, and the training steps.

**Design.** Five cases (`benchmark_cases.LEARNED_TREND_CASES`). Each repeats
a declared case with the same acquisitions, and only the trend's prior
changes. The known-noise prior is not carried over: each case then differs
from its declared partner in one setting only.

| Case | Scans | Rank | Fixed-trend partner (median 1 / 0.75 / 0.5x) |
|---|---|---|---|
| `midcourse-reduced-trend-chemo-weekly-scans-learned-trend` | weekly to day 84 (12) | 4 | 22.1 / 24.1 / 32.4 |
| `midcourse-reduced-trend-chemo-every8-scans-learned-trend` | every 8 days to day 84 (11) | 4 | 47.2 / 32.9 / 27.0 |
| `midcourse-reduced-trend-chemo-regular-scans-learned-trend` | 120 evenly spaced to day 84 (control) | 4 | 8.9 / 6.3 / 6.1 |
| `reduced-regimen-trend-chemo-weekly-scans-learned-trend` | weekly to day 57 (8) | 3 | 42.4 / 49.1 / 53.9 |
| `reduced-regimen-trend-chemo-every8-scans-learned-trend` | every 8 days to day 57 (8) | 3 | 39.2 / 47.6 / 54.4 |

**Rule.** As in "Realistic scan schedules":
1. **Runs.** Seeds 48-50, each run once. Nothing is tuned.
2. **Metric and arms.** The same metric. Scenario arms are 1, 0.75 and 0.5x;
   0.25x and 0x are stress arms, reported and not judged.
3. **Bar.** A case passes on its split if every scenario arm's median is at
   most 18.9%.
4. **Also reported.**
   - The learned kappa (main fit, and the data-only fit) and the tempering's
     effective observations per mode.
   - The training-window error, the day-119 burden and the share of the
     true dose effect.
   - How often the 0.5x and 1x bands hold the truth, and their widths.
   - Mode 1's drug coefficient, c_1 and N_11, and the Gaussian process's
     lengthscale and noise.

**Expectations, written before any fit.**
- **Control (120 scans).** No change: within about 1 point of 8.9, 6.3 and
  6.1%. With dense scans, the scale made no difference to the targets in the
  data-only check.
- **Weekly and every 8 days, to day 84.**
  - Part of the drug effect should come back, and the errors should fall.
  - A pass at 1x and 0.75x is possible.
  - At 0.5x a fail is more likely: the targets still carry 9-17% error,
    and each mode has 6-8 effective observations for 10 unknowns. The rest
    comes from the operator prior, centred on zero. (Corrected after the
    fits: 9 unknowns, not 10; the free model has no direct drug term B on an
    uncentred basis. The reasoning is unchanged.)
- **To day 57.** Probably fails: 8 scans, of which only 4 are treated. Even
  with 120 scans, the free model's regrowth after a cut from day 57 was too
  slow.

**Disclosures.**
- **The diagnosis used data.** It fitted Gaussian processes alone (no ROM)
  to the scans of evaluation seed 48 and development seed 45. It scored the
  targets against seed 48's noise-free training-window states. No forecast
  was computed.
- **Smoke run.** A smoke run checked the code end to end on development
  seed 45 (`midcourse-reduced-trend-chemo-weekly-scans-learned-trend`). It
  used 20 training updates instead of 12,000, far from converged: the main
  fit's kappa had moved from 1 to 1.2-2.4. The data-only fit, which only
  sets the tempering, ran in full and gave kappa = 3.8, 0.33, 1.5 and 0.61
  for modes 1-4. It printed forecast errors of 21.8, 24.8 and 29.8% (1, 0.75 and
  0.5x). These are not results and set nothing.
- **Seeds are near-replicates.** On a fixed schedule, seeds 45 and 48-50
  differ only in scanner noise. Seed agreement here shows robustness to
  noise, not to scan timing or to another tumour.

If no realistic schedule passes, the next step is a population prior on the
tumour-wide growth and kill rates, learned from a virtual cohort. It will be
declared separately.

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans-learned-trend   # 48-50
# likewise every case in benchmark_cases.LEARNED_TREND_CASES
```

### Results (each seed run once)

**One realistic schedule now passes: scans every 8 days to day 84. The
others still fail, though their errors fall by 33-71%.** All 15 fits
finished with 64 of 64 draws. Median point-forecast error (%) over
seeds 48-50; the bar is 18.9% on every scenario arm:

| Data to | Scans, rank | Trend prior | 1x | 0.75x | 0.5x | Verdict |
|---|---|---|---|---|---|---|
| Day 84 | Weekly, 12, r4 | fixed (earlier) | 22.1 | 24.1 | 32.4 | fail |
| | | learned | 13.6 | 16.1 | 21.1 | fail |
| | Every 8 days, 11, r4 | fixed (earlier) | 47.2 | 32.9 | 27.0 | fail |
| | | learned | 13.6 | 12.8 | 17.8 | **pass** |
| | 120 even, r4 (control) | fixed (earlier) | 8.9 | 6.3 | 6.1 | pass |
| | | learned | 8.9 | 6.4 | 6.1 | pass |
| Day 57 | Weekly, 8, r3 | fixed (earlier) | 42.4 | 49.1 | 53.9 | fail |
| | | learned | 20.1 | 26.7 | 32.1 | fail |
| | Every 8 days, 8, r3 | fixed (earlier) | 39.2 | 47.6 | 54.4 | fail |
| | | learned | 12.3 | 18.2 | 25.4 | fail |

- **The pass is narrow but holds on every seed.** At 0.5x the three seeds
  give 17.80, 17.80 and 17.78%. As before, on a fixed schedule the seeds
  differ only in scanner noise, and they agree within 0.3 points
  everywhere.
- **Stress arms (not judged), 0.25x and 0x.** Day 84: weekly 26.7 and
  32.1%, every 8 days 24.6 and 31.4%, control 7.0 and 8.0%. Day 57: weekly
  34.8 and 34.8%, every 8 days 30.7 and 33.6%.
- **For reference, 120 random scans** (the reported cases): 9.1, 6.9 and
  6.7% from day 84; 16.6, 12.0 and 13.9% from day 57.

What changed inside the fits (ranges over seeds 48-50). kappa_1 is mode 1's
learned trend scale; 1 is the fixed prior. The dose effect is the share of
the true day-119 difference between 0.5x and 1x that the forecast gets:

| Data to | Scans | Trend prior | Training error | kappa_1: fit (data-only) | c_1 | N_11 | Dose effect at 0.5x | Day-119 burden / truth, 1x and 0.5x |
|---|---|---|---|---|---|---|---|---|
| Day 84 | Weekly | fixed | 18.2% | 1 | 0.09 | -0.15 to -0.14 | 0.06-0.07 | 0.80-0.82, 0.47-0.49 |
| | | learned | 4.5% | 2.85-2.86 (3.83) | 0.68-0.69 | -0.73 | 0.38 | 0.85, 0.64 |
| | Every 8 days | fixed | 23.3% | 1 | -0.02 | -0.04 | 0.02 | 1.21-1.22, 0.69 |
| | | learned | 5.9% | 3.01-3.02 (3.28) | 0.77 | -0.65 | 0.33 | 0.95, 0.68 |
| | 120 even | fixed | 1.2% | 1 | 2.24-2.25 | -0.95 | 0.73-0.74 | 1.03-1.04, 0.90 |
| | | learned | 1.2% | 2.22 (3.12-3.13) | 2.25 | -0.95 | 0.75 | 1.04, 0.91 |
| Day 57 | Weekly | fixed | 3.2% | 1 | 0.29 | -0.66 | 0.25 | 0.18, 0.22-0.23 |
| | | learned | 2.3% | 3.53 (3.19) | 0.91-0.92 | -0.84 | 0.44 | 0.59-0.60, 0.50 |
| | Every 8 days | fixed | 5.5% | 1 | 0.15 | -0.56 | 0.21 | 0.17-0.18, 0.20 |
| | | learned | 2.4% | 4.15 (3.67-3.68) | 1.21-1.22 | -0.84 | 0.44 | 0.75, 0.56 |
| | 120 random (reported) | fixed | 0.8% | 1 | 2.24-2.43 | -0.99 to -0.96 | 0.50-0.60 | 1.01-1.06, 0.70-0.75 |

- **The learned scale lands where the data-only check pointed.** Mode 1's
  kappa is 2.9-4.2 on the sparse schedules: a trend variance 8-17 times the
  fixed one, close to the factor of 10 the check preferred. The other modes
  stay between 0.3 and 1.7 (for example 0.39, 1.6 and 0.79 for modes 2-4,
  weekly to day 84).
- **The drug effect partly comes back.** To day 84 the training error falls
  from 18-23% to 4.5-5.9%, and the dose effect from 0.02-0.07 to 0.33-0.38.
  To day 57 the dose effect rises from 0.21-0.25 to 0.44.
- **The growth and kill rates are still about half what the scans need.**
  c_1 is 0.68-1.22 against 2.0-2.4 with dense scans, and N_11 is -0.65 to
  -0.84 against -0.95 to -0.99. The forecasts therefore regrow too little
  after a cut: at 0.5x the day-119 burden is 0.50-0.68 of the truth, against
  0.70-0.91 with dense scans.
- **Scans every 8 days now beat weekly scans.** To day 84 they tie at 1x
  and win at 0.75x and 0.5x; to day 57 they win at every arm. Once the
  Gaussian process can draw the dose cycle, scans at different points of the
  cycle help, as idea 2 intended. Weekly scans still sit near the top of
  every cycle.
- **The bands are wide.** At day 119 the 5-95% band spans 0.73-1.15 times
  the true burden from day 84, and 0.80-2.40 times from day 57. With 120
  random scans it spans 0.10-0.18 and 0.19-0.25 times. From day 84 the bands
  hold the truth at 1x and 0.75x all the time, and at 0.5x 91% (weekly) and
  100% (every 8 days) of the time. From day 57 the 0.5x band holds it 24%
  (weekly) and 98-99% (every 8 days) of the time.
- **The Gaussian process still smooths heavily.** Mode 1's noise estimate
  falls from about 290,000 to 77,000-92,000 times the scanner's (day 84),
  and its lengthscale is 13-14 days, against 3.3 days with dense scans. The
  trend's cumulative drug exposure now carries the steps.
- **The control does not move**, as expected: dense scans fix the trend
  whatever its prior. Its kappa_1 is 2.2.

**Expectations against outcomes.**

| Case | Expected | Outcome |
|---|---|---|
| Control | No change, within about 1 point | 8.9, 6.4 and 6.1%: within 0.1 points |
| Weekly to day 84 | Errors fall and part of the drug effect comes back; a pass at 1x and 0.75x possible, a fail at 0.5x more likely | As expected: 13.6, 16.1 and 21.1%, fails at 0.5x only |
| Every 8 days to day 84 | The same | Better than expected: passes at every arm, at 0.5x by 1.1 points |
| To day 57 | Probably fails | Both fail. Errors fall by 22-29 points; every 8 days is under the bar at 1x and 0.75x and 25.4% at 0.5x |

**What this means.** The trend's prior was the main reason the sparse fits
learned no drug effect. With it learned, a realistic schedule (11 scans in
84 days) passes the declared bar. What remains is a lack of information:
each mode has 6-8 effective observations for 9 unknowns, and the operator
prior pulls the growth and kill rates towards zero.

The declared next step was a population prior "if no realistic schedule
passes". One schedule now passes, so that trigger holds only for the three
that still fail. The population prior is still the next idea in the agreed
order, and it targets what is left: it centres the operator prior on rates
learned from other patients instead of on zero. It will be declared before
any fit.

## A population prior from past patients (rule declared before any fit)

**Why.** With the learned trend, the sparse fits still get the growth and
kill rates about half what the scans need. Each mode has 6-8 effective
observations for 9 unknowns, and the operator prior pulls the rest towards
zero. A clinic does not start from zero: it knows how similar tumours
responded before. This test gives the ROM that knowledge. An operator
learned from past patients' scans becomes the centre of the operator
prior, and each new patient still gets their own fit. It is the "combine
several patients' scans into a general model" idea, with the general model
used as a starting point rather than as the answer.

**The cohort** (`benchmark_cohort.py`, design `benchmark_cases.COHORT_DESIGN`).
1. **Eight virtual past patients.** They share the target's breast and its
   tumour at the first visit, as TumorTwin's own TNBC demo patients do.
   Each has their own growth rate, diffusion and drug sensitivity: the
   twin's value times exp(0.3 z), where z is drawn once for the cohort's
   centre and once for each member (seed 101). So the cohort's typical
   patient is not the target.
2. **Their courses.** All finished the twelve weekly doses from day 35.
   Three had a neuropathy hold and then a reduced dose, as the protocol
   prescribes: member 1 held dose 8 (day 84), then 0.75x; member 4 held
   dose 6 (day 70), then 0.75x; member 6 held dose 7 (day 77), then 0.5x.
3. **Simulated with the target's TumorTwin solver.** With the twin's values
   and the planned course, the member solver reproduces the target's
   simulation bit for bit (days 1-119).
4. **Scanned on the target's protocol.** From day 5 to day 119, with the
   target's scanner noise and segmentation, every 7 days (132 scans in
   all) or every 8 days (115). The members' first scans are staggered
   (day 5 + member mod 7 or 8), so the cohort sees the dose cycle at
   several points.
5. **One basis for everyone.** The POD of all members' noisy scans, pooled.
   Eight modes clear the noise threshold for both protocols; the rank-4
   and rank-3 bases are its first columns.
6. **The population operator.** The production fit, with the target case's
   own settings (free cAN, input-aware GP, learned trend scale), to all
   eight members at once, one trajectory each, in the cohort basis. Its
   point operator is the population centre M.

**Change for the target: two settings.**
- **Basis.** The target's own scans are projected onto the cohort basis of
  its protocol, instead of making their own POD.
- **Prior centre.** O_i ~ N(M_i, Σ_i) instead of N(0, Σ_i) for each
  mode's row. The prior widths are unchanged and still learned, so the
  target's scans decide how far the fit moves from M. The evidence is the
  same closed form after shifting the targets to y − A M_i (exact for the
  linear-Gaussian model; `core/weakform_opinf/tests/test_prior_mean.py`).
  Earlier sections called this a prior on "the tumour-wide growth and kill
  rates". In a shared basis those rates are the operator's entries (c, A
  and N), so the whole operator is centred.

**Design.** Eight cases (`benchmark_cases.COHORT_CASES`), two per realistic
schedule. Each has the same acquisitions as its learned-trend partner (the
same scan days and noise; checked for seeds 48-50). The cohort uses the
target's own imaging protocol.
- **`-cohort-basis`:** the cohort basis, prior centred on zero. It shows
  what the basis alone does.
- **`-cohort-prior`:** the cohort basis, prior centred on M.

| Target schedule | Rank | Cohort protocol | Learned-trend partner (own basis, median 1 / 0.75 / 0.5x) |
|---|---|---|---|
| Weekly to day 84 (12 scans) | 4 | weekly | 13.6 / 16.1 / 21.1, fail |
| Every 8 days to day 84 (11) | 4 | every 8 days | 13.6 / 12.8 / 17.8, pass |
| Weekly to day 57 (8) | 3 | weekly | 20.1 / 26.7 / 32.1, fail |
| Every 8 days to day 57 (8) | 3 | every 8 days | 12.3 / 18.2 / 25.4, fail |

Case names: `midcourse-reduced-trend-chemo-{weekly,every8}-scans-learned-trend-{cohort-basis,cohort-prior}`
(day 84) and `reduced-regimen-trend-chemo-...` (day 57).

**The cohort as built** (data only, before this declaration; record
`experiments/tumor/data/benchmarks/cohort_064076a008cb.json`, generated by
`benchmark_cohort.py build` and not committed, like the scans).
- **Members** (growth rate k, diffusion D, sensitivity; the twin is 0.0521,
  0.0965 and 0.23):

  | Member | k | D | Sensitivity | Course | Burden on day 119 |
  |---|---|---|---|---|---|
  | 0 | 0.051 | 0.048 | 0.31 | planned | 2,848 |
  | 1 | 0.069 | 0.072 | 0.34 | hold, then 0.75x | 12,937 |
  | 2 | 0.051 | 0.040 | 0.37 | planned | 1,091 |
  | 3 | 0.025 | 0.047 | 0.24 | planned | 1,147 |
  | 4 | 0.024 | 0.051 | 0.45 | hold, then 0.75x | 168 |
  | 5 | 0.037 | 0.036 | 0.27 | planned | 1,839 |
  | 6 | 0.053 | 0.056 | 0.28 | hold, then 0.5x | 15,543 |
  | 7 | 0.027 | 0.061 | 0.32 | planned | 389 |
  | Target | 0.052 | 0.097 | 0.23 | planned | 9,753 |

- **The target is not typical.** The cohort's centre is k = 0.041,
  D = 0.052 and sensitivity 0.28. The target sits 0.8, 2.0 and -0.6
  population SDs from it: it grows and spreads faster and responds less.
  The five members on the planned course end with 389-2,848 cells on day
  119; the target keeps 9,753.
- **Checks.** The clone check is bitwise. The members' drug input
  reproduces the target's exposure table (largest difference 1.6e-15).
  The member scans have no false-positive voxels. Their error is 3.3-3.8%
  (the target's training scans: 2.5-3.1%).
- **The target's scans in the cohort basis** (seeds 48-50): the cohort
  basis keeps 99.86% of their energy to day 84 and 99.83% to day 57,
  against 99.93% and 99.95% for their own POD. As a relative error:
  3.7-3.8% against 2.6%, and 4.1% against 2.2-2.3%. Their own POD also
  fits part of the scans' noise.

**Rule.**
1. **Runs.** The four population operators (weekly and every 8 days, rank 4
   and 3) are fitted once with the production recipe (12,000 updates). They
   are used as fitted, whatever their in-sample error. Then seeds 48-50,
   both arms, each run once. Nothing is tuned.
2. **Metric and arms.** The same metric. Scenario arms are 1, 0.75 and 0.5x;
   0.25x and 0x are stress arms, reported and not judged.
3. **Bar.** A case passes if every scenario arm's median is at most 18.9%.
4. **Reading.** The population prior works on a schedule if its
   `-cohort-prior` case passes. The `-cohort-basis` case shows how much of
   any change the basis alone makes.
5. **Also reported.**
   - The population fits: each member's in-sample error and last-scan
     burden ratio, and M's c_1 and N_11.
   - The target fits: kappa, c_1 and N_11, the training-window error, the
     share of the true dose effect, and the day-119 burden.
   - How often the 0.5x and 1x bands hold the truth, and their widths.
   - The learned prior widths (how far the fit moved from M).

**Expectations, written before any fit.**
- **Population fits.** One operator cannot fit eight different tumours
  exactly. A rough guess: in-sample errors of 5-20%, largest for the
  members who regrow after a cut (1 and 6) and the strongest responder (4).
- **Cohort basis alone.** Close to the partners, within a few points either
  way, and no change of verdict. It keeps slightly less of the target's
  scans but has seen regrowth after cuts.
- **Cohort prior, to day 84.** Errors at 0.5x fall: the rates are pulled
  towards the population's instead of towards zero. Weekly may now pass;
  every 8 days should stay a pass.
- **Cohort prior, to day 57.** Uncertain, leaning fail. With only 4 treated
  scans the fit leans most on the prior, and the target is not typical. A
  prior centred on the cohort pulls towards too much shrinkage, so the
  forecast burden may come out too low.

**Disclosures.**
- **The best case for sharing.** One anatomy and one baseline tumour make a
  shared basis easy. Real patients would need image registration onto a
  common grid, and their tumours would differ in place and shape.
- **A narrow spread.** One SD of 0.3 in log is a factor of 1.35 either
  way; the members span factors of 2-3 (k 0.024-0.069, D 0.036-0.072,
  sensitivity 0.24-0.45). Measured patients vary more: in Jarrett et al.
  (2018, Phys. Med. Biol. 63:105015; five patients, 3D) the fitted drug
  efficacy spans 18-485 (a factor of 27) and the diffusion a factor of 10.
  A wider spread would make the population centre less informative about
  any one patient.
- **Design choices, not calibrated rates.** Three of eight members (37.5%)
  have a dose modification, so that the cohort contains cuts like the
  scenario arms. The scan phases are staggered on purpose.
- **Common drug units.** Each member's drug input uses the target's units
  (0.23 times the dose fraction). Their own sensitivities differ, so the
  differences land in the operator's N, as they would for real patients.
- **The trigger.** The declared trigger was "if no realistic schedule
  passes". Scans every 8 days to day 84 now pass, so that schedule is
  included only to check that the prior does not break it.
- **Built before declaring.** The cohort was simulated, scanned and pooled,
  and the captured-energy check run, before this declaration. That is data
  only. The parameter seed (101) was fixed in the code before the build.
- **Smoke runs.** A smoke run checked the code end to end, written outside
  the results tree. It used development seed 45, weekly scans to day 84,
  and 20 training updates instead of 12,000, both for the population fit
  and the target. It printed forecast errors of 19.6, 24.1 and 29.9%
  (basis alone) and 27.6, 32.8 and 38.1% (with the prior), for 1, 0.75 and
  0.5x. At 20 updates these are not results and set nothing.
- **Unchanged earlier fits.** The SVI loop was moved into a shared function
  (`benchmark_models._run_svi`). Refitting a declared case (every 8 days to
  day 84, seed 48) gives a bitwise-identical final state and operator file.
  The logged loss values jitter in the last float32 digit between two runs
  of the same code; the parameter updates do not. The fingerprints of all
  earlier cases are unchanged.

```bash
cd experiments/tumor
conda run -n prob_rom python benchmark_cohort.py build
conda run -n prob_rom python benchmark_cohort.py fit midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-prior
# likewise each -cohort-prior case (four population operators)
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-prior   # 48-50
# likewise every case in benchmark_cases.COHORT_CASES
```

### Results (each seed run once)

**The population prior passes one schedule, weekly scans to day 57 (8
scans), one of the two with the least data. It fails the other three, and
on both every-8-days schedules it makes the forecast much worse: 1.5-2.1
times the true burden on day 119.** All 28 fits finished (4 population
fits, 24 target fits), and every target fit has 64 of 64 draws. Median
point-forecast error (%) over seeds 48-50; the bar is 18.9% on every
scenario arm:

| Data to | Scans, rank | Basis, prior centre | 1x | 0.75x | 0.5x | Verdict |
|---|---|---|---|---|---|---|
| Day 84 | Weekly, 12, r4 | own, zero (learned trend, earlier) | 13.6 | 16.1 | 21.1 | fail |
| | | cohort, zero | 14.8 | 18.9 | 23.7 | fail |
| | | cohort, M | 16.2 | 18.7 | 21.4 | fail |
| | Every 8 days, 11, r4 | own, zero (earlier) | 13.6 | 12.8 | 17.8 | pass |
| | | cohort, zero | 10.7 | 13.8 | 20.0 | fail |
| | | cohort, M | 100.5 | 95.2 | 89.2 | fail |
| Day 57 | Weekly, 8, r3 | own, zero (earlier) | 20.1 | 26.7 | 32.1 | fail |
| | | cohort, zero | 25.4 | 31.5 | 35.6 | fail |
| | | cohort, M | 11.1 | 10.6 | 12.1 | **pass** |
| | Every 8 days, 8, r3 | own, zero (earlier) | 12.3 | 18.2 | 25.4 | fail |
| | | cohort, zero | 8.7 | 15.5 | 23.0 | fail |
| | | cohort, M | 45.0 | 43.6 | 45.4 | fail |

- **The seeds agree within 0.9 points.** On a fixed schedule they differ
  only in scanner noise.
- **Stress arms (not judged), 0.25x and 0x.** Cohort basis alone: 28.6 and
  33.1% (weekly to day 84), 26.7 and 33.0% (every 8 days to day 84), 36.7
  and 34.9% (weekly to day 57), 28.2 and 31.1% (every 8 days to day 57).
  Centred on M: 24.0 and 26.2%, 83.1 and 77.7%, 18.5 and 32.6%, 52.5 and
  66.5%.

**The population fits.** In-sample error is each member's scans replayed
by M from its first scan, against its clean coefficients. The burden ratio
is M's last-scan burden over the member's:

| Population fit | In-sample error per member 0-7, % (median) | Last-scan burden ratio per member | M: c_1, N_11 |
|---|---|---|---|
| Weekly, r4 (day-84 targets) | 15.6, 31.8, 12.1, 5.6, 43.5, 5.7, 28.8, 18.0 (16.8) | 0.57, 0.34, 1.46, 1.23, 19.4, 0.84, 0.42, 3.81 | 0.033, -0.951 |
| Every 8 days, r4 | 15.4, 20.8, 28.1, 19.5, 48.2, 14.0, 26.4, 15.0 (20.1) | 2.22, 0.59, 5.17, 2.50, 24.0, 2.17, 0.50, 3.46 | 0.016, -0.754 |
| Weekly, r3 (day-57 targets) | 14.5, 29.9, 12.5, 6.0, 45.6, 5.0, 26.3, 19.0 (16.7) | 0.80, 0.42, 2.06, 1.58, 23.4, 1.13, 0.50, 5.14 | 0.045, -0.963 |
| Every 8 days, r3 | 9.1, 28.2, 13.8, 9.3, 44.6, 4.6, 27.8, 22.0 (17.9) | 1.06, 0.44, 2.61, 1.78, 22.4, 1.36, 0.48, 6.10 | 0.020, -0.824 |

- **One operator cannot fit eight tumours.** The worst fit is member 4, the
  strongest responder (168 cells on day 119): M leaves it 19-24 times its
  last scan. Members 1 and 6, who regrow after a cut, are under-predicted
  (0.34-0.59). The every-8-days fits are worse: at rank 4 they over-predict
  all five planned-course members, by 2.2-5.2 times.
- **The two cohorts give different centres.** They are the same eight
  patients scanned on different days, yet M's mode-1 drug term is -0.95
  and -0.96 for weekly scans and -0.75 and -0.82 for every 8 days. Each
  cohort has its own basis, so the numbers are not strictly comparable;
  the forecasts below are.
- **M grows in proportion to size.** Its constant source is almost zero
  (c_1 0.016-0.045), and its linear part carries the growth (largest
  growth rate 0.010-0.033 per day). The target's fits centred on zero do
  the reverse: a constant source (c_1 0.55-1.22) and a linear part near
  zero (largest rate 0.002-0.006 per day in the cohort basis). Across
  eight tumours of different sizes, growth in proportion to size is the
  law that fits. Within one patient's narrow range of sizes the two look
  alike.
- **Against the expectation:** median errors of 16.7-20.1% are at the top
  of the guessed 5-20%, and the range is 4.6-48.2%. In three of the four
  fits members 4, 1 and 6 are the worst, as guessed, but member 4 at
  43-48% is well above the guess. In the every-8-days rank-4 fit, member 2
  (28.1%) is second.

What changed inside the target fits (ranges over seeds 48-50). tau is the
learned prior width of each block (c, A, N), as a multiple of the default
prior's scale. The dose effect is the share of the true day-119 difference
between 0.5x and 1x that the forecast gets. c_1 and N_11 in the own basis
are not strictly comparable with those in the cohort basis:

| Data to | Scans | Basis, centre | Training error | c_1 | N_11 | tau (c, A, N) | Dose effect at 0.5x | Day-119 burden / truth, 1x and 0.5x | 0.5x band holds truth |
|---|---|---|---|---|---|---|---|---|---|
| Day 84 | Weekly | own, zero | 4.5% | 0.68-0.69 | -0.73 | 1.5, 1.2, 1.5 | 0.38 | 0.85, 0.64 | 91% |
| | | cohort, zero | 4.7% | 0.55-0.56 | -0.72 | 1.2, 1.1, 1.6 | 0.43 | 0.74-0.75, 0.60-0.61 | 99% |
| | | cohort, M | 6.0% | 0.27 | -0.95 | 0.64, 0.50, 0.031 | 0.57 | 0.75, 0.67 | 100% |
| | Every 8 days | own, zero | 5.9% | 0.77 | -0.65 | 1.7, 1.1, 1.3 | 0.33 | 0.95, 0.68 | 100% |
| | | cohort, zero | 6.0% | 0.65-0.66 | -0.63 | 1.4, 1.2, 1.4 | 0.36 | 0.85, 0.63 | 100% |
| | | cohort, M | 14.9% | 0.11 | -0.75 | 0.47, 0.49, 0.015 | 1.27-1.28 | 2.13, 1.75-1.76 | 100% |
| Day 57 | Weekly | own, zero | 2.3% | 0.91-0.92 | -0.84 | 0.98, 0.88, 1.4 | 0.44 | 0.59-0.60, 0.50 | 24% |
| | | cohort, zero | 4.0% | 0.62-0.65 | -0.80 | 0.70, 0.84, 1.4 | 0.45 | 0.57-0.58, 0.49-0.50 | 18% |
| | | cohort, M | 5.5% | 0.19 | -0.96 | 0.26, 0.36, 0.050 | 0.90-0.91 | 1.03-1.04, 0.95 | 100% |
| | Every 8 days | own, zero | 2.4% | 1.21-1.22 | -0.84 | 1.3, 0.87, 1.3 | 0.44 | 0.75, 0.56 | 98-99% |
| | | cohort, zero | 3.6% | 1.14-1.15 | -0.83 to -0.82 | 1.2, 0.75, 1.4 | 0.45 | 0.82-0.83, 0.59 | 52-53% |
| | | cohort, M | 4.0% | 0.13 | -0.83 | 0.29, 0.43, 0.066 | 1.27 | 1.80-1.81, 1.47 | 100% |

- **The target keeps the population's drug term.** Centred on M, the
  learned width of the drug block falls to 0.015-0.067 (1.4-1.6 when
  centred on zero). After the fact (`cohort_blocks.py`): every fit's drug
  term N is within 1% of M's and its linear part within 5-29%; the
  constant source moves most (c_1 from M's 0.02-0.05 to 0.11-0.27). The
  whole operator stays 17-32% from M. The weak-form evidence from 8-12
  scans does not demand a different drug term, so the learned width
  shrinks until the two are equal, even where the fitted trajectory then
  misses the scans (every 8 days to day 84).
- **That helps when the population's drug term suits the target.** With
  the weekly M (mode-1 drug term -0.96), weekly scans to day 57 get the
  dose response right: the forecast has 0.90-0.91 of the true dose effect,
  0.95-1.04 of the true day-119 burden on every scenario arm, and the 0.5x
  band holds the truth all the time. The target's dense-scan fits in its
  own basis learn -0.95 to -0.99. To day 84 the weekly M changes little:
  0.5x stays at 21.4%, and 1x worsens to 16.2%, with the day-119 burden at
  0.75 of the truth.
- **It breaks the forecast when it does not.** With the every-8-days M
  (mode-1 drug term -0.75 and -0.82) the pinned kill is too weak. To day
  84 the fit cannot follow its own training scans (training error 14.9%,
  against 6.0% centred on zero): it runs above them from day 45. After the
  training window both every-8-days forecasts grow too fast, to 1.5-2.1
  times the truth by day 119. The dose effect is overstated (1.27-1.28).
- **M alone does not describe this patient** (after the fact,
  `cohort_M_alone.py`, no fits). M run from the target's first scan has a
  forecast error of 52-72% over the five arms and a training-window error
  of 22-29%. It forecasts 0.21-0.45 of the true day-119 burden: the
  cohort's tumours shrink much more than the target's, which grows and
  spreads faster and responds less.
- **The cohort basis alone moves the errors by -3.6 to +5.3 points.** It
  helps every 8 days to day 57 on every arm and every 8 days to day 84 at
  1x, and it hurts both weekly schedules on every arm. It turns the narrow
  pass (every 8 days to day 84) into a narrow fail: 0.5x goes from 17.8 to
  20.0%.
- **The cohort basis halves the projection floor from day 57.** From day
  84 the floor rises by 0.6-1.1 points (0.5x: 6.5 to 7.3-7.4%), because
  the cohort basis keeps a little less of the target's scans. From day 57
  it falls from 9.0-9.4 to 5.2-5.3% at 1x and from 13.6-14.1 to 7.9-8.0% at
  0.5x. The target's own scans stop at day 57, but the cohort was scanned
  to day 119, after cuts too, so its basis already holds the later shapes.
  This is the basis loss found in "Why the draw band misses after cuts".
  The forecast errors from day 57 do not follow: the fitted dynamics, not
  the floor, dominate them (errors of 9-36% against floors of 5-8%).
- **Bands.** Centred on M, the band holds the truth at every arm except the
  stress arms of weekly scans to day 84 (74% and 54%). At day 119 the 5-95%
  band spans 0.76-1.87 times the true burden at 0.5x and 0.96-2.40 times at
  1x. Weekly scans to day 57 give 0.82 and 1.37-1.39.

**Expectations against outcomes.**

| Case | Expected | Outcome |
|---|---|---|
| Population fits | In-sample errors of 5-20%, largest for members 1, 6 and 4 | Medians 16.7-20.1%, range 4.6-48.2%; largest for 4 (44-48%), then 1 and 6 |
| Cohort basis alone | Within a few points of the partners, no change of verdict | Within -3.6 to +5.3 points; one verdict changes (every 8 days to day 84: pass to fail, 0.5x 20.0%) |
| Centred on M, weekly to day 84 | Errors at 0.5x fall; may pass | 0.5x unchanged (21.4%), 1x worse (16.2%); fails |
| Centred on M, every 8 days to day 84 | Stays a pass | Fails badly: 100.5, 95.2 and 89.2% |
| Centred on M, to day 57 | Uncertain, leaning fail, burden too low | Weekly passes (11.1, 10.6 and 12.1%) with the burden right; every 8 days fails with the burden too high (1.5-1.8 times) |

**What this means.** A population prior can rescue the hardest schedule,
but only by handing the target the population's drug response. The
target's own 8-12 scans do not move it. Whether the forecast is then right
depends on whether that one operator happens to suit this patient. Here it
did for one cohort and failed badly for the other, though both hold the
same eight patients. That is not robust enough to report as a fix.

Two parts of this design are to blame, and both are fixable:
- **M is one operator fitted to all eight members.** It fits them with
  5-48% error, and it changes with the scan days.
- **The target alone sets how far it may move from M.** With so few scans,
  the learned width of the drug block collapses to almost zero.

Idea 4, partial pooling, is the next in the agreed order and changes
exactly these two parts. Each past patient gets their own operator around
a shared mean, all fitted together with the target. The spread between the
past patients then sets how far the target may move from the mean, instead
of the target's own few scans. It needs new code and will be declared
before any fit.

## Partial pooling of past patients (rule declared before any fit)

**Why.** The population prior failed for two reasons, both found in its
results:
- **One operator for eight tumours.** M fitted the members with 5-48%
  error, and its drug term changed with the scan days (-0.95 weekly, -0.75
  to -0.82 every 8 days).
- **The target alone set how far it may move from M.** From 8-12 scans
  the learned width of the drug block collapsed to 0.015-0.067, so the
  target took M's drug term whether or not it suited this patient.

Partial pooling changes exactly these two parts. It is the standard way to
combine patients in a hierarchical model: each patient keeps their own
rates, and the patients together say how much rates differ between
patients.

**The model.** Every patient k gets their own operator, the population
operator plus their own deviation: O_k = M + δ_k. The target is k = 0;
members 0-7 are k = 1-8. All nine patients' scans are fitted at once.
- **M** is the population operator. It is not fixed in advance but
  learned with everything else. Prior: each entry ~ N(0, (κ_M,b s_j)²).
- **δ_k** is how patient k differs from M. Prior: each entry
  ~ N(0, (κ_δ,b s_j)²), the same width for every patient.
- **κ_M,b and κ_δ,b** are learned for each block b (c, A, N), with the
  log-normal(0, 3²) prior every block width has had so far. κ_δ is the
  spread between patients. The eight members' full courses (14-17 scans
  each, three with a dose cut) mostly set it, so the target's few scans
  no longer decide how far the target may move.
- **The target's forecast** uses draws of O_0 = M + δ_0, as before. Its
  own scans inform δ_0 directly; the members' scans inform it through M
  and κ_δ.
- **Unchanged for each patient:** the Gaussian process of their scans,
  its hyperparameters (including the learned trend scale) and the
  tempering exponents. The column scales s_j come from all nine patients
  together: T is their mean training window (105 days in the smoke run,
  against the target's 79), and S and U are the RMS of all their scans and
  inputs. So κ is not on quite the same scale as in earlier sections.
- **Exact.** The evidence and the operator posterior keep their closed
  forms. Each mode's unknowns become [M, δ_0, ..., δ_8], and patient k's
  rows see M + δ_k (`core/weakform_opinf/model.py`,
  `build_model(pooled=True)`). Tests
  (`core/weakform_opinf/tests/test_partial_pooling.py`):
  - with one patient it equals the single fit with width
    √(κ_M² + κ_δ²);
  - for two patients, the evidence and the posteriors of M, of each
    patient and of the target match a brute-force Gaussian calculation
    (changing κ_δ by 1%, or swapping κ_M and κ_δ, fails this test);
  - as κ_δ → 0 it becomes one operator shared by all;
  - unsupported settings are refused.

**Design.** Four cases (`benchmark_cases.POOLED_CASES`), one per realistic
schedule. They use the same acquisitions, cohort, cohort basis and rank as
the `-cohort-basis` and `-cohort-prior` cases. The direct control is
`-cohort-basis`: the same basis, with the target fitted alone. Medians over
seeds 48-50 at 1 / 0.75 / 0.5x:

| Target schedule | Rank | Cohort protocol | Learned trend, own basis | Cohort basis, centred on zero | Centred on M |
|---|---|---|---|---|---|
| Weekly to day 84 (12 scans) | 4 | weekly | 13.6 / 16.1 / 21.1, fail | 14.8 / 18.9 / 23.7, fail | 16.2 / 18.7 / 21.4, fail |
| Every 8 days to day 84 (11) | 4 | every 8 days | 13.6 / 12.8 / 17.8, pass | 10.7 / 13.8 / 20.0, fail | 100.5 / 95.2 / 89.2, fail |
| Weekly to day 57 (8) | 3 | weekly | 20.1 / 26.7 / 32.1, fail | 25.4 / 31.5 / 35.6, fail | 11.1 / 10.6 / 12.1, pass |
| Every 8 days to day 57 (8) | 3 | every 8 days | 12.3 / 18.2 / 25.4, fail | 8.7 / 15.5 / 23.0, fail | 45.0 / 43.6 / 45.4, fail |

Case names: `midcourse-reduced-trend-chemo-{weekly,every8}-scans-learned-trend-cohort-pooled`
(day 84) and `reduced-regimen-trend-chemo-...-cohort-pooled` (day 57).

**Rule.**
1. **Runs.** Seeds 48-50 of the four cases, each run once with the
   production recipe (12,000 updates). Nothing is tuned. The cohort is the
   one built for the population prior, unchanged; its population fits are
   not used.
2. **Metric and arms.** The same metric. Scenario arms are 1, 0.75 and
   0.5x; 0.25x and 0x are stress arms, reported and not judged.
3. **Bar.** A case passes if every scenario arm's median is at most 18.9%.
4. **Reading.** Partial pooling works on a schedule if its case passes.
   Set against `-cohort-basis`, it shows what pooling adds. It is a fix
   for realistic imaging only if all four schedules pass. Otherwise the
   results are reported schedule by schedule, beside the earlier ideas.
5. **Also reported.**
   - The learned widths κ_M and κ_δ of each block, against the population
     prior's collapsed 0.015-0.067.
   - M's c_1 and N_11, and how far the target's operator is from M in each
     block.
   - Each member's in-sample error with their own operator M + δ_k, and
     with M alone, replayed from their first scan as for the population
     fits.
   - The target fits: c_1 and N_11, the training-window error, the share of
     the true dose effect and the day-119 burden.
   - How often the 0.5x and 1x bands hold the truth, and their widths.

**Expectations, written before any fit (but after the smoke run below).**
- **Members.** With their own operators, in-sample errors well below the
  population fits' 5-48%: most under 10%. The largest deviations from M
  belong to members 4, 1 and 6.
- **Spread.** κ_δ of the drug block is well above 0.015-0.067. The
  members' drug responses differ (sensitivity 0.24-0.45, three cuts), so
  the target's drug term is free to move.
- **To day 84.** Weekly: 0.5x may fall below the bar; uncertain. Every 8
  days: near the `-cohort-basis` control (0.5x 20.0%), not the 89-100% of
  the fixed centre, because M is now the centre of nine patients' own
  operators instead of one compromise operator. A narrow call either way.
- **To day 57.** With four treated scans the target's deviation is weakly
  determined, so the forecast leans on M. Weekly: near the fixed centre's
  pass, perhaps a little worse. Every 8 days: much better than 45%, but
  leaning fail at 0.5x.
- **Bands.** Wider than with the fixed centre, as they now include the
  spread between patients.

**Disclosures.**
- **The same best case as before.** One anatomy, one baseline tumour and a
  narrow spread of patients (log SD 0.3), as disclosed for the population
  prior. The members' scans cover days 5-119, so the cohort has seen the
  target's forecast window, cuts included. The target's own future is not
  in the data.
- **Choices made before any fit, with no alternative tried.**
  - One spread κ_δ per block, shared by all nine patients: the target is
    treated like any member.
  - The log-normal(0, 3²) prior for both widths.
  - Column scales from all nine patients.
  - The members' full courses.
  - This cohort and its basis.
- **Seeds.** Seeds 48-50 have now been used by every sparse-scan idea. On a
  fixed schedule they differ only in scanner noise (the population prior's
  seeds agreed within 0.9 points), so they do not test another patient. If
  pooling is adopted, a confirmation should use a different virtual target,
  declared first.
- **Smoke run.** One smoke run checked the code end to end, written outside
  the results tree. It used development seed 45, weekly scans to day 84,
  and 20 training updates instead of 12,000.
  - It printed forecast errors of 9.0, 9.8 and 17.7% (1, 0.75 and 0.5x), and
    26.3 and 34.4% (0.25x and 0x). 3-6 of the 64 draws per arm were within
    25%.
  - After 20 updates the widths were κ_M 1.4, 2.1 and 1.0, and κ_δ 9.9,
    1.1 and 0.8 (c, A, N).
  - At 20 updates these are not results and set nothing. They were,
    however, seen before the expectations above were written.
- **Cost.** One update takes about 19 ms, so one fit takes about 4 minutes
  (about 1 minute for the target alone). This was timed on the smoke run's
  data before declaring.
- **Unchanged earlier fits.** The fingerprints of all 43 earlier cases are
  unchanged. With this code, refits of a learned-trend case (every 8 days to
  day 84, seed 48) and of a population-prior case (weekly to day 57, seed
  48) reproduce their stored operator files bit for bit.

```bash
cd experiments/tumor
conda run -n prob_rom python benchmark_cohort.py build   # the population prior's cohort (cached, unchanged)
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort-pooled   # 48-50
# likewise every case in benchmark_cases.POOLED_CASES
```

### Results (each seed run once)

**Partial pooling passes two of the four schedules, both with weekly
scans. It passes to day 84 (7.2, 8.4 and 10.8% at 1, 0.75 and 0.5x), which
no earlier idea passed, and to day 57 (12.4, 12.0 and 15.1%). It fails
both every-8-days schedules, so by the declared reading it is not a fix for
realistic imaging.** Its bands hold the truth at all times on every arm of
every schedule, the failures and the stress arms included. All 12 fits
finished, and every arm has 64 of 64 draws. Median point-forecast error
(%) over seeds 48-50; the bar is 18.9% on every scenario arm:

| Data to | Scans, rank | Basis, prior centre | 1x | 0.75x | 0.5x | Verdict |
|---|---|---|---|---|---|---|
| Day 84 | Weekly, 12, r4 | own, zero (learned trend, earlier) | 13.6 | 16.1 | 21.1 | fail |
| | | cohort, zero (the control, earlier) | 14.8 | 18.9 | 23.7 | fail |
| | | cohort, M fixed (earlier) | 16.2 | 18.7 | 21.4 | fail |
| | | cohort, partial pooling | 7.2 | 8.4 | 10.8 | **pass** |
| | Every 8 days, 11, r4 | own, zero (earlier) | 13.6 | 12.8 | 17.8 | pass |
| | | cohort, zero (earlier) | 10.7 | 13.8 | 20.0 | fail |
| | | cohort, M fixed (earlier) | 100.5 | 95.2 | 89.2 | fail |
| | | cohort, partial pooling | 21.4 | 19.0 | 17.5 | fail |
| Day 57 | Weekly, 8, r3 | own, zero (earlier) | 20.1 | 26.7 | 32.1 | fail |
| | | cohort, zero (earlier) | 25.4 | 31.5 | 35.6 | fail |
| | | cohort, M fixed (earlier) | 11.1 | 10.6 | 12.1 | pass |
| | | cohort, partial pooling | 12.4 | 12.0 | 15.1 | **pass** |
| | Every 8 days, 8, r3 | own, zero (earlier) | 12.3 | 18.2 | 25.4 | fail |
| | | cohort, zero (earlier) | 8.7 | 15.5 | 23.0 | fail |
| | | cohort, M fixed (earlier) | 45.0 | 43.6 | 45.4 | fail |
| | | cohort, partial pooling | 14.3 | 16.2 | 22.4 | fail |

- **The seeds agree within 0.3 points.**
- **Against the direct control** (the same basis, the target fitted
  alone), pooling lowers the error on eight of the twelve scenario arms,
  including 0.5x on every schedule. It raises it on the 1x and 0.75x arms
  of both every-8-days schedules: from 10.7 to 21.4% and from 13.8 to 19.0%
  to day 84, and from 8.7 to 14.3% and from 15.5 to 16.2% to day 57.
- **Stress arms (not judged), 0.25x and 0x:** 13.8 and 17.2% (weekly to day
  84), 17.4 and 18.8% (every 8 days to day 84), 24.8 and 43.4% (weekly to
  day 57), 34.1 and 53.3% (every 8 days to day 57). From day 57 the
  forecast without drug regrows too fast: on day 119 it is 1.26 and
  1.37-1.38 times the true burden at 0x.

**Run note.** The first launch ran six fits at once. A pooled fit needs
about 3.7 GB of memory, so the machine ran short of memory and disk, and I
stopped the launch. One fit had reached 2,000 of its 12,000 updates. None
had finished or been evaluated, and nothing from them was looked at. Their
partial folders were deleted, and all 12 fits were then run from scratch,
two at a time. A fit took 7.2-11.2 minutes, plus 6-8 minutes of
evaluation. The declared estimate, about 4 minutes, was timed with one fit
running alone.

**The pooled fits** (ranges over seeds 48-50). κ is a multiple of the
column scales, which come from all nine patients. "Target from M" is
‖X_target − X_M‖ / ‖X_M‖ for each block X:

| Data to, scans | κ_M (c, A, N) | κ_δ (c, A, N) | κ_δ,N 5-95% | M: c_1, N_11 | Fixed centre (population prior): c_1, N_11 | ‖M − M_fixed‖ / ‖M_fixed‖ | Target from M (c, A, N) |
|---|---|---|---|---|---|---|---|
| Day 84, weekly | 0.24, 5.5, 2.5 | 0.034, 0.57, 0.010 | 0.001-0.030 | 0.063, -0.84 | 0.033, -0.95 | 0.55 | 4, 8, 0% |
| Day 84, every 8 days | 0.20, 6.1, 2.0 | 0.023-0.024, 0.75, 0.004 | 0.0004-0.011 | 0.037, -0.69 | 0.016, -0.75 | 0.46 | 2, 8, 0% |
| Day 57, weekly | 0.24, 4.0, 3.4 | 0.045, 0.23, 0.018 | 0.002-0.058 | 0.056-0.057, -0.93 | 0.045, -0.96 | 0.41 | 4, 2, 0% |
| Day 57, every 8 days | 0.19, 5.4, 3.2 | 0.036, 0.24, 0.016 | 0.001-0.054 | 0.029, -0.88 | 0.020, -0.82 | 0.52 | 3, 1, 0% |

- **The spread of the drug block collapsed again, against the
  expectation.** κ_δ,N is 0.004-0.018, below even the population prior's
  0.015-0.067. The spread between patients went into the linear block A
  (κ_δ 0.23-0.75) and a little into the constant source c (0.023-0.045).
  Every member's drug term equals M's (distance 0.00 in every fit), and so
  does the target's.
- **Every patient's operator is close to M.** The members' differ from it
  by 0-11% in A, 1-25% in c and 0% in N. The target's differs by 1-8%, 2-4%
  and 0%, and by 0-2% over the whole operator. Member 4, the strongest
  responder, is furthest from M in every fit, in c (13-25%) or in A (11%).
  Members 1 and 6 are not among the furthest.
- **M is not the population prior's fixed centre.** The two differ by
  41-55%. M still changes with the scan days, as the fixed centre did: its
  mode-1 drug term is -0.84 for weekly scans and -0.69 for every 8 days at
  rank 4 (-0.93 and -0.88 at rank 3).

**The members' in-sample errors** (%), each member's scans replayed from
their first scan by their own operator M + δ_k and by M alone (seed 48;
the seeds agree within 0.4 points):

| Pooled fit | Own operator, members 0-7 (median) | M alone (median) | Population fit, earlier (median) |
|---|---|---|---|
| Day 84, weekly (r4) | 6.0, 9.6, 7.2, 4.2, 31.5, 6.4, 8.0, 11.5 (7.6) | 6.5, 25.1, 14.7, 13.6, 50.2, 7.6, 23.6, 27.2 (19.2) | 16.8 |
| Day 84, every 8 days (r4) | 8.4, 12.4, 8.9, 6.0, 19.5, 6.2, 7.1, 8.1 (8.2) | 16.3, 21.0, 29.7, 19.5, 51.9, 14.7, 20.4, 20.3 (20.3) | 20.1 |
| Day 57, weekly (r3) | 10.0, 14.9, 8.5, 5.1, 57.9, 4.1, 17.1, 20.4 (12.4) | 8.2, 23.6, 14.0, 13.6, 53.2, 6.8, 20.9, 26.9 (17.4) | 16.7 |
| Day 57, every 8 days (r3) | 8.5, 13.1, 8.0, 6.6, 56.2, 4.8, 17.2, 20.1 (10.8) | 6.3, 21.5, 16.2, 14.7, 50.7, 7.3, 21.1, 25.4 (18.7) | 17.9 |

- With their own operators, the median error falls to 7.6-8.2% at rank 4
  and 10.8-12.4% at rank 3, from 17-20% for M alone or for the population
  fits. Six of the eight members are under 10% at rank 4, but only three
  or four at rank 3.
- **Member 4 is still badly fitted:** 20-58%, and it ends at 9-37 times its
  last scan. At rank 3 its own operator does worse than M alone (56-58%
  against 51-53%), and so does member 0's (8.5-10.0% against 6.3-8.2%).
  The fit matches each member's smoothed scans in the weak form, not their
  replayed trajectory, so a member's own operator need not replay better.

**The target fits** (ranges over seeds 48-50). The dose effect is the
share of the true day-119 difference between 0.5x and 1x that the forecast
gets. The band width is the 5-95% band's width on day 119, as a share of
the true burden:

| Data to | Scans | Training error | c_1 | N_11 | Dose effect at 0.5x | Day-119 burden / truth, 1x and 0.5x | 1x band: holds truth, width | 0.5x band: holds truth, width |
|---|---|---|---|---|---|---|---|---|
| Day 84 | Weekly | 3.3% | 0.06 | -0.84 | 0.78 | 0.93, 0.86-0.87 | 100%, 0.70-0.71 | 100%, 0.61-0.62 |
| | Every 8 days | 5.7% | 0.04 | -0.69 | 0.74-0.75 | 1.08-1.09, 0.93-0.94 | 100%, 0.86 | 100%, 0.70 |
| Day 57 | Weekly | 6.4% | 0.06 | -0.93 | 0.97 | 0.99-1.00, 0.98 | 100%, 1.06 | 100%, 0.83 |
| | Every 8 days | 4.9% | 0.03 | -0.88 | 1.08-1.09 | 1.18, 1.12 | 100%, 1.16 | 100%, 0.99 |

- **The target is forecast with M, run from its own scans.** Its operator
  is within 0-2% of M.
- **On the weekly schedules M suits the target.** To day 84 the forecast
  gets 0.78 of the true dose effect (0.38 with the learned trend alone,
  0.57 with the fixed centre) and 0.86-0.93 of the true day-119 burden. To
  day 57 it gets 0.97 of the dose effect and 0.98-1.00 of the burden.
- **On the every-8-days schedules it does not.** To day 84, M's kill is
  weaker (drug term -0.69, against -0.84 weekly), so on the full course the
  forecast does not fall enough. It ends at 1.08-1.09 times the true
  burden, and its largest errors are at 1x and 0.75x. From day 57 every arm
  grows too fast: 1.12-1.18 times the truth on day 119 on the scenario
  arms, and the dose effect is overstated (1.08-1.09).
- **The bands hold the truth at all times,** on every arm of every
  schedule, stress arms included. Against the expectation, they are
  narrower than with the fixed centre, or about the same. On day 119 they
  are 0.61-0.99 of the true burden wide at 0.5x (fixed centre 0.76-1.87)
  and 0.70-1.16 at 1x (fixed centre 0.96-2.40).

**After the fact: what pooling learned about the patients**
(`pooled_members_check.py`, no fits; seeds 48 and 50 agree). The members
differ in growth rate (0.025-0.069 per day), in drug sensitivity
(0.24-0.45) and in three dose cuts. The ROM's input carries each member's
doses but not their sensitivity, so only δ_k can carry it.
- **In every pooled fit, the nine patients' whole-tumour (mode-1) entries
  come out almost the same:** c_1 within 0.004, N_11 within 0.0001 and
  A_11 within 0.0006-0.0055 per day.
- **The members' true net rates are far more spread.** Under the full
  course, growth minus sensitivity times the mean exposure (ignoring
  crowding) runs from -0.068 to -0.001 per day, a spread of 0.067. Their
  fitted net mode-1 rates (A_11 plus N_11 times the mean input) span only
  0.0006-0.0055 per day. They do not follow the true ones either: the
  correlation across the members is -0.31 to +0.08.
- **So pooling learned a common centre and how each member's shape
  changes, but not that the patients grow and respond at different
  rates.** Why the members' 14-17 scans each do not set their own rates was
  not tested.

**Expectations against outcomes.**

| Item | Expected | Outcome |
|---|---|---|
| Members, own operators | Most under 10%, well below the population fits' 5-48% | Medians 7.6-8.2% at rank 4 (6 of 8 under 10%) and 10.8-12.4% at rank 3 (3-4 of 8); member 4 still 20-58% |
| Furthest from M | Members 4, 1 and 6 | Member 4 in every fit; not 1 and 6. All are close: A 0-11%, c 1-25%, N 0% |
| Spread of the drug block, κ_δ,N | Well above 0.015-0.067 | 0.004-0.018, lower still; the target's drug term is M's |
| Weekly to day 84 | 0.5x may pass; uncertain | Passes on every arm: 7.2, 8.4 and 10.8% |
| Every 8 days to day 84 | Near the control (0.5x 20.0%); a narrow call | 0.5x better (17.5%), but 1x 21.4% and 0.75x 19.0%: fails |
| Weekly to day 57 | Near the fixed centre's pass, perhaps a little worse | Passes, a little worse: 12.4, 12.0 and 15.1% (fixed centre 11.1, 10.6 and 12.1%) |
| Every 8 days to day 57 | Much better than 45%, leaning fail at 0.5x | 14.3, 16.2 and 22.4%: fails at 0.5x |
| Bands | Wider than with the fixed centre | Narrower or about the same; they hold the truth at all times on every arm |

**What this means.**
- **Partial pooling is the strongest of the declared sparse-scan ideas,
  but not a fix.** It passes two of the four schedules, including weekly
  scans to day 84 for the first time, and its bands always hold the truth.
  It fails both every-8-days schedules, so by the declared reading it is
  not a fix for realistic imaging.
- **It fixed the first problem named in the declaration, not the second.**
  M is now the centre of nine patients' own operators, not one compromise
  operator, and the members are fitted better. But the spread between the
  patients' drug terms was again learned as almost zero. After the fact,
  the patients share practically the same whole-tumour growth and kill. So
  the target is again forecast with the cohort's average rates. That suits
  it with the weekly cohort and not with the every-8-days cohort, and
  nothing in the target's own 8-12 scans decides which.
- **No one version passes everywhere.** Across all the declared ideas,
  three of the four schedules have a version that passes: weekly scans to
  day 84 with pooling, every 8 days to day 84 with the learned trend alone,
  and weekly scans to day 57 with the fixed centre or with pooling. No
  version passes more than two, and every 8 days to day 57 has none.
  Choosing a version per schedule after seeing the results would not be an
  honest fix.
- **Seeds.** Seeds 48-50 differ only in scanner noise, and every idea has
  now used them. Any claim for pooling would need a confirmation on another
  virtual target, declared first.

This completes the agreed order: ideas 1, 2 and 5, then the population
prior, then partial pooling. No further sparse-scan fit is planned. What
to report is an open decision.

## More past patients (design written before any fit)

**Why.** On the training-data plot the new patient's tumour ends above all
eight past patients. It is not like them: relative to the population it
grows faster (+0.8 SD), spreads faster (+2.0 SD) and responds less to the
drug (-0.6 SD). Pooling at K = 8 past patients forecast it with the
cohort's average rates. This sweep asks what more past patients do. More
patients tell the model more about how patients differ, but they also pin
the average more firmly. No patient is picked to resemble the new one: the
past patients are the first K of one random draw.

**The draw, before any fit.** The draw continues the eight-patient
cohort's draw (same seed, centre and spread), so past patients 0-7 are the
same eight as before. Compared with the new patient (k, D and s in
population SDs; computed from the draw alone):

| K | Cohort mean minus the new patient (k, D, s) | Grow faster | Spread faster | Respond less | All three |
|---|---|---|---|---|---|
| 4 | -0.4, -2.1, +1.0 | 1 of 4 | 0 | 0 | 0 |
| 8 | -0.9, -2.2, +1.1 | 2 of 8 | 0 | 0 | 0 |
| 16 | -0.8, -2.2, +1.1 | 5 of 16 | 0 | 3 of 16 | 0 |
| 32 | -0.8, -2.2, +0.8 | 7 of 32 | 0 | 6 of 32 | 0 |
| 64 | -1.0, -2.1, +0.8 | 12 of 64 | 0 | 14 of 64 | 0 |

None of the 64 spreads as fast as the new patient (largest D 0.0954,
against 0.0965). More past patients add faster growers and weaker
responders, but never one like the new patient.

**Design** (`benchmark_cases.COHORT_SWEEP_DESIGN`, `COHORT_SWEEP_CASES`;
`benchmark_cohort.py build-sweep`).
- **Past patients.** 64, drawn as above. Patient m has patient (m mod 8)'s
  dose change, so 3 in every 8 have a cut (1 of 4 at K = 4). Each has a
  weekly scan from day 5 + (m mod 7) to day 119 (16-17 scans), with the
  same scanner and segmentation as before.
- **K = 4, 8, 16, 32 and 64:** the first K past patients. Each K has its
  own POD basis from those patients' scans: rank 4, checked against the
  noise threshold when the cohort is built. The new patient's scans are
  only projected onto it.
- **New patient:** as before. Weekly scans to day 84 (12 scans), the
  learned trend scale, seed 48 only. Seeds differ only in scanner noise,
  and the pooled seeds agreed within 0.1 points.
- **Two fits per K.** "Pooled": jointly with the K past patients, as at
  K = 8. "Basis alone": the new patient alone on the same basis, which
  shows what the basis does by itself. That makes 10 fits with the
  production recipe, each run once, with nothing tuned.
- **References, already run (seed 48):** the new patient alone on its own
  basis (13.6 / 16.1 / 21.1% at 1 / 0.75 / 0.5x), alone on the K = 8
  basis (14.8 / 18.9 / 23.8%), and K = 8 pooled with the earlier solver
  (7.1 / 8.4 / 10.8%).
- **Solver.** The earlier pooled calculation solves for all patients'
  operators at once, so its cost grows with K³; one K = 64 fit would take
  about a day. The "blocked" solver gives the same answer one patient at
  a time, at a cost that grows with K (`build_model(pooled_solver=
  "blocked")`). Tests show that the evidence, its gradient and every
  posterior equal the earlier solver's to rounding, for one to three
  patients. K = 8 is rerun with it (both fits), as a check against the
  earlier K = 8 fits. It will not be equal bit for bit: the basis is
  computed from sparse scans, so it differs by rounding (measured when the
  cohort is built), and the fits run in float32.
- **K = 64 only if it is practical.** A smoke run times it first
  (development seed 45, 20 updates; not a result). K = 64 is run if one
  fit takes under about 4 hours. If it needs more than about 4 GB of
  memory, it runs alone. If it is not run, it is reported as not run.

**Reported.** The same metric and arms: 1, 0.75 and 0.5x, plus 0.25x and
0x. The 18.9% bar is drawn for reference. There is no pass or fail rule
for the sweep.
- Error against K, pooled and basis alone, beside the references.
- The day-119 burden, how often the bands hold the truth, and their width.
- The learned spreads κ_δ per block (at K = 8 the drug block's collapsed
  to 0.004-0.018), and how far the new patient's operator is from M.
- Each past patient's in-sample error with their own operator.
- The share of the new patient's scans each basis captures.

**Expectations (written before any fit).**
- **K = 4** is worse than K = 8: fewer patients and a basis from fewer
  scans.
- **Beyond K = 8** the error stays about the same or gets worse. If
  pooling again learns mainly the average, a firmer average pulls the
  forecast toward slower spread and a stronger drug effect. An uncertain
  call: more patients with different responses could also make κ_δ
  larger and let the new patient move away from the average.
- **Basis alone** improves a little with K, since its basis has seen more
  tumour shapes.

```bash
cd experiments/tumor
conda run -n prob_rom python benchmark_cohort.py build-sweep   # 64 past patients, five bases (cached)
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort16-pooled --seeds 48
# likewise every case in benchmark_cases.COHORT_SWEEP_CASES
```

## Open decisions

The following are not done and await a decision: the Neural-ODE ensembles on
this design (about 2-3 hours per seed), tutorial-style figures, and moving the
manuscript's tumour section and dose description to this design. With the
free model, the only levers were more information in training (a longer
window, or a scan after a recovery period). The after-the-fact check shows
that the model change works on the original window. The reported cases
remain the manuscript's chemo results until then.

For the mid-course design, the Neural ODE should get the same known physics
for a like-for-like comparison. That means the same drug term, and no growth
without tumour. Its exact form (a hybrid with a learned remainder) is still
to be decided.

Which model and design to report are also open. The free ROM with the
input-aware GP passes on the protocol's dose cuts, so the manuscript's model
could stay as it is. The projected form is more accurate, but it adds a
model section. With the projected form the earlier tutorial-schedule
question works too, and it is the more realistic one: data
to the second MRI, forecast to the third. Its model was chosen on the
mid-course development seeds, not on seeds 48-50. Those seeds had, however,
been seen with the free model. A cleaner confirmation, declared before it is
run, would be the chosen model on the standard regimen's seeds 51-53 (1x),
which have not been run with it.

Whether to build the POD from smoothed maps is open too. If adopted, the
smoothing width, the variant and the rank rule would be declared first and
tested on fresh seeds. The variants are smoothed maps, smooth modes, or the
production modes blurred.

The realistic-scan result bears on all of these choices. Every chemo result
here assumes about 120 scans in the training window, far denser than
clinical MRI. The declared fixes (known noise, scans every 8 days, two
modes) did not change that. A learned trend scale for the Gaussian process
did, for one schedule: scans every 8 days to day 84 pass, narrowly and with
wide bands. Weekly scans to day 84 and both schedules to day 57 still fail.
A population prior from a virtual cohort passes weekly scans to day 57,
but it breaks both every-8-days schedules, so it is not a robust fix.
Partial pooling of the past patients' scans with the target passes both
weekly schedules, to day 84 for the first time, and its bands always hold
the truth. It fails both every-8-days schedules, so it is not a fix either.
No version passes more than two of the four schedules, and every 8 days to
day 57 has none. The agreed list of ideas is done. The options are open:
- state the dense-imaging assumption in the paper;
- report the sparse-scan results as a limitation: the learned trend scale
  (one schedule passes) and partial pooling (both weekly schedules pass),
  beside the failures;
- test the projected form on the realistic schedules, declared first;
- move the example to a setting with frequent imaging.

Run notes, the pre-declared rule, logs, and plots
(`figures/reported_burden.png`, `figures/dev45_burden.png`,
`figures/why_2x_looks_best.png`, `figures/midcourse_eval_burden.png`,
`figures/midcourse_eval_burden_all_seeds.png`,
`figures/midcourse_eval_error_vs_time.png`, `figures/smoother_seed45.png`,
`figures/what_changed_seed48.png`, `figures/free_dose_mod_burden.png`,
`figures/band_miss_seed48.png`, `figures/four_shapes_seed48.png`,
`figures/pod_limits_seed48.png`, `figures/pod_fix_seed48.png`,
`figures/blur_fit_burden.png`, `figures/smoothing_profile_seed48.png`,
`figures/blurred_modes_burden.png`, `figures/scan_schedule_burden.png`,
`figures/scan_schedule_error.png`, `figures/scan_sweep_burden.png`,
`figures/sparse_fix_burden.png`, `figures/sparse_fix_error.png`,
`figures/gp_mode1_seed48.png`, `figures/gp_trend_seed48.png`,
`figures/gp_trend_seed45.png`, `figures/learned_trend_burden.png`,
`figures/learned_trend_error.png`, `figures/cohort_prior_burden.png`,
`figures/cohort_prior_error.png`, `figures/pooled_burden.png`,
`figures/pooled_error.png`, the band-coverage scripts, the
smoothed-POD scripts and fits, the scan-schedule check, summary and sweep
scripts, the sparse-fix check, summary and Gaussian-process check scripts,
the learned-trend summary script, the cohort data check, summary,
M-alone and block scripts, and the pooled summary and member-check
scripts) are local session artifacts, not
bundled with the repository:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-tutorial-runs/`.
