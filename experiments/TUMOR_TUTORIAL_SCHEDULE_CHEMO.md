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

Run notes, the pre-declared rule, logs, and plots
(`figures/reported_burden.png`, `figures/dev45_burden.png`,
`figures/why_2x_looks_best.png`, `figures/midcourse_eval_burden.png`,
`figures/midcourse_eval_burden_all_seeds.png`,
`figures/midcourse_eval_error_vs_time.png`, `figures/smoother_seed45.png`,
`figures/what_changed_seed48.png`, `figures/free_dose_mod_burden.png`,
`figures/band_miss_seed48.png`, `figures/four_shapes_seed48.png`,
`figures/pod_limits_seed48.png`, `figures/pod_fix_seed48.png`,
`figures/blur_fit_burden.png`, `figures/smoothing_profile_seed48.png`,
`figures/blurred_modes_burden.png`, the band-coverage scripts, and the
smoothed-POD
scripts and fits) are local session artifacts, not bundled with the
repository:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-tutorial-runs/`.
