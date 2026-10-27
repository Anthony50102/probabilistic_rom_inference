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
- In both, the target takes the cohort's drug term (how it responds to
  the dose). That suits it with the weekly cohort and not with the
  every-8-days cohort. *Corrected after the cohort-size sweep; this point
  first said the target takes the cohort's whole-tumour growth and kill.
  With pooling its growth is its own.*

No version passes all four realistic schedules. See "A population prior
from past patients" and "Partial pooling of past patients".

**More past patients: no steady gain.** Pooling with 4, 8, 16, 32 and 64
past patients (weekly scans to day 84, seed 48, declared before the fits)
beats the new patient alone at every K. The error is lowest at K = 8-16
(6.2, 6.7 and 10.4% at K = 16) and back near K = 4 by K = 64 (11.6, 13.9
and 19.0%). Every patient shares the cohort's drug term, and with more
patients everyone is fitted worse. See "More past patients".

**The variant round: no clear winner on the benchmark's protocol.** Seven
variants (pooling, a drug factor per patient, and the projected form, alone
or pooled) ran on the twin and nine new virtual patients, on four realistic
schedules (136 fits on an Azure machine, declared before the fits). The
declared rule picked E, the projected form fitted to the new patient alone,
on four development patients. On six fresh patients E is low on every one
(mean 39.2%). Pooling (B) gives the best typical forecast (median 16.8%)
but fails strong responders. On sparse scans no variant is best on all
three schedules, and only 10 of 90 patient means are under 18.9%. See
"Variant round".

**Forecasting from the last scan: most of the error was the starting
point.** The benchmark integrates every forecast from the first scan
(day 5), so any miss on the training scans carries into the forecast.
Starting instead from the last training scan, as a clinic would, needs no
refit. It was found after the fact and declared before 48 more fits. It
lowers every variant's error on every schedule. G (the projected form,
pooled, with a drug factor per patient) then has mean errors of 12.6,
12.1, 19.7 and 18.9% on weekly, fortnightly and 4-weekly scans to day 84
and weekly scans to day 57. It never forecasts a negative tumour, and it is
under the 18.9% bar for 9, 9, 4 and 6 of the ten patients. B (pooled) has
16.2, 16.2, 18.5 and 17.7. Three of the five declared expectations were
not met: on the six new patients G ties with B, C and F (11.0-11.8), B is
lower on 4-weekly scans and on weekly scans to day 57, and fortnightly
scans do as well as weekly ones. Still left: too little regrowth after a dose cut, and bands
that hold the truth less than 90% of the time. See "Restarting the
forecast from the last scan".

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
  growth rate 0.023-0.029 per day). The target's fits centred on zero do
  the reverse: a constant source (c_1 0.55-1.22) and a linear part near
  zero (largest rate 0.003-0.007 per day in the cohort basis). Across
  eight tumours of different sizes, growth in proportion to size is the
  law that fits. Within one patient's narrow range of sizes the two look
  alike. *Corrected after the cohort-size sweep: the script read the
  wrong columns for the linear part (see the correction at the end of
  "More past patients"). The two rates first read 0.010-0.033 and
  0.002-0.006.*
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
  term N is within 1% of M's and its linear part within 6-29%
  (*corrected as above; first read 5-29%*); the
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
‖X_target − X_M‖ / ‖X_M‖ for each block X (its A column is corrected
after the cohort-size sweep; it first read 8, 8, 2 and 1%):

| Data to, scans | κ_M (c, A, N) | κ_δ (c, A, N) | κ_δ,N 5-95% | M: c_1, N_11 | Fixed centre (population prior): c_1, N_11 | ‖M − M_fixed‖ / ‖M_fixed‖ | Target from M (c, A, N) |
|---|---|---|---|---|---|---|---|
| Day 84, weekly | 0.24, 5.5, 2.5 | 0.034, 0.57, 0.010 | 0.001-0.030 | 0.063, -0.84 | 0.033, -0.95 | 0.55 | 4, 9-10, 0% |
| Day 84, every 8 days | 0.20, 6.1, 2.0 | 0.023-0.024, 0.75, 0.004 | 0.0004-0.011 | 0.037, -0.69 | 0.016, -0.75 | 0.46 | 2, 9, 0% |
| Day 57, weekly | 0.24, 4.0, 3.4 | 0.045, 0.23, 0.018 | 0.002-0.058 | 0.056-0.057, -0.93 | 0.045, -0.96 | 0.41 | 4, 4, 0% |
| Day 57, every 8 days | 0.19, 5.4, 3.2 | 0.036, 0.24, 0.016 | 0.001-0.054 | 0.029, -0.88 | 0.020, -0.82 | 0.52 | 3, 4, 0% |

- **The spread of the drug block collapsed again, against the
  expectation.** κ_δ,N is 0.004-0.018, below even the population prior's
  0.015-0.067. The spread between patients went into the linear block A
  (κ_δ 0.23-0.75) and a little into the constant source c (0.023-0.045).
  Every member's drug term equals M's (distance 0.00 in every fit), and so
  does the target's.
- **Every patient's operator is close to M.** The members' differ from it
  by 1-12% in A, 1-25% in c and 0% in N. The target's differs by 4-10%, 2-4%
  and 0%, and by 0-2% over the whole operator. Member 4, the strongest
  responder, is furthest from M in every fit, in c (13-25%) or in A (12%).
  No other member is furthest in every fit; the order of the rest changes
  with the fit and the block. *Corrected after the cohort-size sweep: the
  A distances first read 0-11%, 1-8% and 11%, and this point ended
  "Members 1 and 6 are not among the furthest". With the right columns,
  member 1 is second or third in A at rank 3, and member 6 is third in A
  at rank 4.*
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

- **The target shares M's drug term; the rest of its operator is its
  own.** Its operator is within 0-2% of M, with the same drug term. The
  small rest matters: forecast with M alone, the target would be 57% off
  at 1x (weekly to day 84, seed 48), not 7.1%
  (`cohort_size_M_forecast.py`). *Corrected after the cohort-size sweep;
  this point first read "The target is forecast with M, run from its own
  scans."*
- **On the weekly schedules the cohort's drug term suits the target.** To
  day 84 the forecast gets 0.78 of the true dose effect (0.38 with the
  learned trend alone, 0.57 with the fixed centre) and 0.86-0.93 of the
  true day-119 burden. To day 57 it gets 0.97 of the dose effect and
  0.98-1.00 of the burden.
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
- **In every pooled fit, the nine patients' whole-tumour (mode-1) source
  and drug entries come out almost the same** (c_1 within 0.004, N_11
  within 0.0001), **but their growth entries differ.** A_11 spans
  0.007-0.023 per day across the nine patients. Across the members it
  follows the true growth rate (correlation with log k +0.89 to +0.94),
  not the drug sensitivity (-0.34 to 0.00).
- **The members' fitted net rates follow the true ones, squeezed.** Under
  the full course, growth minus sensitivity times the mean exposure
  (ignoring crowding) runs from -0.068 to -0.001 per day, a spread of
  0.067. The fitted net mode-1 rates (A_11 plus N_11 times the mean input)
  follow it (correlation +0.71 to +0.91) but span only 0.006-0.021 per
  day. A_11 carries the differences in growth; no mode-1 entry carries
  those in sensitivity. Replayed over their scans with their own
  operators, the members' net rates (change of log burden per day,
  against the clean scans') tell the same story: correlation +0.59 on
  seeds 48 and 50, spanning 0.026 per day against the true 0.040
  (`cohort_size_member_rates.py`).
- **So pooling learned that the patients grow at different rates.** What
  it shares is the drug term, so each patient's response to a change of
  dose is the cohort's. Why the drug term's spread collapses was not
  tested.
- *Corrected after the cohort-size sweep.* The first version of
  `pooled_members_check.py` read the wrong column for A_11 (A_12, how
  mode 2 feeds mode 1). It reported A_11 within 0.0006-0.0055 per day,
  fitted net rates spanning 0.0006-0.0055 per day and correlations of
  -0.31 to +0.08, and concluded: "So pooling learned a common centre and
  how each member's shape changes, but not that the patients grow and
  respond at different rates." That conclusion was wrong.

**Expectations against outcomes.**

| Item | Expected | Outcome |
|---|---|---|
| Members, own operators | Most under 10%, well below the population fits' 5-48% | Medians 7.6-8.2% at rank 4 (6 of 8 under 10%) and 10.8-12.4% at rank 3 (3-4 of 8); member 4 still 20-58% |
| Furthest from M | Members 4, 1 and 6 | Member 4 in every fit; 1 and 6 only in some fits and blocks. All are close: A 1-12%, c 1-25%, N 0% (*corrected; first read A 0-11% and "not 1 and 6"*) |
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
  patients' drug terms was again learned as almost zero, so every patient,
  the target included, gets the cohort's drug term. The weekly cohort's
  suits the target and the every-8-days cohort's does not, and nothing in
  the target's own 8-12 scans decides which. *Corrected after the
  cohort-size sweep; this point first said that the patients share
  practically the same whole-tumour growth and kill, and that the target
  is forecast with the cohort's average rates. Their net growth differs;
  only the drug term is shared.*
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

### Amendment before any sweep fit: the blocked solver in batches

**What happened.** The K = 64 smoke run was stopped at 23 GB of memory,
before its first update. A stage profile on the smoke data showed why. The
program was written out once per past patient, so compiling it took
1.7 minutes at K = 8 and 14 minutes at K = 16, growing about as K^3.3. At
K = 64 that is about 16 hours and well over 16 GB. The updates themselves
were fast (13 and 41 ms each).

**The change.** Past patients with the same number of scans (12, 16 or 17)
are now computed together as one batch, so the program has at most three
batches whatever K is. The Gaussian-process settings (length scale,
variance, noise and trend scale) are drawn as one array over patients and
modes, not one number per patient and mode. The model, its priors, the
equations and the approximation (an independent normal for each log
setting) are unchanged. Only the blocked solver changes, and only the
sweep uses it.

**Checks.**
- Tests: the evidence, its gradient and every posterior equal the earlier
  (stacked) solver's to rounding, for one to four patients. This also holds
  with the drug input, the learned trend scale and tempering
  (`core/weakform_opinf/tests/test_partial_pooling.py`).
- The real K = 8 data (seed 48), at that fit's mean settings: in float64,
  the log density agrees to 2e-16 and the gradients to 1e-12. In float32,
  as the fits run, they agree to rounding (log density 2e-6).
- Fits that do not use the blocked solver are unchanged. Three updates of
  four existing fits give the same parameters bit for bit. The printed loss
  differs by 1-4 units in the last place from run to run, as it did before.
- What does change is the random stream: the starting point and each
  update's noise are drawn differently, as with another fit seed. So the
  K = 8 rerun checks the stacked K = 8 fits only up to such differences.

**New profile** (development seed 45, 20 updates; not a result):

| K | Compile | Peak memory | Per update | 12000 updates |
|---|---|---|---|---|
| 8 | 6 s | 2.6 GB | 13 ms | 2.6 min |
| 16 | 6 s | 2.6 GB | 29 ms | 5.7 min |
| 64 | 8 s | 3.5 GB | 171 ms | 34 min |

A K = 64 smoke run through the whole pipeline (fit, export and
evaluation) took 7.5 minutes and peaked at 3.7 GB. One K = 64 fit
therefore takes about 40 minutes. That is under the declared 4 hours, so
K = 64 runs. It stays below 4 GB, so it need not run alone; it is queued
last, at most two fits at a time. No sweep fit had run when this was
written.

### Results (seed 48, each fit run once)

All ten fits finished, with 64 of 64 draws on every arm. The pooled fits
took 3, 5, 11, 23 and 35 minutes at K = 4, 8, 16, 32 and 64 (the K = 64
job 41 minutes with preparation and evaluation). Summary script
`cohort_size_summary.py` (log `logs/cohort_size_summary.txt`); figures
`figures/cohort_size_error.png`, `figures/cohort_size_burden.png` and
`figures/cohort_size_cohort.png`, with the new patient's scans as black
stars.

**Error (%), days 84-119,** at 1 / 0.75 / 0.5x, with the stress arms
0.25 / 0x in brackets:

| K | Pooled | New patient alone on the K-patient basis |
|---|---|---|
| 4 | 11.9 / 14.1 / 17.0 (20.1 / 23.1) | 12.7 / 16.8 / 21.8 (27.0 / 31.7) |
| 8 | 7.2 / 8.4 / 10.8 (13.8 / 17.2) | 14.8 / 18.9 / 23.8 (28.7 / 33.2) |
| 16 | 6.2 / 6.7 / 10.4 (14.9 / 19.5) | 13.8 / 17.9 / 22.9 (27.8 / 32.2) |
| 32 | 9.9 / 9.5 / 13.9 (19.6 / 25.4) | 13.4 / 17.5 / 22.5 (27.6 / 32.2) |
| 64 | 11.6 / 13.9 / 19.0 (24.5 / 29.9) | 15.0 / 19.2 / 24.0 (28.8 / 33.2) |
| Reference | K = 8, stacked solver: 7.1 / 8.4 / 10.8 (13.7 / 17.1) | Own basis: 13.6 / 16.1 / 21.1 (26.8 / 32.1) |

- **Pooling beats the new patient alone at every K, on every arm.**
- **The pooled error is lowest at K = 8-16 and rises after that.** At
  K = 64 it is back near K = 4, and worse at 0.5x and on the stress arms.
  At 0.5x, K = 64's 19.0% is just above the 18.9% bar; the sweep has no
  pass or fail rule.
- **Alone on the K-patient basis, the error does not change with K:**
  12.7-15.0% at 1x.

**The pooled forecasts.** The day-119 burden and the band width are shares
of the true burden. The dose effect is the share of the true day-119
difference between 0.5x and 1x that the forecast gets. The burden error is
over days 84-119:

| K | Day-119 burden at 1 / 0.75 / 0.5x | Dose effect at 0.5x | Bands hold the truth at 1 / 0.75 / 0.5 / 0.25 / 0x | 0.5x band width | Burden error at 1 / 0.75 / 0.5x | Training error (days 5-84) |
|---|---|---|---|---|---|---|
| 4 | 0.84 / 0.77 / 0.73 | 0.58 | 100 / 100 / 100 / 99 / 68% | 0.57 | 10.7 / 15.2 / 19.8% | 5.3% |
| 8 | 0.93 / 0.89 / 0.87 | 0.78 | 100% on every arm | 0.70 | 3.3 / 6.2 / 9.0% | 3.2% |
| 16 | 0.96 / 0.89 / 0.83 | 0.66 | 100% on every arm | 0.58 | 1.9 / 6.1 / 10.9% | 4.3% |
| 32 | 0.85 / 0.83 / 0.81 | 0.75 | 100 / 100 / 93 / 79 / 75% | 0.40 | 8.1 / 10.2 / 12.5% | 4.9% |
| 64 | 0.97 / 0.93 / 0.90 | 0.81 | 100 / 100 / 100 / 100 / 93% | 0.43 | 6.1 / 5.2 / 6.0% | 6.2% |

Alone on the K-patient basis, at every K: day-119 burden 0.74-0.80 /
0.66-0.71 / 0.60-0.63, dose effect 0.42-0.44, and 0.5x band width
0.75-0.96.
- **The bands narrow as K grows** (0.5x width 0.57-0.70 up to K = 16,
  0.40-0.43 from K = 32). At K = 32 they start to miss the truth.
- **K = 64 gets the number of cells right but not where they are.** Its
  day-119 burden and dose effect are the closest of all, and its burden
  error (5-6%) is as small as at K = 8-16. Its field error is the largest,
  19.0% at 0.5x.

**The pooled fits.** In-sample errors replay each past patient's scans
from their first scan, as before:

| K | κ_δ (c, A, N) | M: c_1, N_11 | Past patients, own operator: median (first eight) | M alone: median | New patient's scans the basis misses |
|---|---|---|---|---|---|
| 4 | 0.030, 0.37, 0.013 | 0.03, -0.82 | 7.5% | 17.6% | 4.5% |
| 8 | 0.036, 0.57, 0.009 | 0.07, -0.84 | 7.6% (7.6%) | 19.1% | 3.7% |
| 16 | 0.040, 0.50, 0.004 | 0.25, -0.83 | 11.0% (8.6%) | 20.8% | 4.0% |
| 32 | 0.057, 0.69, 0.003 | 0.30, -0.74 | 11.5% (9.8%) | 24.6% | 3.7% |
| 64 | 0.077, 0.75, 0.0004 | 0.23, -0.67 | 12.4% (13.3%) | 25.5% | 3.5% |

The share the basis misses is of the scans' norm; the new patient's own
basis misses 2.6%.

**K = 8 solver check.** Rerun at K = 8 with the batched blocked solver,
the pooled fit is within 0.09 points of the stacked K = 8 fit on every arm
(κ_δ 0.036, 0.574, 0.009 against 0.034, 0.572, 0.010). The basis-alone
fit, which does not use that solver, agrees to two decimals.

**Expectations against outcomes.**

| Expected | Outcome |
|---|---|
| K = 4 is worse than K = 8 | Yes: 11.9 / 14.1 / 17.0% against 7.2 / 8.4 / 10.8% |
| Beyond K = 8, about the same or worse, through a firmer average with slower spread and a stronger drug effect. Uncertain: a larger κ_δ could let the new patient move away from the average | About the same at K = 16 (6.2 / 6.7 / 10.4%), worse at K = 32 and 64. κ_δ grew for c and A, but the drug block's fell 20-fold (0.009 to 0.0004). M's drug term got weaker, not stronger (N_11 -0.84 to -0.67) |
| Basis alone improves a little with K | No: no trend (12.7-15.0% at 1x), though the basis misses less of the new patient's scans (4.5% to 3.5%) |

**After the fact, no new fits: why more patients do not help**
(`cohort_size_M_forecast.py`, `cohort_size_member_rates.py`; logs of the
same names).
- **Every patient gets the cohort's drug term.** κ_δ,N falls from 0.013
  at K = 4 to 0.0004 at K = 64. The patients differ only in the linear
  block A and the source c, which do not scale with the dose.
- **The rest of the new patient's operator is its own, and it matters.**
  It is about 1% from M overall, within the past patients' range at every
  K. In A it is 6-10% from M: the furthest of the nine patients at K = 8,
  within the past patients' range at the other K. Forecast with M alone,
  the new patient's error would be 50-57% at 1x at every K, against 6-12%
  with its own operator.
- **The past patients get their own net growth, squeezed more as K
  grows.** Replayed over their scans with their own operators, their net
  rates (change of log burden per day) follow the true ones: correlation
  +0.98 at K = 4 and +0.59 to +0.68 at K = 8-64. But the fitted rates span
  95% of the true spread at K = 4, 66% at K = 8 and 41% at K = 64. They
  reach only -0.017 to -0.024 per day, against true rates down to -0.038
  to -0.055.
- **Everyone is fitted worse as K grows.** The first eight past patients'
  median in-sample error is 7.6, 8.6, 9.8 and 13.3% at K = 8, 16, 32 and
  64. The new patient's training error rises from 3.2% (K = 8) to 6.2%
  (K = 64).

**What this means.**
- **More past patients do not help this patient beyond about 8-16.**
  Pooling beats the patient alone at every K. Choosing K = 16 from these
  results would be after the fact; a claim would need K declared first and
  new virtual patients.
- **The shared drug term is the likely limit.** The past patients differ
  in drug sensitivity, but the fit gives them all one drug term. With more,
  and more varied, patients, each patient's own growth is squeezed harder
  toward the centre, and everyone is fitted worse, the new patient
  included. That the drug term is the cause is a reading of these
  diagnostics, not a test.
- **A fix to try would let the drug response vary by patient,** for
  example one sensitivity factor per patient on the shared drug term. Not
  run; it would be declared first.
- **One seed, one draw of past patients and one new patient.** Rerunning
  K = 8 with a different random stream moved the errors by up to 0.09
  points. Another draw of past patients was not tried.

**Corrections to earlier results,** found while analysing the sweep and
corrected in place, marked:
- **The target is not forecast with M** (partial pooling). It shares M's
  drug term, but the rest of its operator is its own, and that matters.
- **A wrong column in the after-the-fact scripts.** The scripts that split
  an operator into blocks (`cohort_blocks.py`, `pooled_summary.py`,
  `pooled_members_check.py`) took the linear block A from columns 2 to
  r + 1 of the operator instead of 1 to r. They dropped A's first column
  and added the input column, which is zero here. The fits, forecasts,
  errors, the c and N blocks and the whole-operator distances are
  unaffected. Corrected: the A distances from M (population prior and
  partial pooling), M's largest growth rate (population prior) and the
  members' A_11 (partial pooling). The last changes a conclusion: pooling
  did learn that the patients grow at different rates. The opening
  summary's point on what the target takes from the cohort is corrected
  to match.

This section's "Why", and its expectation "if pooling again learns mainly
the average", rest on the earlier statements. As part of the design
written before any fit, they are left as written.

## Variant round: a drug factor per patient, the projected form and new patients (design written before any fit)

**Why.** Two weaknesses remain after the sweep, and both have a known fix
to try.
- **Every patient gets the cohort's drug term.** In TumorTwin the drug
  kills the same share of cells everywhere: the term is -s C(t) u, with
  s the patient's sensitivity. On any POD basis it therefore reduces to
  -(s / 0.23) α(t) q, where α(t) is the exposure input in the twin's
  units (s = 0.23). Patients differ by one number, s / 0.23. The pooled
  fits shared one drug term (N_11 -0.84 at K = 8). That is weaker than
  every past patient's true kill (median -1.36 at K = 8), and the new
  patient's is -1.00. A learned factor per patient on the drug input lets
  each patient's kill differ.
- **The free form regrows in a straight line.** With dense scans, the
  projected PDE form (logistic crowding) was 1.2-2.4 times more accurate
  after dose cuts (mid-course design: 5.8 / 4.3 / 3.8%). It has not been
  tried with sparse scans or with pooling.
- **Only one new patient so far.** The new patient (the tutorial twin) is
  out of distribution: it spreads faster than all 64 past patients. So
  this round adds nine virtual new patients from the past patients' own
  population, each declared before any fit.

**The two model changes** (`core/weakform_opinf/model.py`).
- **Drug factor per patient** (`build_model(..., input_scale=True)`,
  blocked solver only). Patient k's input is multiplied by its own learned
  factor exp(η_k), with η_k = ω z_k, z_k ~ N(0, 1) and the spread
  ω ~ log-normal(log 0.5, 1). So, a priori, patients' factors typically
  differ by ×0.6 to ×1.6. The true sensitivities have a log SD of 0.3. Given
  the factors, the operators are still solved in closed form. The
  exported operator of patient k is the one for inputs in the twin's units:
  its drug columns are multiplied by exp(η_k). M is the operator at η = 0.
  The factors are learned from the scans; nothing about s is given.
- **The projected form with pooling.** Each row keeps the linear part A_i
  free. Two shapes come from projecting the PDE onto the basis: crowding
  (the projected u², rate h_i) and the drug term (q_i α, rate n_i). The
  model learns one rate on each shape. That is r + 2 unknowns per mode
  (6 at rank 4), against 9 for the free form, which has c, A and N. As
  in the mid-course design (`structure="galerkin_logistic"`). With
  pooling, M and every δ_k are over these coefficients, each with the
  κ_M and κ_δ of its block.
- **Tests** (`core/weakform_opinf/tests/test_partial_pooling.py`):
  - the blocked solver with the projected rows matches a brute-force
    Gaussian calculation (evidence and posteriors);
  - identity row maps reproduce the unstructured solver;
  - fixed factors give exactly the fit with the inputs scaled by them;
  - unsupported settings and missing or misshapen factors are refused.
  A deliberately wrong factor (the drug column not scaled) fails these
  tests.

**The new patients, drawn before any fit**
(`benchmark_cases.VIRTUAL_PATIENT_DESIGN`, `VIRTUAL_PATIENTS`;
`benchmark_cohort.virtual_patient_parameters`). They are drawn from the
past patients' population: the same centre and log SD 0.3, from a new
seed (202), rounded to four significant figures. Position z is in
population SDs (k, D, s). The kill in the twin's units is s / 0.23; the
pooled fits' shared value was 0.84 at K = 8.

| Patient | Role | k | D | s | z (k, D, s) | Kill (s / 0.23) |
|---|---|---|---|---|---|---|
| Twin | development | 0.0521 | 0.0965 | 0.23 | +0.79, +2.03, -0.60 | 1.00 |
| vp1 | development | 0.07078 | 0.04212 | 0.199 | +1.81, -0.73, -1.09 | 0.87 |
| vp2 | development | 0.03644 | 0.03709 | 0.4333 | -0.40, -1.15, +1.51 | 1.88 |
| vp3 | development | 0.05352 | 0.06346 | 0.4207 | +0.88, +0.64, +1.41 | 1.83 |
| vp4 | confirmation | 0.03432 | 0.03534 | 0.3059 | -0.60, -1.31, +0.35 | 1.33 |
| vp5 | confirmation | 0.0451 | 0.04055 | 0.3237 | +0.31, -0.86, +0.54 | 1.41 |
| vp6 | confirmation | 0.06393 | 0.04466 | 0.2922 | +1.47, -0.53, +0.19 | 1.27 |
| vp7 | confirmation | 0.04558 | 0.06064 | 0.3746 | +0.34, +0.49, +1.02 | 1.63 |
| vp8 | confirmation | 0.04904 | 0.07976 | 0.2162 | +0.59, +1.40, -0.81 | 0.94 |
| vp9 | confirmation | 0.03568 | 0.06703 | 0.2414 | -0.47, +0.82, -0.44 | 1.05 |

- The roles follow the draw's order: the first three pick the variant,
  the other six only confirm it.
- Only k, D and s differ. Each new patient has the twin's anatomy,
  baseline tumour, planned course (no dose change), scanner and
  acquisition seed. Their ROM inputs are in the twin's units (0.23), as
  in a clinic where the sensitivity is not known. The simulator uses
  their true s.

**Variants** (`benchmark_cases.VARIANT_ARMS`, `VARIANT_CASES`;
`variant_name(source, K, arm, patient)`). All use the past patients of the
sweep (weekly scans, days 5-119) and their K-patient basis:

| Variant | Name | Fit | Form | Drug factor |
|---|---|---|---|---|
| A | `cohort8-basis` | new patient alone, K = 8 basis | free | no |
| B | `cohort8-pooled` | pooled with K = 8 | free | no (the sweep's model) |
| C | `cohort8-pooled-drug` | pooled with K = 8 | free | yes |
| D | `cohort64-pooled-drug` | pooled with K = 64 | free | yes |
| E | `cohort8-projected-basis` | new patient alone, K = 8 basis | projected | no |
| F | `cohort8-projected-pooled` | pooled with K = 8 | projected | no |
| G | `cohort8-projected-pooled-drug` | pooled with K = 8 | projected | yes |

**Schedules** (the new patient's scans; same ranks as before):
- weekly scans to day 84 (12 scans, rank 4);
- fortnightly to day 84 (7, rank 4);
- 4-weekly to day 84 (4, rank 3);
- weekly to day 57 (8, rank 3).

The fortnightly and 4-weekly schedules get the learned trend scale too
(`VARIANT_LEARNED_TREND_CASES`). Rank 3 means the first three modes of the
same basis.

**Stage 1: choose the variant.** Weekly scans to day 84, seed 48, all
seven variants on the four development patients: 28 fits. The twin's A and
B were already run on the Mac (14.8 / 18.9 / 23.8% and
7.2 / 8.4 / 10.8%). They are rerun on Azure to measure what the platform
changes.
- **Choice.** For each patient, average the error over the three scenario
  arms (1, 0.75 and 0.5x; days 84-119). The variant with the lowest mean
  over the four patients wins. If a simpler variant is within 0.5 points
  of the best, the simpler one wins. From simplest:
  A < E < B < F < C < G < D. Alone comes before pooled, pooled before
  pooled with factors, free before projected, and K = 8 before 64.
- A variant with a failed fit (non-finite) on a development patient cannot
  be chosen. Failed fits are reported, not rerun.

**Stage 2: confirm and widen.** The winner, B and A:
- weekly to day 84 on the six confirmation patients;
- fortnightly to day 84, 4-weekly to day 84 and weekly to day 57 on all
  ten patients.
That is up to 108 fits, each run once with seed 48.

**Reported** (no pass or fail rule; the 18.9% bar is drawn for
reference).
- Error at 1, 0.75 and 0.5x (0.25 and 0x as stress arms), per patient and
  as the mean over patients.
- The day-119 burden against the truth, how often the bands hold the
  truth, and their width.
- The learned factors exp(η) of every patient, past and new, against the
  true s / 0.23, and the spread ω.
- M's drug term N_11 and each new patient's effective kill.
- The twin's A and B on Azure against the Mac.

**Expectations (written before any fit).**
- **Drug factor (C against B).** Better on the strong responders vp2 and
  vp3 (kill 1.8-1.9, against B's shared 0.84), where B's shared drug term
  is furthest off. About the same on vp1 (0.87).
- **Factors.** The past patients' learned factors rank their true
  sensitivities (rank correlation 0.7 or more across the eight). The new
  patients' factors move from 1 toward the truth for at least three of the
  four.
- **K = 64 with factors (D).** No longer worse than K = 8 (C), since the
  shared drug term was the likely cause of the sweep's rise. An uncertain
  call.
- **Projected form (E, F, G).** Better at 0.5x than the free variants at
  the same K. Also uncertain: from 12 sparse scans the crowding rate may
  not be pinned down.
- **Winner.** C or G.
- **The twin stays the hardest patient,** being out of distribution.

**Disclosures.**
- **A best case for the drug factor.** It has exactly the simulator's
  structure: the drug kills uniformly, and s scales the kill. A real
  tumour's response varies in space and over time (resistance). The factor
  does not change the free form's straight-line regrowth.
- **A best case for pooling.** All patients share one anatomy and one
  baseline tumour, and the new ones also share the twin's scanner,
  acquisition seed and planned course. vp1-9 are in the past patients'
  population by construction. The twin is not.
- **Seen before.** The twin's A and B on weekly scans to day 84 (the
  sweep). Every other variant, patient and schedule is new.
- **One seed per patient.** Seeds differ only in scanner noise; the pooled
  seeds agreed within 0.1-0.3 points.
- **Platform.** The runs are on an Azure Linux virtual machine (x86) with
  the same package versions. The repository and data sit at the same paths
  as on the Mac, because the sweep's record stores absolute paths. The
  twin's simulated data are copied from the Mac (checked by SHA-256). The
  new patients' data are simulated on the virtual machine. Fits are not
  equal bit for bit across platforms; the twin's reruns measure the
  difference.
- **Smoke runs before stage 1** (development seed 45, 20 updates, a
  separate output folder; not results). They check C, D, F and G end to
  end, including a new patient's simulation.
- **Unchanged earlier cases.** All 60 earlier cases' fingerprints are
  unchanged (checked against the previous commit). The twin's two sweep
  cases keep their definitions; the variant grid reuses them. Tests: core
  52 and tumour 77 pass.

```bash
cd experiments/tumor
# new patients' simulations are generated on first use (pre-generated once per patient for parallel runs)
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-weekly-scans-learned-trend-cohort8-pooled-drug-vp2 --seeds 48
# likewise every stage-1 case: variant_name(source, K, arm, patient) in benchmark_cases
```

### Results (seed 48, each fit run once)

**Run log.**
- **Machine.** An Azure Standard_F16as_v6 virtual machine (16 cores,
  64 GB, Linux x86), from 15:58 UTC. Core 52 and tumour 77 tests pass on
  it. The twin's simulated data were copied from the Mac (21 files,
  SHA-256 checked). Each new patient's simulation was generated on the
  machine once, before any fit.
- **Smoke runs** (seed 45, 20 updates, a separate folder; not results):
  C-G on vp2, and one variant on each other schedule (fortnightly C on
  vp4, 4-weekly G on vp5, weekly to day 57 E on vp6 and D on vp7). All
  finished with 64 draws on every arm; peak memory 2.1-5.4 GB per fit.
- **Stage 1** started at 16:21 UTC, 12 fits at a time. Each fit started
  about 80 threads, which overloaded the machine. From 16:33 each new fit
  was pinned to one free core (`taskset`); a few ran unpinned when no
  core was free. Pinning changes only how many threads a fit uses.
- **The twin's D** (K = 64) ran at about 0.9 s per update on one core. It
  was stopped after 2250 s, past update 2000 and before writing any
  output. It was resumed from its saved state at update 2000 on four
  cores (the runner's own resume; it reported 38 minutes after resuming).
- **Stage 2 started early.** A and B are run whatever wins, so their 72
  fits started at 16:57, before the choice. E's 36 fits started at 17:46,
  before the twin's D had finished. With three of the four patients, D's
  mean was already at least 101.5, so D could not win. The twin's D
  finished at 18:15; D's mean is 104.5.
- **Every fit finished** (exit 0), apart from the stopped run above.
  Every arm evaluated 64 draws. Stage 2 finished at 19:53 UTC.
- **Draws cut off at the safety threshold** (each counts as a miss in the
  band). One draw in seven weekly-to-day-84 fits: E on vp2 and vp3; F on
  vp1, vp3 and the twin (0x only); G on vp1 and vp3. Many more for the
  projected form fitted alone on sparse scans: E lost 2-14 of its 64
  draws in all ten 4-weekly fits and 3-9 in six of the ten fortnightly
  fits. A lost one draw (vp2, weekly to day 57, 1x). The free pooled
  variants (B, C, D) lost none.
- Summary script `variant_summary.py` (log `logs/variant_summary.txt`);
  figures `figures/variant_stage1_error.png`,
  `figures/variant_stage1_burden.png`,
  `figures/variant_stage1_burden_focus.png`,
  `figures/variant_stage2_weekly84_error.png`,
  `figures/variant_stage2_weekly84_burden.png` and
  `figures/variant_stage2_<schedule>_error.png` for the sparser
  schedules, with the new patient's scans as black stars. They are local
  session artifacts, like the earlier ones, in
  `.../files/azure-variants/`.

**Platform.** On Azure, the twin's A and B agree with the Mac's sweep runs
to 0.001 points on every arm (A 14.80 / 18.92 / 23.81 / 28.70 / 33.17%,
B 7.21 / 8.43 / 10.78 / 13.76 / 17.18%).

#### Stage 1: choosing the variant

**Error (%), days 84-119,** at 1 / 0.75 / 0.5x, with the mean of the three
in brackets:

| Patient | A alone | B pooled | C pooled + factor | D K = 64 + factor | E projected alone | F projected pooled | G projected pooled + factor |
|---|---|---|---|---|---|---|---|
| Twin | 14.8 / 18.9 / 23.8 (19.2) | 7.2 / 8.4 / 10.8 (8.8) | 7.7 / 8.8 / 11.0 (9.2) | 10.2 / 10.3 / 15.4 (12.0) | 27.4 / 30.5 / 34.5 (30.8) | 8.6 / 10.2 / 13.2 (10.7) | 8.4 / 9.8 / 12.5 (10.2) |
| vp1 | 11.0 / 11.4 / 15.7 (12.7) | 10.6 / 13.4 / 17.0 (13.6) | 21.4 / 23.1 / 25.7 (23.4) | 8.0 / 13.0 / 19.3 (13.4) | 7.7 / 10.7 / 15.4 (11.3) | 16.8 / 16.8 / 18.3 (17.3) | 11.3 / 11.8 / 13.6 (12.2) |
| vp2 | 2993 / 2622 / 2138 (2584) | 447 / 428 / 385 (420) | 316 / 302 / 270 (296) | 331 / 328 / 306 (322) | 59.6 / 51.9 / 44.4 (51.9) | 299 / 280 / 243 (274) | 424 / 400 / 351 (392) |
| vp3 | 108.5 / 113.3 / 115.5 (112.4) | 160.0 / 141.0 / 111.7 (137.5) | 9.0 / 9.2 / 18.9 (12.4) | 79.0 / 73.5 / 60.7 (71.1) | 46.4 / 52.0 / 59.8 (52.7) | 100.4 / 84.3 / 62.1 (82.3) | 81.7 / 87.3 / 91.8 (86.9) |
| **Mean** | 682.1 | 145.0 | 85.2 | 104.5 | **36.7** | 96.1 | 125.2 |
| Median | 65.8 | 75.6 | 17.9 | 42.2 | 41.4 | 49.8 | 49.6 |

Stress arms, 0.25 / 0x:

| Patient | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| Twin | 28.7 / 33.2 | 13.8 / 17.2 | 13.9 / 17.3 | 21.7 / 27.9 | 38.8 / 42.7 | 16.7 / 19.9 | 15.6 / 18.6 |
| vp1 | 21.0 / 26.1 | 21.3 / 26.7 | 29.4 / 34.2 | 25.9 / 32.7 | 20.3 / 24.8 | 21.0 / 24.4 | 16.4 / 19.5 |
| vp2 | 1587 / 1065 | 317 / 235 | 220 / 161 | 261 / 201 | 41.9 / 47.2 | 189 / 128 | 279 / 196 |
| vp3 | 114.6 / 111.7 | 76.9 / 45.1 | 31.3 / 42.8 | 43.4 / 28.6 | 67.9 / 74.8 | 39.6 / 28.8 | 94.3 / 95.2 |

**Choice: E,** the projected form fitted to the new patient alone on the
K = 8 basis. Its mean is 36.7; no simpler variant is within 0.5 points.
- **A mixed picture.** E has the lowest mean and the smallest worst case
  (52.7%). But on the twin it is the second worst (30.8%, against 8.8% for
  B). C has the lowest median (17.9%).
- **Two strong responders decide the mean.** vp2 and vp3 have kills of
  1.88 and 1.83, almost twice the twin's. By day 84 their tumours have
  shrunk to 5% and 12% of their peak, and they keep shrinking: on day 119
  vp2 has 126-442 cells, against 9,753-17,439 for the twin. The error is
  relative to the true field over days 84-119, so it becomes very large
  when the true tumour is that small.

**What the forecasts do on the strong responders** (burden in cells: at
the cut, day 84, then on day 119 at 1 / 0.75 / 0.5x):

| Patient | Truth | E | C | B |
|---|---|---|---|---|
| vp2 | 801, then 126 / 236 / 442 | 961, then 243 / 322 / 435 | 2,615, then 904 / 1,273 / 1,780 | 3,221, then 1,499 / 1,984 / 2,627 |
| vp3 | 2,182, then 657 / 1,199 / 2,176 | 1,431, then 289 / 414 / 595 | 2,076, then 564 / 878 / 1,344 | 4,192, then 2,618 / 3,376 / 4,378 |

- **On vp2 every pooled variant already misses the last scans.** They
  start the forecast at 2.6-4.0 times the true burden, and their training
  errors are 14-23% (E: 12.8%). Pooling pulls the patient toward the past
  patients, whose shared drug term is weaker. Alone, the free form (A)
  fails outright on vp2: 64% training error, and it forecasts growth.
- **On vp3 the drug factor works:** C fits the scans (8.6%) and forecasts
  within 9-19%, though with too little regrowth at 0.5x (0.62 of the
  truth). E starts low (0.66 of the truth at the cut) and regrows too
  little (0.27-0.44).
- **A and G forecast a negative burden on vp3** after about day 100,
  which is impossible.

**The other forecast measures.** Day-119 burden as a share of the truth, at
1 / 0.75 / 0.5x:

| Variant | Twin | vp1 | vp2 | vp3 |
|---|---|---|---|---|
| A | 0.74 / 0.66 / 0.60 | 0.93 / 0.82 / 0.74 | 119 / 63 / 34 | -1.39 / -0.76 / -0.42 |
| B | 0.93 / 0.89 / 0.87 | 0.99 / 1.00 / 1.01 | 11.9 / 8.4 / 6.0 | 3.98 / 2.82 / 2.01 |
| C | 0.95 / 0.91 / 0.88 | 1.19 / 1.17 / 1.17 | 7.2 / 5.4 / 4.0 | 0.86 / 0.73 / 0.62 |
| D | 0.84 / 0.83 / 0.81 | 1.01 / 0.99 / 0.98 | 11.0 / 7.5 / 5.2 | 2.67 / 1.94 / 1.44 |
| E | 0.64 / 0.56 / 0.50 | 0.88 / 0.82 / 0.76 | 1.93 / 1.37 / 0.99 | 0.44 / 0.35 / 0.27 |
| F | 0.90 / 0.82 / 0.77 | 1.12 / 1.07 / 1.04 | 13.1 / 8.3 / 5.3 | 2.53 / 1.85 / 1.36 |
| G | 0.91 / 0.84 / 0.78 | 1.04 / 0.99 / 0.96 | 16.3 / 10.3 / 6.7 | -0.66 / -0.28 / -0.08 |

How often the 5-95% band of the 64 draws holds the true burden over days
84-119, at 1 / 0.5x (a draw cut off at the safety threshold counts as a
miss), and the 0.5x band's width on day 119 as a share of the truth:

| Variant | Twin | vp1 | vp2 | vp3 |
|---|---|---|---|---|
| A | 100 / 99%, 0.94 | 100 / 54%, 0.26 | 100 / 100%, 105 | 100 / 63%, 1.90 |
| B | 100 / 100%, 0.70 | 100 / 100%, 0.83 | 100 / 100%, 19.5 | 100 / 100%, 5.15 |
| C | 100 / 100%, 0.69 | 100 / 100%, 0.77 | 100 / 100%, 21.7 | 100 / 100%, 3.65 |
| D | 100 / 100%, 0.41 | 100 / 100%, 0.42 | 0 / 0%, 9.07 | 61 / 100%, 1.86 |
| E | 100 / 77%, 0.64 | 100 / 54%, 0.30 | 58 / 58%, 22.8 | 56 / 63%, 2.27 |
| F | 100 / 92%, 0.50 | 37 / 36%, 0.87 | 100 / 100%, 56.6 | 0 / 0%, 6.87 |
| G | 100 / 100%, 0.55 | 80 / 71%, 0.74 | 100 / 100%, 44.5 | 19 / 19%, 7.82 |

On vp2 and vp3 the bands that always hold the truth are 3.7-105 times the
true burden wide, so holding it says little.

Dose effect at 0.5x (the share of the true day-119 difference between 0.5x
and 1x that the forecast gets) and training error (days 5-84), for the
twin / vp1 / vp2 / vp3:

| Variant | Dose effect at 0.5x | Training error (%) |
|---|---|---|
| A | 0.43 / 0.34 / -0.32 / 0.00 | 4.7 / 5.6 / 64.4 / 14.9 |
| B | 0.78 / 1.06 / 3.57 / 1.16 | 3.2 / 5.3 / 23.4 / 14.6 |
| C | 0.80 / 1.13 / 2.77 / 0.51 | 3.3 / 4.8 / 19.9 / 8.6 |
| D | 0.78 / 0.92 / 2.89 / 0.90 | 4.5 / 5.6 / 15.0 / 8.1 |
| E | 0.33 / 0.51 / 0.61 / 0.20 | 5.4 / 4.5 / 12.8 / 14.5 |
| F | 0.59 / 0.86 / 2.22 / 0.85 | 3.9 / 4.6 / 14.0 / 14.3 |
| G | 0.62 / 0.78 / 2.82 / 0.17 | 3.7 / 4.5 / 17.9 / 7.0 |

**The drug factors** (C, D and G). The truth is s / 0.23; for the new
patients it is 1.00 / 0.87 / 1.88 / 1.83 (twin, vp1-3). The effective kill
is -N_11 of the new patient's operator, in the twin's units:

| Variant | Past patients' factors against the truth: rank correlation (twin, vp1-3 fits) | Spread ω | M's N_11 | New patient's factor: twin / vp1 / vp2 / vp3 | New patient's effective kill |
|---|---|---|---|---|---|
| C | +0.90 / +0.98 / +0.93 / +0.90 | 0.11-0.24 | -0.86 to -0.93 | 1.00 / 0.72 / 1.05 / 1.26 | 0.85 / 0.64 / 0.95 / 1.17 |
| D | +0.83 / +0.82 / +0.83 / +0.83 | 0.25-0.26 | -0.80 | 0.94 / 0.72 / 1.36 / 1.34 | 0.75 / 0.58 / 1.09 / 1.07 |
| G | +0.81 / +0.79 / +0.76 / +0.76 | 0.13-0.29 | -0.77 to -0.97 | 0.93 / 0.69 / 1.04 / 1.28 | 0.91 / 0.64 / 0.81 / 1.09 |

- **The factors put the past patients in the right order but squeeze
  them.** In C's twin fit the first eight past patients' factors span
  0.92-1.08, against true 1.05-1.96.
- **The new patients' factors move only part of the way.** For vp2 and
  vp3 they reach 1.04-1.36, against 1.83-1.88. For vp1 (0.87) they
  overshoot to 0.69-0.72. With 12 weekly scans the new patient's own data
  do not pull the factor far from the cohort.

**Expectations against outcomes.**

| Expected | Outcome |
|---|---|
| Drug factor (C against B): better on vp2 and vp3; about the same on vp1 | Better on vp2 (296 against 420) and vp3 (12.4 against 137.5). Worse on vp1 (23.4 against 13.6) |
| The past patients' factors rank their sensitivities (rank correlation 0.7 or more) | Yes: C +0.90 to +0.98, G +0.76 to +0.81, D +0.82 to +0.83 |
| The new patients' factors move from 1 toward the truth for at least three of the four | Partly: toward the truth for vp2 and vp3, past it for vp1, near 1 for the twin (whose truth is 1) in C, D and G |
| K = 64 with factors (D) no worse than C | No: mean 104.5 against 85.2. Better on vp1 only |
| Projected (E, F, G) better at 0.5x than the free variants at the same K | E against A: yes (mean 38.5 against 573). F against B: yes (84.3 against 131.1). G against C: no (117.3 against 81.4). On the twin all three are worse at 0.5x (34.5 against 23.8, 13.2 against 10.8, 12.5 against 11.0) |
| Winner C or G | No: E |
| The twin stays the hardest patient | No: vp2 is the hardest for every variant except E (vp3) |

**After the fact, no new fits: other ways to read the same errors.** The
mean over patients is dominated by vp2. Three other summaries:
- **Error relative to the tumour at the cut.** The same squared field
  error, divided by the true field on day 84 instead of over days 84-119,
  so that a shrinking tumour does not inflate it. Mean of the three arms:

  | Patient | A | B | C | D | E | F | G |
  |---|---|---|---|---|---|---|---|
  | Twin | 18.5 | 8.5 | 8.8 | 11.5 | 29.4 | 10.3 | 9.8 |
  | vp1 | 13.7 | 14.8 | 25.1 | 14.7 | 12.3 | 18.5 | 13.1 |
  | vp2 | 1285 | 211 | 149 | 162 | 25.9 | 137 | 196 |
  | vp3 | 72.1 | 85.6 | 8.4 | 44.5 | 34.3 | 50.8 | 55.9 |
  | Mean | 347 | 79.9 | 47.7 | 58.2 | **25.5** | 54.2 | 68.8 |

- **Mean rank over the four patients** (1 is best): E 2.75, C 3.25,
  G 3.75, D and F 4.00, B 4.75, A 5.50.
- **Day-119 burden, mean |log(forecast / truth)|:** E 0.54, C 0.56, D
  0.72, F 0.75, B 0.82; A and G have negative burdens.

E stays first on all three; C is second. The order is a reading, not a
second rule.

**Why E wins (a reading).** E fits each new patient to its own scans, so a
strong responder is not pulled toward the cohort. But from 12 weekly scans
it learns little of the dose response (dose effect at 0.5x 0.20-0.61,
against 0.78-0.80 for B and C on the twin), so on the twin it forecasts
too little regrowth (day-119 burden 0.50 of the truth at 0.5x). Pooling
helps the twin and vp1, whose kills (1.00 and 0.87) are close to the
cohort's shared drug term (0.84-0.93). It fails the strong responders,
whose kills (1.83-1.88) are near the top of the past patients' range
(1.05-1.96) but twice the shared term. The drug factor rescues vp3 but not
vp2, whose tumour shrinks fastest.

#### Stage 2: new patients and sparser scans

**The confirmation patients, weekly scans to day 84.** Error (%), days
84-119, at 1 / 0.75 / 0.5x, with the mean of the three in brackets:

| Patient | E projected alone | B pooled | A alone |
|---|---|---|---|
| vp4 | 33.5 / 37.6 / 42.9 (38.0) | 23.2 / 18.3 / 14.0 (18.5) | 25.2 / 19.2 / 16.0 (20.1) |
| vp5 | 44.2 / 49.2 / 55.1 (49.5) | 10.6 / 8.0 / 12.8 (10.4) | 65.3 / 57.1 / 51.2 (57.9) |
| vp6 | 39.7 / 43.5 / 47.7 (43.6) | 29.1 / 32.4 / 36.1 (32.6) | 12.1 / 18.4 / 26.2 (18.9) |
| vp7 | 35.7 / 41.4 / 49.2 (42.1) | 113.2 / 100.5 / 81.4 (98.4) | 56.2 / 68.3 / 79.8 (68.1) |
| vp8 | 25.1 / 27.8 / 31.3 (28.1) | 7.9 / 8.6 / 9.9 (8.8) | 12.1 / 15.7 / 20.3 (16.1) |
| vp9 | 30.7 / 33.8 / 37.4 (34.0) | 13.6 / 14.8 / 16.7 (15.0) | 15.6 / 18.9 / 23.6 (19.4) |
| **Mean** | 39.2 | **30.6** | 33.4 |
| Median | 40.1 | **16.8** | 19.7 |

The stress arms (0.25 / 0x) range over 35.0-61.0 / 38.5-66.3 for E,
11.7-58.4 / 13.9-42.3 for B and 19.1-88.8 / 26.6-94.6 for A (per patient
in the log).
- **E, the stage-1 choice, has the highest mean on the new patients.** It
  is low on every one: the day-119 burden is 0.34-0.68 of the truth, and
  it already starts at 0.68-0.88 of the truth at the cut. Its dose effect
  at 0.5x is 0.25-0.39.
- **B is the best typical forecast** (median 16.8; mean rank 1.5 of 3),
  but it fails vp7 (98.4%), a strong responder (kill 1.63). It starts
  vp7's forecast at 1.69 times the truth and ends at 1.77-3.03 times. On
  vp6, the fastest grower after vp1 (k 0.064, against the twin's 0.052),
  it is low (0.57-0.69).
- **A forecasts a negative burden** on vp5 at 1x (-0.19 of the truth) and
  on vp7 at every arm (-0.39 / -0.23 / -0.14).

**Sparser scans, all ten patients.** Mean of the three scenario arms (%;
per-arm numbers and stress arms in the log):

| Patient | Fortnightly to day 84: E / B / A | 4-weekly to day 84: E / B / A | Weekly to day 57: E / B / A |
|---|---|---|---|
| Twin | 27.6 / 12.5 / 97.7 | 85.1 / 55.8 / 55.2 | 39.3 / 13.4 / 30.8 |
| vp1 | 14.9 / 43.7 / 16.7 | 32.0 / 90.5 / 16.6 | 46.6 / 50.8 / 23.6 |
| vp2 | 64.7 / 453 / 2408 | 1093 / 352 / 2637 | 29.7 / 142.3 / 22.4 |
| vp3 | 24.8 / 99.3 / 852 | 705 / 42.2 / 838 | 48.2 / 51.9 / 40.4 |
| vp4 | 33.7 / 22.3 / 609 | 60.0 / 23.1 / 616 | 35.5 / 11.4 / 28.8 |
| vp5 | 44.8 / 13.1 / 389 | 240 / 46.3 / 421 | 45.4 / 15.0 / 30.2 |
| vp6 | 43.0 / 44.6 / 60.0 | 117.9 / 80.8 / 101.7 | 55.3 / 45.3 / 27.3 |
| vp7 | 23.6 / 67.1 / 742 | 581 / 27.8 / 756 | 43.2 / 40.4 / 32.7 |
| vp8 | 30.7 / 20.6 / 95.9 | 67.3 / 63.2 / 55.5 | 37.3 / 18.6 / 28.6 |
| vp9 | 24.9 / 25.2 / 252 | 99.9 / 38.0 / 253 | 33.5 / 14.0 / 29.0 |
| **Mean** | **33.3** / 80.2 / 552 | 308 / **82.0** / 575 | 41.4 / 40.3 / **29.4** |
| Median | **29.2** / 34.5 / 320 | 109 / **51.0** / 337 | 41.3 / 29.5 / **28.9** |

- **No variant does well on sparse scans, and none is best on all three.**
  E has the lowest mean fortnightly, B 4-weekly, A weekly to day 57. Only
  10 of the 90 patient means are under the 18.9% bar (B 7, A 2, E 1).
- **Fortnightly (7 scans).** A, the free form fitted alone, cannot fit the
  scans: training error 27-61%, and its forecast starts at 0.88-13.3 times
  the truth. E fits better and is the most even (worst 64.7). B is good
  on some patients (twin 12.5, vp5 13.1) and fails the strong responders
  (vp2 453, vp3 99.3, vp7 67.1).
- **4-weekly (4 scans).** Everything fails. E's training error is 16-51%,
  and its day-119 burden runs up to 33 times the truth. B has the lowest
  mean but is far too low on vp1 and vp6 (day-119 burden 0.04-0.06 and
  0.07-0.09 of the truth; training error 46% and 27%).
- **Weekly to day 57 (8 scans; a 62-day forecast).** A has the lowest mean
  (29.4) and worst case (40.4), but forecasts a negative burden on three
  patients at 1x. B's median is about the same (29.5), with a worse tail
  (vp2 142.3). E is low: day-119 burden 0.19-0.51 of the truth on eight of
  the ten patients.

**Reading.** On the declared protocol the round has no clear winner. The
stage-1 choice (E) is consistently biased low; pooling (B) gives the best
typical forecast when the patient is like the cohort and fails when it is
not; the free form alone (A) fails as soon as the scans thin out. The
next section shows how much of these errors comes from the protocol's
starting point rather than the models.

## Restarting the forecast from the last scan (design written before the new fits)

**Why (found after the fact).** The benchmark scores every forecast by
integrating the fitted model from the first scan (day 5) all the way to
day 119. So any miss on the patient's own training scans is carried into
the forecast window. This is much of why the strong responders' errors
were so large in the variant round: the pooled variants start vp2's
forecast window at 2.6-4.0 times its true tumour. In a clinic, a forecast
would start from the latest scan. So the same fitted models are also
integrated from the patient's last training scan: its coordinates on the
POD basis, noisy as measured. Nothing is refitted. Only the forecast's
starting point changes.

**What has been seen.** After stage 1 and the confirmation patients'
weekly results, I restarted the point forecast of every weekly-to-day-84
fit (session script `restart_check.py`). Started from the first scan
instead, the script reproduces the official errors to 3e-12 points.
Mean over patients of the 1 / 0.75 / 0.5x mean error:

| Variant | Patients | From the first scan | Restarted | Worst, restarted |
|---|---|---|---|---|
| A alone | 10 | 292.9 | 34.4 | 113.4 (vp2) |
| B pooled | 10 | 76.4 | 16.2 | 46.4 (vp2) |
| E projected alone | 10 | 38.2 | 21.3 | 28.4 (vp3) |
| C pooled + factor | 4 development | 85.2 | 22.4 | 46.4 (vp2) |
| D K64 pooled + factor | 4 development | 104.5 | 60.5 | 176.9 (vp2) |
| F projected pooled | 4 development | 96.1 | 18.7 | 28.1 (vp2) |
| G projected pooled + factor | 4 development | 125.2 | 13.8 | 19.4 (vp2) |

Restarted:
- **G is best on the development patients:** 8.0 / 13.6 / 19.4 / 14.1 on
  the twin and vp1-3. All its day-119 burdens are positive (0.50-1.18 of
  the truth).
- **Some still predict a negative tumour at day 119:** B, C and F on vp2
  (B -1.75 / -0.72 / -0.21 of the truth, C -2.09 / -0.94 / -0.35, F -0.22
  at 1x), and A on vp3 and vp7. Nothing in the reduced models keeps the
  burden positive.
- **E stays biased low:** below the truth on 29 of its 30 patient-arms
  (0.42-0.90; vp2 at 1x is 1.18). Errors 10.2-28.4.
- **On the confirmation patients vp4-9:** B's restarted mean is 11.6
  (5.5-14.8), E's 22.9 and A's 21.3. G, C and F were never fitted there.

Not seen yet: any restarted number on the sparse schedules, and any
restarted band apart from two timing checks (0.5x; the band holds the truth
55% of the time for G on vp2 and 13% for E on the twin). Standard-protocol
stage-2 results were seen as they finished: fortnightly E on the twin and
vp1-5, B and A on the twin and vp1-6, and 4-weekly A on the twin.

**The restart protocol.**
- **Start:** the reduced coordinates of the last training scan. On every
  schedule this scan is on the cut day (84, or 57), because the schedules
  fix their first and last scan days. The script checks this for every
  fit.
- **Point forecast:** the posterior-mean operator from that start.
- **Draws:** the benchmark's same 64 operator draws, each from the same
  start. So the band (5-95% of the draws) holds the operator's uncertainty
  only. It has zero width at the cut and leaves out the scan's own noise.
- **Everything else is the benchmark's.** The drug input still includes
  every earlier dose. The safety threshold is the same: a draw that
  crosses it is cut off and counts as a miss. The window (cut to day 119)
  is the same, and so is the field error, which includes what the basis
  cannot represent. Band coverage is counted strictly, as before.
- **Both protocols are reported.** The standard one stays the benchmark's
  definition.

**New fits** (48; seed 48; each run once; no code change).
- **G, C and F on the six confirmation patients,** weekly scans to day 84
  (18 fits). With B, E and A already there, every variant except D then
  has all ten patients on this schedule.
- **G on all ten patients for fortnightly to day 84, 4-weekly to day 84
  and weekly to day 57** (30 fits). B, E and A are already on these
  schedules in stage 2.

G is carried to the sparse schedules because it was the best restarted
variant on the development patients, an after-the-fact choice. C, F and D
are not fitted on the sparse schedules. D is the slowest (65 minutes a fit)
and the worst restarted.

**Evaluation.** The restarted point forecast and 64 draws at all five arms,
for every fit in the variant round and in this follow-up, next to the
standard protocol.

**Reported** (no pass or fail; the 18.9% bar is drawn for reference).
- Restarted error at 1, 0.75 and 0.5x, with the stress arms, per patient,
  schedule and variant.
- The mean, median and worst over patients, with development and
  confirmation patients also shown separately.
- The day-119 burden against the truth, including negative burdens.
- Restarted band coverage and width, and draws that were cut off.

**Expectations** (written before the new fits, and before any restarted
number on the sparse schedules).
1. **Confirmation patients, weekly to day 84:** G's restarted mean is at
   most B's 11.6, and lower than C's and F's. G gives no negative burden.
   This is uncertain, since B is already good on these patients.
2. **Restarting helps on the sparse schedules:** it lowers the mean error
   of each of A, B, E and G on each one.
3. **G is best on the sparse schedules:** it has the lowest restarted mean
   of A, B, E and G on each one. I am least sure about 4-weekly scans.
4. **Fewer scans, larger errors:** restarted errors rise from weekly to
   fortnightly to 4-weekly (to day 84). Weekly to day 57 is worse than
   weekly to day 84: the forecast covers 62 days instead of 35, from fewer
   treated scans.
5. **The restarted bands hold the truth less than 90% of the time** on
   most fits, because they start with zero width.

**What it is for.** Suppose G, restarted, is best on the confirmation
patients and on most sparse schedules. Then "G, forecast from the latest
scan" is the setting I bring to the user for the Neural-ODE comparison,
where both models would be scored both ways. Otherwise the results are
reported schedule by schedule, and the user chooses.

**Disclosures.**
- **After the fact.** Both the restart and the choice of G for the sparse
  schedules came after seeing the results above. G was the best of seven
  on four patients, which flatters it. The confirmation patients and the
  sparse schedules are the test.
- **Session script.** The restart is computed by a session script that
  reuses the benchmark's functions, not by repository code. If the
  protocol is adopted, it moves into `benchmark_evaluation.py` with a
  test.
- **The start is a noisy scan,** as in a clinic. The band does not include
  that noise.
- **Run timing.** The new fits start on the same virtual machine while
  stage 2 is still running. They take cores as stage-2 jobs finish.

```bash
cd experiments/tumor
# new fits: the variant round's case names, e.g.
conda run -n prob_rom python 04_unified_benchmark.py midcourse-reduced-trend-chemo-fortnightly-scans-learned-trend-cohort8-projected-pooled-drug-vp4 --seeds 48
# restart (session script restart_check.py): verify | run -j 16 | report
```

### Results (restarted; seed 48, each fit run once)

**Run log.**
- **The 48 new fits** started at 18:56 UTC on the same machine, while
  stage 2 was still running, each pinned to a free core as before. A
  pooled fit holds 3.0-3.4 GB and peaks at 3.9 GB, and the machine has no
  swap, so from 19:06 at most 14 fits ran at once. All 48 finished (exit 0) by
  21:01 UTC, and every arm evaluated 64 draws.
- **Check of the script** (on the Mac, before any restarted number from
  the new fits; `logs/restart_verify.txt`). Started from the first scan,
  it reproduces the official errors of the 28 stage-1 fits at 1 / 0.75 /
  0.5x to 3e-12 points. With draws it reproduces the official band
  counts, complete draws and cut-off draws exactly on 15 fit-arms: the
  twin's G, vp3's F and vp2's E at all five arms. The last two each have
  a cut-off draw.
- **The two timing checks** named in the design (G on vp2 and E on the
  twin, 0.5x) come out the same in the full run.
- **The restarted evaluation** ran on the machine at the lowest priority
  (`nice 19`) while fits were running, which slowed them a little. It
  covered the 585 fit-arms present at 19:03 with 16 processes (19:03 to
  20:07, about 100 s per fit-arm). Then passes on two processes picked up
  each newly finished fit, and a last pass ran after the final fit.
  The third pass was stopped when the last fit finished, so that the
  final pass could use all 16 cores; the runner resumes from its saved
  results. A last pass found nothing left: 920 fit-arms (184 fits, five
  arms each), 21:13 UTC.
- **The machine was deleted** with its resource group (requested at
  21:16 UTC, about 5.3 hours after it was created; checked gone
  afterwards). Estimated cost about $6 (F16as_v6 at $1.093 an hour, plus
  disk and address).
- Session scripts `restart_check.py` (verify, run, report),
  `restart_expectations.py` (the expectations and tables below) and
  `restart_figures.py`. Logs `logs/restart_verify.txt`,
  `logs/restart_report.txt` and `logs/restart_expectations.txt`. Figures
  `figures/restart_error.png` and
  `figures/restart_burden_<schedule>_0.5x.png`, with the new patient's
  scans as black stars. All are in the variant round's session folder.

**Restarted error by schedule.** A patient's error is the mean of the
1 / 0.75 / 0.5x arms; the table gives the mean over patients, then the
median and the worst patient. "From the first scan" is the same fits
scored the benchmark's way. "Band" is the mean share of forecast times
at which the 5-95% band of the 64 draws holds the true burden. Negative
burdens count patient-arms (1 / 0.75 / 0.5x) whose day-119 forecast is
below zero.

**Weekly to day 84.**

| Variant | Patients | From the first scan | Restarted: mean | Median | Worst | vp4-9 mean | Under 18.9% | Band at 1 / 0.5x | Negative burdens |
|---|---|---|---|---|---|---|---|---|---|
| A alone | 10 | 292.9 | 34.4 | 15.5 | 113.4 (vp2) | 21.3 | 6 of 10 | 80 / 36% | 6 (vp3, vp7) |
| B pooled | 10 | 76.4 | 16.2 | 14.3 | 46.4 (vp2) | 11.6 | 8 of 10 | 82 / 61% | 3 (vp2) |
| C pooled + factor | 10 | 43.0 | 16.0 | 13.3 | 46.4 (vp2) | 11.7 | 8 of 10 | 87 / 69% | 3 (vp2) |
| D K64 pooled + factor | 4 | 104.5 | 60.5 | 26.9 | 176.9 (vp2) | — | 2 of 4 | 26 / 35% | 0 |
| E projected alone | 10 | 38.2 | 21.3 | 20.6 | 28.4 (vp3) | 22.9 | 3 of 10 | 92 / 48% | 0 |
| F projected pooled | 10 | 58.4 | 14.1 | 13.3 | 28.1 (vp2) | 11.0 | 8 of 10 | 61 / 51% | 1 (vp2) |
| G projected pooled + factor | 10 | 66.0 | 12.6 | 12.6 | 19.4 (vp2) | 11.8 | 9 of 10 | 89 / 58% | 0 |

**Fortnightly to day 84.**

| Variant | Patients | From the first scan | Restarted: mean | Median | Worst | vp4-9 mean | Under 18.9% | Band at 1 / 0.5x | Negative burdens |
|---|---|---|---|---|---|---|---|---|---|
| A alone | 10 | 552.1 | 46.9 | 41.4 | 112.5 (vp2) | 42.0 | 1 of 10 | 40 / 63% | 0 |
| B pooled | 10 | 80.2 | 16.2 | 13.1 | 47.2 (vp2) | 11.9 | 8 of 10 | 82 / 66% | 3 (vp2) |
| E projected alone | 10 | 33.3 | 20.0 | 20.6 | 27.7 (vp6) | 21.1 | 3 of 10 | 87 / 59% | 0 |
| G projected pooled + factor | 10 | 68.6 | 12.1 | 12.6 | 20.7 (vp2) | 10.8 | 9 of 10 | 90 / 71% | 0 |

**4-weekly to day 84.**

| Variant | Patients | From the first scan | Restarted: mean | Median | Worst | vp4-9 mean | Under 18.9% | Band at 1 / 0.5x | Negative burdens |
|---|---|---|---|---|---|---|---|---|---|
| A alone | 10 | 575.0 | 48.3 | 44.7 | 118.7 (vp2) | 43.8 | 3 of 10 | 18 / 51% | 0 |
| B pooled | 10 | 82.0 | 18.5 | 13.9 | 55.8 (vp2) | 12.5 | 6 of 10 | 71 / 69% | 2 (vp2) |
| E projected alone | 10 | 308.0 | 29.1 | 18.2 | 72.1 (vp2) | 22.6 | 5 of 10 | 38 / 59% | 0 |
| G projected pooled + factor | 10 | 216.3 | 19.7 | 19.7 | 34.7 (vp1) | 15.8 | 4 of 10 | 76 / 71% | 0 |

**Weekly to day 57.**

| Variant | Patients | From the first scan | Restarted: mean | Median | Worst | vp4-9 mean | Under 18.9% | Band at 1 / 0.5x | Negative burdens |
|---|---|---|---|---|---|---|---|---|---|
| A alone | 10 | 29.4 | 26.7 | 26.1 | 35.1 (vp3) | 26.3 | 0 of 10 | 97 / 56% | 2 (vp3, vp7) |
| B pooled | 10 | 40.3 | 17.7 | 15.4 | 30.0 (vp1) | 14.2 | 6 of 10 | 56 / 66% | 1 (vp2) |
| E projected alone | 10 | 41.4 | 35.9 | 35.3 | 48.7 (vp6) | 35.7 | 0 of 10 | 61 / 19% | 0 |
| G projected pooled + factor | 10 | 32.5 | 18.9 | 15.2 | 36.4 (vp6) | 18.9 | 6 of 10 | 80 / 58% | 0 |

**Restarted error per patient** (mean of 1 / 0.75 / 0.5x):

**Weekly to day 84.**

| Patient | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| twin | 14.8 | 9.0 | 9.3 | 11.4 | 20.3 | 8.4 | 8.0 |
| vp1 | 10.9 | 16.6 | 20.5 | 13.5 | 10.2 | 17.2 | 13.6 |
| vp2 | 113.4 | 46.4 | 46.4 | 176.9 | 17.1 | 28.1 | 19.4 |
| vp3 | 77.2 | 20.3 | 13.5 | 40.3 | 28.4 | 21.1 | 14.1 |
| vp4 | 12.1 | 13.8 | 15.8 | — | 20.9 | 7.5 | 10.0 |
| vp5 | 20.8 | 12.3 | 14.9 | — | 27.8 | 10.8 | 15.6 |
| vp6 | 13.2 | 14.8 | 13.1 | — | 26.4 | 17.7 | 18.8 |
| vp7 | 53.3 | 14.8 | 11.9 | — | 24.3 | 15.8 | 11.6 |
| vp8 | 12.1 | 5.5 | 6.0 | — | 18.1 | 7.0 | 5.9 |
| vp9 | 16.2 | 8.5 | 8.5 | — | 19.8 | 7.1 | 8.9 |

**Fortnightly to day 84.**

| Patient | A | B | E | G |
|---|---|---|---|---|
| twin | 21.1 | 10.4 | 21.7 | 8.2 |
| vp1 | 13.1 | 12.9 | 12.5 | 12.4 |
| vp2 | 112.5 | 47.2 | 16.7 | 20.7 |
| vp3 | 70.1 | 19.8 | 22.8 | 14.4 |
| vp4 | 62.7 | 13.3 | 15.3 | 8.0 |
| vp5 | 48.5 | 12.7 | 23.1 | 12.8 |
| vp6 | 19.8 | 16.1 | 27.7 | 16.1 |
| vp7 | 66.8 | 14.3 | 19.6 | 13.3 |
| vp8 | 19.8 | 5.9 | 21.7 | 6.0 |
| vp9 | 34.3 | 8.8 | 19.3 | 8.6 |

**4-weekly to day 84.**

| Patient | A | B | E | G |
|---|---|---|---|---|
| twin | 18.5 | 8.9 | 17.2 | 20.8 |
| vp1 | 13.4 | 22.4 | 12.9 | 34.7 |
| vp2 | 118.7 | 55.8 | 72.1 | 22.8 |
| vp3 | 69.6 | 22.9 | 53.1 | 24.0 |
| vp4 | 64.2 | 9.8 | 12.7 | 15.4 |
| vp5 | 51.2 | 9.8 | 26.7 | 20.2 |
| vp6 | 24.4 | 19.7 | 19.2 | 15.3 |
| vp7 | 67.6 | 18.0 | 47.4 | 19.2 |
| vp8 | 17.2 | 8.6 | 13.8 | 13.1 |
| vp9 | 38.1 | 8.9 | 15.5 | 11.5 |

**Weekly to day 57.**

| Patient | A | B | E | G |
|---|---|---|---|---|
| twin | 28.5 | 11.9 | 35.4 | 12.1 |
| vp1 | 22.9 | 30.0 | 42.1 | 23.5 |
| vp2 | 22.8 | 27.3 | 27.8 | 12.5 |
| vp3 | 35.1 | 22.8 | 39.7 | 27.1 |
| vp4 | 26.1 | 8.4 | 30.2 | 13.1 |
| vp5 | 25.0 | 12.8 | 37.5 | 26.9 |
| vp6 | 25.1 | 26.3 | 48.7 | 36.4 |
| vp7 | 29.0 | 18.0 | 35.2 | 17.3 |
| vp8 | 26.3 | 9.4 | 33.4 | 9.1 |
| vp9 | 26.1 | 10.1 | 28.9 | 10.9 |

**Expectations against outcomes.**

| Expected | Outcome |
|---|---|
| 1. Confirmation patients, weekly to day 84: G at most B's 11.6, below C and F; no negative burden | No: F 11.0, B 11.6, C 11.7, G 11.8, a tie within 0.8 points. G gives no negative burden. Over all ten patients G is lowest (12.6, against F 14.1, C 16.0 and B 16.2) and has the smallest worst case (19.4) |
| 2. Restarting lowers the mean of A, B, E and G on each sparse schedule | Yes, 12 of 12. The biggest drops are where the first-scan forecasts were worst: 4-weekly scans (G 216.3 to 19.7, E 308.0 to 29.1, A 575.0 to 48.3) and A on fortnightly scans (552.1 to 46.9). The smallest are A and E on weekly scans to day 57 (29.4 to 26.7, 41.4 to 35.9) |
| 3. G lowest of A, B, E and G on each sparse schedule (least sure on 4-weekly) | Partly: fortnightly yes (G 12.1, B 16.2); 4-weekly no (B 18.5, G 19.7); weekly to day 57 no (B 17.7, G 18.9). By median, G is lowest on weekly to day 57 (15.2 against 15.4), not on 4-weekly (19.7 against 13.9) |
| 4. Errors rise from weekly to fortnightly to 4-weekly (to day 84); weekly to day 57 worse than weekly to day 84 | No, for all four. Weekly and fortnightly scans give about the same error for B, E and G (16.2 and 16.2, 21.3 and 20.0, 12.6 and 12.1); only A rises (34.4, 46.9). 4-weekly is the worst of the three for all four (A 48.3, B 18.5, E 29.1, G 19.7). Weekly to day 57 is worse than weekly to day 84 for B, E and G, not for A, whose weekly-to-84 mean is dominated by the strong responders vp2, vp3 and vp7 (113.4, 77.2, 53.3) |
| 5. Restarted bands hold the truth less than 90% of the time on most fits | Yes: 147 of 184 fits (mean over 1 / 0.75 / 0.5x) and 363 of 552 fit-arms |

**Reading.**
- **Most of the benchmark's error was the starting point.** Every variant
  improves on every schedule once its forecast starts from the last
  scan. Nothing was refitted. G's means under the benchmark's protocol
  (66.0, 68.6, 216.3 and 32.5 on the four schedules) fall to 12.6, 12.1,
  19.7 and 18.9.
- **G and B are the two best restarted variants.**
  - G (projected form, pooled, a drug factor per patient) has the lowest
    mean on weekly (12.6) and fortnightly (12.1) scans to day 84 and the
    smallest worst case on three of the four schedules (19.4, 20.7,
    34.7). It never forecasts a negative tumour. It is under the 18.9%
    bar for 9, 9, 4 and 6 of the ten patients.
  - B (free form, pooled) has the lowest mean on 4-weekly scans (18.5)
    and weekly scans to day 57 (17.7), 1.2 points below G. It is under
    the bar for 8, 8, 6 and 6 of the ten. It forecasts a negative
    tumour for vp2, the strongest responder, on every schedule.
  - On the confirmation patients F, B, C and G are within 0.8 points
    (11.0-11.8).
- **Fewer scans cost less than expected.** Restarted, fortnightly scans
  (7 MRIs to day 84) do as well as weekly ones (12): G 12.1 against
  12.6. Four-weekly scans (4 MRIs) cost G about 7 points and B about 2.
- **What is left is too little regrowth after a dose cut.** G's day-119
  burden at 0.5x is below the truth for all ten patients with weekly or
  fortnightly scans (median 0.76 and 0.83 of the truth), while at 1x it
  is close (median 0.97 and 1.06). On the three day-84 schedules every
  variant's day-119 burden falls against the truth from 1x to 0.5x (weekly
  scans: B 0.93 to 0.78, E 0.76 to 0.59). The dense-scan fits earlier in
  this record had the same weakness.
- **The restarted bands are too narrow.** They start with zero width and
  hold only the operators' uncertainty. G's hold the truth 89% of the
  time at 1x and 58% at 0.5x (weekly to day 84): at 0.5x the truth
  regrows out of the band.
- **The projected form alone (E) stays low:** median day-119 burden at
  0.5x is 0.59 of the truth with weekly scans to day 84 and 0.42 to
  day 57. Pooled, the same form comes closer (weekly to day 84: F 0.85,
  G 0.76; to day 57: G 0.89). Reading: the past patients supply the dose
  response that a few scans of one patient cannot.
- **The declared condition for the Neural-ODE setting is not met.** G is
  not best on the confirmation patients (a tie) and is best on one of the
  three sparse schedules. So, as declared, the results are reported
  schedule by schedule and the user chooses. The candidates are G and B,
  both forecast from the latest scan.

**Disclosures.**
- **After the fact,** as stated in the design: the restart and the choice
  of G for the sparse schedules came after the stage-1 and confirmation
  results had been seen.
- **Draws cut off at the safety threshold** (each counts as a miss).
  Scored the benchmark's way, the new fits lose draws on sparse scans:
  G on all ten 4-weekly fits (0-25 of 64 per arm; vp1 23-25), at most one
  per arm on six of the ten fortnightly fits, and on five of the ten
  weekly-to-57 fits (one or two at 0x only, except vp1 with 1-11 per
  arm). F on vp7 (weekly to day 84) loses one per arm. Restarted, only
  two fits lose any: E on 4-weekly vp1 (one per arm) and G on
  weekly-to-57 vp1 (2 at 0.25x and 17 at 0x; none at the judged arms).
- **Restart workers ran beside the fits** at the lowest priority. This
  changes only run times.
- **One anatomy, one seed per patient, a best-case cohort,** as in the
  variant round's disclosures.
- **The start is a noisy scan,** and the band leaves out its noise.
- **Session scripts.** The restart, the expectations check and the
  figures are session scripts that reuse the benchmark's functions. If
  the protocol is adopted, it moves into `benchmark_evaluation.py` with a
  test.

## The Neural ODE on the dose-change design (design written before any fit)

**Why.** The manuscript's tumour comparison still uses the older design
(demo grid, prescribed parameters, half-strength drug, arms up to 2x). The
realistic design is the one in "Dose modifications with the free ROM"
above: the calibrated twin, the protocol's dose cuts (continue, 75%, 50%;
stress 25% and stopped) and about 120 scans in training. The free ROM with
the input-aware GP works there. The paper needs the data-driven
alternative, the Neural-ODE ensemble, on exactly the same data. It also
needs brackets that show how hard each forecast is: a naive forecast below
and the true equation above.

**Already seen** (before any Neural-ODE or dense calibrated-PDE fit).
- Ours on all three cases: the table above.
- Two naive forecasts on the same scans, basis and metric (session script
  `dense_persistence_check.py`, after the fact). **Persistence** holds the
  last training scan. **Dose-blind** uses the true reduced trajectory of
  the planned course for every arm. Medians at 1 / 0.75 / 0.5x:
  - from day 84: persistence 28.2 / 14.0 / 17.3, dose-blind
    4.1 / 16.5 / 30.9;
  - from day 57: persistence 42.1 / 19.6 / 30.1, dose-blind
    5.8 / 26.2 / 46.4.
- The calibrated PDE on the variant round's weekly scans (session baseline
  `calibrated_pde.py`), not on these scans.
- A plumbing smoke of the Neural-ODE runner on development seed 45:
  2 members, 50 updates each. Its errors mean nothing.

**Runs** (each run once; nothing tuned).
1. **Neural-ODE ensemble, recipe unchanged.**
   - 20 members. Each is a tanh MLP from [q, alpha(t)] with 3 hidden
     layers of 128 units.
   - Adam at 5e-4 with gradient clipping at norm 1, for 6000 updates in
     float32. Solver: Tsit5 with adaptive steps.
   - The reported loss filter drops members whose final loss is more than
     3 times the median. The headline centre is the median of the kept
     members.
   - It runs on the three cases above and their evaluation seeds:
     mid-course 48-50, 5-57 48-50, standard regimen 51-53. That makes 9
     ensembles and 180 member fits.
   - Data: the same data folders as ours, whose observation hashes are
     checked on loading.
2. **Calibrated PDE on the same scans.** The session baseline unchanged:
   TumorTwin's LM calibration of (d, k, s), with its guess, bounds, at most
   30 iterations and its stopping rule. The only change is the data: all
   120 segmented scans of each acquisition. That makes 9 jobs, declared as
   an amendment in the baseline's `DECLARATION.md`.
3. **Development seed 45:** smoke tests only, never reported as results.

**Scoring.** The benchmark's evaluation is unchanged. Every forecast runs
from the first scan (day 5), as everywhere in the manuscript. The restart
from the last scan is not used for ours or the Neural ODE here.
- **Headline:** the full-field error over the forecast window (84-119 or
  57-119), per seed and as the median over seeds.
- **Also reported:**
  - the training-window error;
  - the error in each half of the forecast window;
  - the dose effect: the predicted day-119 change from the 1x arm over the
    true change, at 0.75, 0.5, 0.25 and 0x;
  - the day-119 burden over the truth;
  - the band coverage, next to ours. For the Neural ODE the band is the
    5-95% range of the 20 members' burdens, an empirical spread, not
    calibrated. Ours uses 64 posterior draws;
  - the members that complete, and the members that pass the float64
    training replay.
- **Brackets,** on the same scans and metric:
  - the naive forecasts, persistence and dose-blind;
  - the basis floor;
  - the calibrated PDE. It is the true equation, so it is an upper
    reference (an "inverse crime"), not a competitor. It is scored both
    from the first scan (like ours) and restarted from the last;
  - the projected logistic ROM where it was run (1x and 0.5x).

**No pass or fail.** Ours and the Neural ODE are reported arm by arm, as
measured.

**Expectations** (written before any fit).
1. The Neural ODE fits its training scans about as well as ours. Its
   training-window error is at most ours plus 1 point (ours: 1.0-1.4% to
   day 84, 0.7-0.9% to day 57).
2. Ours is lower than the Neural ODE at 0.75x and 0.5x, from both cut days.
   A dose cut needs a dose response learned from one treated course.
3. The Neural ODE's dose effect at 0.5x is below 0.3 of the true one. In
   the older design it was about 0.
4. At 1x the two are within 3 points of each other, either way. Both
   extrapolate the planned course. In the older single-dose task the
   Neural ODE was better.
5. On the stress arms (0.25x, 0x), ours is lower.
6. The calibrated PDE is under 1% from the first scan and under 2%
   restarted, on every arm.

**Platform.** The member fits and the dense calibrated-PDE jobs run on
Azure virtual machines: four F16as_v6 for the members (one member per
core), and one E16as_v6 for the PDE plus one Neural-ODE seed. The OS is
Ubuntu 24.04 on x86-64. Packages match the Mac's pins: Python 3.13.5, jax
0.6.2, torch 2.7.1 CPU. Settings are single-threaded, as in the
benchmark's environment. Each member's random key depends only on its
index, so spreading members across machines changes nothing. Float32
training can differ from a Mac run in the last bits. The evaluation
replays every member in float64 and checks its training fit to 1e-3
relative, the benchmark's own check.

**Disclosures.**
- **After the fact:** this design was chosen after ours had been run on it
  and after the naive forecasts had been seen. The Neural ODE and the PDE
  are run once, with their recipes unchanged.
- **One virtual patient:** the seeds vary only the scan times and the
  noise.
- If the paper adopts this design, it replaces the single-dose and
  multi-dose tasks. Untreated growth stays. The older results stay in the
  repository.

```bash
cd experiments/tumor
# one process per member (any machine; deterministic by index), then the summary and evaluation per seed
python 05_neural_ode_benchmark.py midcourse-reduced-trend-chemo --seeds 48 --only-members 0 --skip-evaluation
python 05_neural_ode_benchmark.py midcourse-reduced-trend-chemo --seeds 48
```

### Results (each seed run once)

Everything ran to the end.
- **Neural ODE:** 9 ensembles of 20 members (180 fits). Every member
  completed and passed the float64 training replay. The loss filter kept
  20/20/20 members from day 84, 18/19/20 from day 57 and 19/19/19 on the
  planned course.
- **Calibrated PDE:** 9 jobs, all rc=0. Each took 18-20 LM iterations
  (none at the cap) and 59-73 min.
- **Ours:** the runs in "Dose modifications with the free ROM" above,
  unchanged.

Every forecast runs from the first scan.

**Forecast error** (full field, % over the forecast window). Each entry is
the median over the three seeds; the per-seed values follow in brackets for
ours and the Neural ODE. At 1x the dose-blind forecast is the true
trajectory itself, so it equals the floor.

From day 84 (`midcourse-reduced-trend-chemo`, seeds 48-50, forecast
84-119):

| arm | persistence | dose-blind | Neural ODE | ours | PDE, first scan | PDE, restarted | floor |
|---|---|---|---|---|---|---|---|
| 1x | 28.2 | 4.1 | 23.7 (23.1/23.7/28.7) | 9.1 (9.1/9.2/8.7) | 0.27 | 1.39 | 4.1 |
| 0.75x | 14.0 | 16.5 | 10.2 (9.3/10.2/12.5) | 6.9 (7.0/6.9/6.9) | 0.23 | 1.27 | 4.5 |
| 0.5x | 17.3 | 30.9 | 16.1 (16.6/16.1/13.2) | 6.7 (6.7/6.6/7.2) | 0.20 | 1.16 | 4.9 |
| 0.25x (stress) | 30.4 | 43.7 | 30.5 (30.5/30.5/27.3) | 7.5 (7.3/7.5/8.8) | 0.17 | 1.08 | 5.4 |
| 0x (stress) | 43.0 | 54.6 | 43.1 (43.1/43.7/40.8) | 8.5 (8.2/8.5/10.5) | 0.16 | 1.03 | 6.0 |

From day 57 (`reduced-regimen-trend-chemo`, seeds 48-50, forecast
57-119):

| arm | persistence | dose-blind | Neural ODE | ours | PDE, first scan | PDE, restarted | floor |
|---|---|---|---|---|---|---|---|
| 1x | 42.1 | 5.8 | 67.1 (67.1/10.3/84.3) | 16.6 (16.9/15.4/16.6) | 0.27 | 0.86 | 5.8 |
| 0.75x | 19.6 | 26.2 | 32.0 (32.0/26.1/44.9) | 12.0 (12.3/11.4/12.0) | 0.23 | 0.76 | 7.4 |
| 0.5x | 30.1 | 46.4 | 22.6 (19.0/45.9/22.6) | 13.9 (15.1/12.8/13.9) | 0.18 | 0.72 | 9.4 |
| 0.25x (stress) | 47.8 | 61.4 | 35.5 (35.5/60.3/32.1) | 16.6 (18.1/14.3/16.6) | 0.15 | 0.75 | 11.9 |
| 0x (stress) | 61.6 | 71.9 | 51.8 (51.8/71.3/48.3) | 20.2 (20.2/18.6/20.4) | 0.16 | 0.87 | 15.0 |

Planned course (`standard-regimen-trend-chemo`, seeds 51-53, forecast
57-119):

| arm | persistence | dose-blind | Neural ODE | ours | PDE, first scan | PDE, restarted | floor |
|---|---|---|---|---|---|---|---|
| 1x | 42.1 | 5.7 | 56.1 (56.1/94.6/53.3) | 16.1 (16.1/15.9/16.3) | 0.26 | 0.86 | 5.7 |

**Dose effect.** This is the day-119 gain: the predicted change from the 1x
arm divided by the true change. 1 is exact and 0 is no response. Values are
the median (range) over seeds.

| arm | true change | ours | Neural ODE |
|---|---|---|---|
| day 84, 0.75x | +34% | 0.70 (0.67-0.72) | -0.00 |
| day 84, 0.5x | +79% | 0.72 (0.68-0.74) | -0.01 |
| day 84, 0.25x | +136% | 0.74 (0.69-0.77) | -0.01 |
| day 84, 0x | +208% | 0.78 (0.72-0.81) | -0.02 |
| day 57, 0.75x | +67% | 0.52 (0.47-0.54) | 0.01 |
| day 57, 0.5x | +167% | 0.56 (0.50-0.60) | -0.01 |
| day 57, 0.25x | +307% | 0.63 (0.57-0.70) | 0.00 |
| day 57, 0x | +490% | 0.76 (0.68-0.87) | -0.00 |

**Training window** (median (range) over seeds):

| | ours | Neural ODE |
|---|---|---|
| from day 84 | 1.19 (0.98-1.38) | 1.82 (1.55-2.16) |
| from day 57 | 0.82 (0.75-0.85) | 1.11 (1.06-1.41) |
| planned course | 0.77 (0.71-0.82) | 1.34 (1.11-1.52) |

**Expectations** (declared above; medians over seeds):
1. **Met.** The Neural ODE fits the training scans within 1 point of ours:
   1.82 vs 1.19 (day 84), 1.11 vs 0.82 (day 57), 1.34 vs 0.77 (planned
   course).
2. **Met.** Ours is lower at 0.75x and 0.5x from both days: 6.9 vs 10.2
   and 6.7 vs 16.1 from day 84; 12.0 vs 32.0 and 13.9 vs 22.6 from day 57.
3. **Met.** The Neural ODE's dose effect at 0.5x is -0.01 from both days.
4. **Not met.** At 1x ours is lower by far more than 3 points: 9.1 vs 23.7
   (day 84), 16.6 vs 67.1 (day 57), 16.1 vs 56.1 (planned course). The
   older single-dose result, where the Neural ODE was better, did not
   carry over.
5. **Met.** Ours is lower on the stress arms: 7.5 vs 30.5 and 8.5 vs 43.1
   from day 84; 16.6 vs 35.5 and 20.2 vs 51.8 from day 57.
6. **Met.** The PDE's worst error on any arm and seed is 0.30% from the
   first scan and 1.39% restarted.

**One exception.** Ours is lower than the Neural ODE on 32 of the 33
arm-acquisition pairs. The exception is the kept arm from day 57 on
acquisition 49: Neural ODE 10.3%, ours 15.4%. On acquisitions 48 and 50
the Neural ODE's errors on that arm are 67.1% and 84.3%; its day-119 burden
there is 1.78 and 2.00 times the truth.

**What else the numbers show.**
- Ours is lower than persistence on every arm and acquisition. The
  Neural ODE's median is worse than persistence's on the planned course
  and in the kept and 0.75x arms from day 57. It is level with it (within
  0.1 points) on the stress arms from day 84.
- Ignoring the dose change costs a lot. The dose-blind forecast's error in
  the changed arms is 16.5-54.6% from day 84 and 26.2-71.9% from day 57.
  Ours is below it in every changed arm.
- Ours still under-predicts the regrowth after a cut: its day-119 burden
  is 0.87-0.95 of the truth at 0.75x and 0.5x from day 84, and 0.70-0.84
  from day 57. Its error in the kept arm grows with the horizon. On the
  planned course it is 6.1-6.6% over days 57-88 and 24.8-25.5% over days
  88-119.
- Our bands (5-95% of 64 draws) hold the true burden at every forecast
  time on the planned course, in both kept arms and at 0.75x from day 84.
  Elsewhere they hold it at 42-64% of the times at 0.75x from day 57,
  61-66% at 0.5x from day 84, 29-36% at 0.5x from day 57 and 20-54% in the
  stress arms.
- The effect band holds the true effect at 5-18 of the 98 forecast times
  from day 84 and 5-13 of the 196 from day 57, except when the doses are
  stopped (7-102).
- The Neural ODE's member band holds the burden less often than ours in
  29 of the 33 pairs.
- All 64 of our draws complete everywhere. 53 of the 2,112 have field
  errors above 25%, all in the stress arms from day 57.
- The PDE recovers (d, k, s) within 0.8%. Its d is 0.65-0.77% low on every
  acquisition, k is off by -0.06 to -0.14% and s by -0.19 to +0.01%. With
  the true parameters the errors change by at most 0.04 points from the
  first scan and 0.10 points restarted. So the restarted error comes from
  the noisy starting scan, not from the calibration.

**Disclosures.**
- **Float32 drift.** The Mac and the Azure machines differ by more than the
  declared "last bits". A 2-member, 50-update smoke on seed 45 showed final
  weights differing by up to 0.2% and errors by up to 0.6 points. All 180
  reported members ran on Azure, so a Mac rerun would not reproduce them
  bit for bit.
- **PDE scan-time check.** After the seed-45 smoke, and before any dense
  fit, it was relaxed from exact equality to 1e-9 days. On Linux the random
  scan times regenerate 1.4e-14 days from the stored ones, and the reduced
  scans agree to 1.5e-14. This is recorded in the baseline's
  `DECLARATION.md`.
- **pytest** was installed on the PDE machine to run the tests; no run
  uses it.
- **Shared machine.** The 9 PDE jobs ran next to Neural-ODE members on the
  same machine, on different cores.
- **After the fact:** as declared, the design was chosen after ours and the
  naive forecasts had been seen.
- **One virtual patient.** The seeds vary only the scan times and the
  noise.

**Cost.** The Azure resource group `rg-thesis-node-dense` was deleted at
18:30 UTC on Oct 27 and confirmed gone. Four F16as_v6 ran for 2.06
VM-hours ($2.26) and one E16as_v6 for up to 1.77 hours ($1.91). Disks and
IPs add $0.16, so the total is about $4.33.

**Paper.** The manuscript's tumour section now reports this design (local
commit, not pushed). It covers untreated growth, the planned course and
the two dose changes, with these brackets. The table `tab:tumor` is
generated from these summaries. The figure `tumor_dose_change.png` uses
seed 48, which was fixed before the Neural-ODE results existed. The
earlier single-dose and multi-dose cases stay reproducible here.

**Where.** Neural-ODE outputs are in
`experiments/tumor/results/benchmarks/<case>/segmented_observed_none_r4/seed*/neural_ode/`
and `evaluation/` (gitignored). The summary and paper scripts are local
session artifacts, not bundled with the repository:
- `.../files/node-dense/`: `node_dense_summary.py`, `paper_numbers.py`,
  `paper_table.py`, `paper_figure.py`, the machine logs and
  `figures/node_dense_*.png`;
- `.../files/standard-baseline/`: `calibrated_pde_dense.py`,
  `results_dense/`, `logs/dense_report.txt`, and `DECLARATION.md` with the
  dense amendment.

## Open decisions

The Neural-ODE ensembles on the dose-change design are done (results
above). The manuscript's tumour section now reports that design with the
free ROM and the input-aware GP (local commit, not pushed).
Tutorial-style figures are not done. With the free model, the only levers
were more information in training (a longer window, or a scan after a
recovery period). The after-the-fact check shows that the model change
works on the original window.

For the mid-course design, the Neural ODE should get the same known physics
for a like-for-like comparison. That means the same drug term, and no growth
without tumour. Its exact form (a hybrid with a learned remainder) is still
to be decided.

The manuscript now reports the free ROM with the input-aware GP, which
passes on the protocol's dose cuts. The method section is unchanged, and
the trend is described in the tumour section. The projected form is more
accurate, but it would add a model section; the paper mentions it only as
a variant outside the method. With the projected form the earlier tutorial-schedule
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
More past patients do not steadily help: with weekly scans to day 84 the
pooled error is lowest at 8-16 past patients and back near 4 by 64. The
likely limit is that every patient shares the cohort's drug term.
No version passes more than two of the four schedules, and every 8 days to
day 57 has none. The agreed list of ideas is done. The options are open:
- state the dense-imaging assumption in the paper (done: the tumour
  section states it);
- report the sparse-scan results as a limitation: the learned trend scale
  (one schedule passes) and partial pooling (both weekly schedules pass),
  beside the failures (done as an exploratory paragraph in the tumour
  section, with the single-patient failures and the restarted B and G
  results from the variant round);
- test the projected form on the realistic schedules, declared first
  (done in the variant round: E, F and G);
- move the example to a setting with frequent imaging;
- let each patient's drug response vary in the pooled model, for example
  one sensitivity factor per patient on the shared drug term, declared
  first (done in the variant round: C, D and G);
- confirm pooling at a declared number of past patients on new virtual
  patients (done in the variant round: B with eight past patients on
  vp4-9).

The variant round and the restart change this picture. Forecast from the
last scan, the best variants are under the 18.9% bar for most patients
with weekly or fortnightly scans to day 84 (G: 9 of 10 on both). Open:
- adopt the restart: move it from the session script into
  `benchmark_evaluation.py` with a test, and report both protocols;
- a Neural-ODE comparison on sparse scans, if wanted: G (no negative
  tumours, the smallest worst case on three of four schedules, lowest on
  weekly and fortnightly scans) or B (lowest mean on 4-weekly scans and weekly scans to day 57,
  by 1.2 points), both forecast from the last scan. The paper's comparison
  uses the dense design; the sparse results are an exploratory paragraph;
- confirm the chosen setting on fresh virtual patients, declared first;
- the remaining weakness: too little regrowth after a dose cut, so the
  0.5x bands often miss (G holds the truth 58-71% of the time);
- a declared misspecification pilot (proposed, not declared or run). The
  truth would get uneven drug delivery: the kill times a patchy map from
  the demo patient's contrast scan, about +-20%. A uniform-kill calibrated
  PDE cannot represent this, but a free drug block can.

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
`figures/pooled_error.png`, `figures/cohort_size_error.png`,
`figures/cohort_size_burden.png`, `figures/cohort_size_cohort.png`, the
band-coverage scripts, the
smoothed-POD scripts and fits, the scan-schedule check, summary and sweep
scripts, the sparse-fix check, summary and Gaussian-process check scripts,
the learned-trend summary script, the cohort data check, summary,
M-alone and block scripts, the pooled summary and member-check
scripts, and the cohort-size summary, M-forecast and member-rate
scripts) are local session artifacts, not
bundled with the repository:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-tutorial-runs/`.
The variant round's and the restart's scripts (`variant_summary.py`,
`restart_check.py`, `restart_expectations.py`, `restart_figures.py`), logs,
results and plots (`figures/variant_*.png`, `figures/restart_*.png`) are in
the sibling folder `.../files/azure-variants/`.
