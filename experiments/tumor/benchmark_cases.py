"""Explicit tumor benchmark recipes, separate from historical entry-point defaults.

Here single-dose means one fixed dose strength, not one administration.
Both chemotherapy cases use repeated pulses; multi-dose changes future strengths.

The two ``*-regimen-chemo`` cases follow TumorTwin's TNBC tutorial instead: the
tutorial's calibrated twin of TNBC_demo_001 on the patient's own schedule
(untreated until day 35, then twelve weekly doses to day 112), trained on days
5-57 (to the second MRI) and forecast to the third MRI on day 119. The
standard regimen keeps the remaining eight doses; the alternative regimens
rescale them, as in the tutorial's total-dose study.

The three ``midcourse-*-chemo`` cases adapt the dose mid-course on the same twin:
eight standard doses are observed (training days 5-84) and the last four
(days 91-112) are given at 0.5, 1, 1.5 or 2 times the standard dose. They differ
only in the reduced model's structure (``STRUCTURES``): ``free`` learns every
operator entry as in the other cases; ``galerkin`` imposes the form that the
projection of the reaction-diffusion PDE onto the uncentred POD basis implies
(no constant term, a diagonal drug-kill term) and ``galerkin_logistic`` adds the
projected logistic crowding term, each with learned rates. The two
``midcourse-*-trend-chemo`` cases repeat free and galerkin_logistic with the
input-aware Gaussian process (``input_aware_gp``: the cumulative training exposure
in each mode's GP mean), the one fix declared in TUMOR_TUTORIAL_SCHEDULE_CHEMO.md.
``alternative-regimen-logistic-trend-chemo`` fits that model to the earlier
days 5-57 split, after the fact, to separate the model change from the split.

The three free-ROM dose-modification cases keep the reported free ROM and add
only the input-aware Gaussian process: ``standard-regimen-trend-chemo`` (planned
course), ``reduced-regimen-trend-chemo`` (data to the second MRI) and
``midcourse-reduced-trend-chemo`` (eight doses seen). The last two give the
remaining doses at 1, 0.75 or 0.5x, the neuropathy dose reductions of the
paclitaxel protocol, plus 0.25x and 0x (stopped) as stress arms.

The ``*-scans`` cases repeat those last two with fixed, evenly spaced scan days
(``scan_days``; the same for every seed, so a seed changes only the scanner
noise) instead of 120 random ones: about weekly, fortnightly or four-weekly,
and (mid-course only) 120 evenly spaced scans. Their POD rank drops below 4
where fewer modes clear the scan-noise threshold.

The sparse-scan fix cases (``SPARSE_FIX_CASES``) keep that weekly scan count or
fewer. ``*-known-noise`` centres each mode's Gaussian-process noise prior on the
scans' known measurement noise (``gp_noise_prior="measurement"``,
``benchmark_data.scan_noise_variances``) instead of 1% of the mode's variance;
``*-every8-scans*`` scans every 8 days (``stride_scan_days``), so the scans fall
at different points of the weekly dose cycle; ``-r2`` keeps two POD modes. The
120-scan ``regular-scans-known-noise`` control checks the noise prior on dense
scans.

The learned-trend cases (``LEARNED_TREND_CASES``) repeat the weekly, every-8-day
and (mid-course) 120-scan schedules with the default noise prior but learn the
input-aware GP's trend-coefficient prior scale with the other GP hyperparameters
(``gp_trend_scale="learned"``) instead of fixing it at each mode's observed spread.

The population-prior cases (``COHORT_CASES``) repeat the four sparse
learned-trend cases on the POD basis of a virtual cohort of past patients
(``COHORT_DESIGN``, built by ``benchmark_cohort.py``): eight virtual patients
with the target's anatomy, varied growth, diffusion and drug sensitivity, three
of them with a dose modification, scanned through the whole course on the same
imaging protocol. ``*-cohort-basis`` keeps the zero-centred operator prior;
``*-cohort-prior`` centres it on the operator fitted to all the cohort's scans
(``operator_prior_mean="cohort"``); ``*-cohort-pooled`` fits the acquisition
together with the cohort's scans, each patient with its own operator around a
shared population operator, and learns how far patients' operators spread
around it (partial pooling, ``operator_prior_mean="pooled"``).

The cohort-size sweep (``COHORT_SWEEP_CASES``) repeats the weekly day-84 case
with the first K = 4, 8, 16, 32 or 64 past patients of a 64-member cohort
(``COHORT_SWEEP_DESIGN``; members 0-7 are the declared cohort's): their weekly
scans give the basis, and ``*-cohort{K}-pooled`` fits the acquisition jointly
with them (blocked partial-pooling solver), ``*-cohort{K}-basis`` alone.

Observation models
------------------
``segmented`` (reported): every scan, including the baseline, adds Gaussian
noise to every breast-tissue voxel. The lesion is segmented from the noisy
scan itself, and cellularity is reported only inside that region (clipped to
[0, theta]) and is zero elsewhere, as in TumorTwin's ADC-to-cellularity step.
The segmentation smooths the scan with a 1 mm Gaussian and thresholds it at a
level that holds the family-wise false-positive rate per scan at 5%. Each
POD is fitted to the acquisition's own noisy training scans, and its rank
never exceeds the number of modes whose singular values clear the scan-noise
threshold (``benchmark_data.noise_threshold``).

``oracle_masked`` (earlier design): noise only where the noise-free field
exceeds 0.1% of its maximum, a noise-free first scan, and noise-free
simulation bases for the chemotherapy cases.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OBSERVATION_MODELS = ("segmented", "oracle_masked")
LEGACY_OBSERVATION = "oracle_masked"
# Drug sensitivity per unit dose and pulse days of the original chemotherapy cases.
HISTORICAL_SENSITIVITY = .5
HISTORICAL_DOSE_DAYS = (20., 40., 60., 80., 100.)
# TNBC_demo_001.json: twelve weekly taxol doses, days 35-112 after the first MRI.
PATIENT_DOSE_DAYS = tuple(35. + 7. * week for week in range(12))
# TumorTwin TNBC tutorial, Levenberg-Marquardt fit to MRIs 1-2 ("Best parameters": d, k, sensitivity).
TUTORIAL_TWIN = {"diffusion": .0965, "growth_rate": .0521, "sensitivity": .23}
# Virtual cohort of past patients for the population-prior cases (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "A population
# prior from past patients"), declared before any fit; benchmark_cohort.py builds it. Every member has the target's
# anatomy and baseline tumour (as TumorTwin's TNBC demo patients do) and completed the course: twelve weekly doses
# from day 35, scans from day 5 to 119 on the target's imaging protocol, at a member-specific offset (member index
# mod the scan stride) in days from day 5, so their scans fall at different points of the weekly dose cycle. log k,
# log D and log sensitivity are the population centre plus log_sd times a standard normal; the centre is itself drawn
# once from that spread around the tutorial twin, so the target is a typical member rather than the median. Three
# members (37.5%) had a neuropathy dose modification: one dose held, then 75% (grade 2) or 50% (grade 3) for the
# rest. Scanner and segmentation are the target's (noise_level x the member's own field range); scan noise is drawn
# from default_rng([parameter_seed, member, stride]). The ROM input of every member is in the target's units:
# alpha(t) = input_sensitivity x sum of the given dose fractions x exp(-0.7 (t - dose day)).
COHORT_DESIGN = {
    "members": 8,
    "parameter_seed": 101,
    "log_sd": .3,
    "dose_modifications": ((1, 7, .75), (4, 5, .75), (6, 6, .5)),  # (member, held dose index 0-11, later fraction)
    "span": (5., 119.),
    "strides": {"weekly": 7., "every8": 8.},
    "rank": 4,
    "input_sensitivity": TUTORIAL_TWIN["sensitivity"],
}
# Cohort-size sweep (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "More past patients"), declared before any fit: the declared
# cohort's draw continued to 64 members, weekly protocol only. Members 0-7 are COHORT_DESIGN's (same parameters, doses,
# scan days and scan noise); member m has member (m mod 8)'s dose modification, so every block of eight has the
# declared mix. The first K members (K in sizes) are the past patients of the cohort-K cases and their pooled noisy
# weekly scans give the cohort-K basis. Partial pooling uses the blocked solver (core build_model pooled_solver).
COHORT_SWEEP_DESIGN = {
    **{key: COHORT_DESIGN[key] for key in ("parameter_seed", "log_sd", "dose_modifications", "span", "rank",
                                           "input_sensitivity")},
    "members": 64,
    "strides": {"weekly": 7.},
    "dose_modification_cycle": 8,
    "sizes": (4, 8, 16, 32, 64),
    "pooled_solver": "blocked",
}
# New patients from the cohort's population (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "Variant round", declared before any
# fit): log k, log D and log sensitivity are the tutorial twin's plus log_sd (centre_z + z), with centre_z the
# cohort's centre draw (COHORT_DESIGN's parameter_seed, first draw) and z the rows of default_rng(seed) standard
# normals of shape (patients, 3), rounded to four significant figures. Each has the tutorial twin's anatomy, baseline
# tumour, planned course (no dose modification), scanner and acquisition seeds; only k, D and the sensitivity differ.
# The first three are development patients (they pick the variant); the other six only confirm it.
VIRTUAL_PATIENT_DESIGN = {"patients": 9, "seed": 202, "log_sd": COHORT_DESIGN["log_sd"],
                          "centre": "COHORT_DESIGN parameter_seed, first draw", "significant_figures": 4,
                          "development": ("vp1", "vp2", "vp3"),
                          "confirmation": ("vp4", "vp5", "vp6", "vp7", "vp8", "vp9")}
VIRTUAL_PATIENTS = {
    name: dict(zip(("growth_rate", "diffusion", "sensitivity"), values)) for name, values in (
        ("vp1", (.07078, .04212, .199)), ("vp2", (.03644, .03709, .4333)), ("vp3", (.05352, .06346, .4207)),
        ("vp4", (.03432, .03534, .3059)), ("vp5", (.0451, .04055, .3237)), ("vp6", (.06393, .04466, .2922)),
        ("vp7", (.04558, .06064, .3746)), ("vp8", (.04904, .07976, .2162)), ("vp9", (.03568, .06703, .2414)))}
OPERATOR_PRIOR_MEANS = ("zero", "cohort", "pooled")
# Reduced-model structure: the operators a structured chemo ROM is fitted with (see benchmark_models.galerkin_row_maps).
STRUCTURES = ("free", "galerkin", "galerkin_logistic")
STRUCTURED_OPERATORS = {"galerkin": "AN", "galerkin_logistic": "AHN"}


@dataclass(frozen=True)
class PODSettings:
    rank: int = 4
    source: str = "nominal_training"
    centering: str = "mean"

    def __post_init__(self):
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("POD rank must be a positive integer.")
        if self.source not in ("observed_training", "nominal_training", "matched_training", "cohort_training"):
            raise ValueError(f"Unknown POD training source: {self.source}")
        if self.centering not in ("mean", "none"):
            raise ValueError(f"Unknown POD centering: {self.centering}")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    training_span: tuple[float, float]
    prediction_end: float
    observations: int
    noise_level: float
    growth_rate: float
    diffusion: float
    operators: str
    dose_days: tuple[float, ...]
    training_strength: float
    future_strengths: tuple[float, ...]
    pod: PODSettings
    observation: str = "segmented"
    sensitivity: float = HISTORICAL_SENSITIVITY
    structure: str = "free"
    input_aware_gp: bool = False
    # Fixed scan days shared by every acquisition; empty means random times per seed (benchmark_data.sample_times).
    scan_days: tuple[float, ...] = ()
    # Centre of each mode's GP noise prior: 1% of the mode's variance ("spectrum") or the scans' known measurement
    # noise ("measurement", benchmark_data.scan_noise_variances).
    gp_noise_prior: str = "spectrum"
    # Prior sd of the input-aware GP's trend coefficients: each mode's observed spread ("fixed") or that spread times a
    # multiplier learned with the other GP hyperparameters ("learned", WeakFormConfig.gp_trend_scale).
    gp_trend_scale: str = "fixed"
    # Past patients' imaging protocol ("weekly", "every8"; COHORT_DESIGN) whose pooled scans give the POD basis
    # (pod.source "cohort_training"); empty: the acquisition's own scans.
    cohort: str = ""
    # Centre of the operator prior: zero, the population operator fitted to the cohort's scans ("cohort"), or a
    # population operator learned jointly with the cohort members' own operators ("pooled": partial pooling).
    operator_prior_mean: str = "zero"
    # Number of past patients from the cohort-size sweep (COHORT_SWEEP_DESIGN; their scans also give the basis);
    # 0: the declared eight-member cohort (COHORT_DESIGN).
    cohort_size: int = 0
    # The new patient: the tutorial twin ("") or a declared virtual patient (VIRTUAL_PATIENTS), whose growth rate,
    # diffusion and sensitivity (the full-order truth) are its own.
    patient: str = ""
    # Drug sensitivity of the ROM's exposure input alpha(t) (its units); 0: the case's own sensitivity. A virtual
    # patient's inputs are in the cohort's units, since its own sensitivity is unknown when the model is fitted.
    input_sensitivity: float = 0.
    # Partial pooling only: every patient's exposure input is scaled by its own learned factor (core build_model
    # input_scale), e.g. its drug sensitivity relative to the input units.
    drug_factor: bool = False

    def __post_init__(self):
        if self.observation not in OBSERVATION_MODELS:
            raise ValueError(f"Unknown observation model: {self.observation}")
        if self.gp_noise_prior not in ("spectrum", "measurement"):
            raise ValueError(f"Unknown GP noise prior: {self.gp_noise_prior}")
        if self.gp_noise_prior == "measurement" and self.observation != "segmented":
            raise ValueError("The measurement-noise prior is derived from segmented scans.")
        if self.gp_trend_scale not in ("fixed", "learned"):
            raise ValueError(f"Unknown GP trend scale: {self.gp_trend_scale}")
        if self.gp_trend_scale == "learned" and not self.input_aware_gp:
            raise ValueError("A learned trend scale needs the input-aware GP's trend.")
        if self.observation == "segmented" and self.pod.source != (
                "cohort_training" if self.cohort else "observed_training"):
            raise ValueError("Segmented acquisitions fit their POD to the noisy training scans "
                             "(observed_training), or to a virtual cohort's (cohort_training, with a cohort); "
                             "simulation bases belong to the oracle_masked design.")
        if self.cohort and self.cohort not in COHORT_DESIGN["strides"]:
            raise ValueError(f"Unknown cohort imaging protocol: {self.cohort}")
        if self.cohort and not (self.observation == "segmented" and self.dose_days and self.pod.centering == "none"
                                and self.pod.rank <= COHORT_DESIGN["rank"]):
            raise ValueError("A cohort basis is the uncentred POD of segmented chemotherapy scans, "
                             f"at most rank {COHORT_DESIGN['rank']}.")
        if isinstance(self.cohort_size, bool) or not isinstance(self.cohort_size, int):
            raise ValueError("cohort_size is a whole number of past patients.")
        if self.cohort_size and not (self.cohort in COHORT_SWEEP_DESIGN["strides"]
                                     and self.cohort_size in COHORT_SWEEP_DESIGN["sizes"]
                                     and self.pod.rank <= COHORT_SWEEP_DESIGN["rank"]):
            raise ValueError(f"The cohort-size sweep has {COHORT_SWEEP_DESIGN['sizes']} past patients on the "
                             f"{tuple(COHORT_SWEEP_DESIGN['strides'])} protocol, at most rank "
                             f"{COHORT_SWEEP_DESIGN['rank']}.")
        if self.operator_prior_mean not in OPERATOR_PRIOR_MEANS:
            raise ValueError(f"Unknown operator prior mean: {self.operator_prior_mean}")
        if self.operator_prior_mean != "zero" and not (self.cohort and (
                self.structure == "free" or (self.cohort_size and self.operator_prior_mean == "pooled"))):
            raise ValueError("The population operator prior needs a cohort basis and the free ROM (partial pooling "
                             "in the cohort-size sweep also takes the projected form).")
        if not isinstance(self.drug_factor, bool):
            raise ValueError("drug_factor is True or False.")
        if self.drug_factor and not (self.operator_prior_mean == "pooled" and self.cohort_size):
            raise ValueError("A learned drug factor per patient needs partial pooling in the cohort-size sweep.")
        if self.patient:
            if self.patient not in VIRTUAL_PATIENTS:
                raise ValueError(f"Unknown virtual patient: {self.patient!r}")
            truth = VIRTUAL_PATIENTS[self.patient]
            if {name: getattr(self, name) for name in truth} != truth:
                raise ValueError(f"{self.patient}'s growth rate, diffusion and sensitivity are its declared ones.")
            if not (self.cohort_size and self.input_sensitivity == COHORT_SWEEP_DESIGN["input_sensitivity"]):
                raise ValueError("Virtual patients are new patients of the cohort-size sweep, with ROM inputs in "
                                 "the cohort's units.")
        elif self.input_sensitivity:
            raise ValueError("Only a virtual patient's ROM inputs are in other units than its own sensitivity.")
        if self.operator_prior_mean == "pooled" and self.gp_noise_prior != "spectrum":
            raise ValueError("Partial pooling uses the spectrum noise prior (the cohort has no measurement record).")
        if self.cohort_size and self.operator_prior_mean == "cohort":
            raise ValueError("The cohort-size sweep has no population-operator fit; use zero or pooled.")
        if self.structure not in STRUCTURES:
            raise ValueError(f"Unknown reduced-model structure: {self.structure}")
        if self.structure != "free" and not (self.dose_days and self.pod.centering == "none"):
            raise ValueError("The Galerkin structure holds for a chemotherapy ROM on an uncentred POD basis.")
        if self.input_aware_gp and not self.dose_days:
            raise ValueError("The input-aware Gaussian process needs a chemotherapy exposure table.")
        if self.scan_days:
            days = self.scan_days
            if len(days) != self.observations:
                raise ValueError("A fixed scan schedule lists one day per observation.")
            if (days[0] != self.training_span[0] or days[-1] > self.training_span[1]
                    or any(later <= earlier for earlier, later in zip(days, days[1:]))):
                raise ValueError("Fixed scan days must increase from the first training day and stay in the "
                                 "training window.")

    def pulse_coefficients(self, future_strength: float) -> tuple[float, ...]:
        if future_strength not in self.future_strengths:
            raise ValueError(f"{future_strength} is not a declared strength for {self.name}.")
        return tuple(self.sensitivity * (self.training_strength if day <= self.training_span[1]
                                         else future_strength) for day in self.dose_days)


def regular_scan_days(span, *, spacing=None, count=None):
    """Evenly spaced scan days over a training window, with the first and last scan at its ends.

    ``spacing`` (days) takes the whole number of intervals closest to it and rounds each day to a whole day;
    ``count`` gives exactly that many scans, unrounded.
    """
    start, end = span
    if (spacing is None) == (count is None) or (count is not None and count < 2):
        raise ValueError("Give a target spacing or a scan count of at least two.")
    intervals = max(1, int((end - start) / spacing + .5)) if count is None else count - 1
    days = [start + (end - start) * i / intervals for i in range(intervals + 1)]
    return tuple(float(int(day + .5)) if count is None else float(day) for day in days)


def stride_scan_days(span, stride):
    """A scan every ``stride`` days from the first training day, plus one on the last training day."""
    start, end = span
    if not stride > 0 or not end > start:
        raise ValueError("Give a positive stride and a training window that ends after it starts.")
    days = []
    while start + stride * len(days) < end:
        days.append(float(start + stride * len(days)))
    return tuple(days) + (float(end),)


CASES = {
    "untreated-growth": BenchmarkCase(
        "untreated-growth", "Untreated faster-spreading tumor: next-month growth.",
        (5., 60.), 90., 40, .01, .05, .1, "cA", (), 0., (0.,),
        PODSettings(3, "observed_training", "mean")),
    "single-dose-chemo": BenchmarkCase(
        "single-dose-chemo", "Fixed half-strength regimen; repeated pulses, one strength.",
        (5., 70.), 110., 120, .01, .025, .05, "cABN",
        HISTORICAL_DOSE_DAYS, .5, (.5,), PODSettings(4, "observed_training", "none")),
    "multi-dose-chemo": BenchmarkCase(
        "multi-dose-chemo", "Same half-strength past; different strengths of future pulses.",
        (5., 70.), 110., 120, .01, .025, .05, "cABN",
        HISTORICAL_DOSE_DAYS, .5, (.25, .5, .75, 1.),
        PODSettings(4, "observed_training", "none")),
    "standard-regimen-chemo": BenchmarkCase(
        "standard-regimen-chemo", "Tutorial twin, patient's weekly schedule: forecast the remaining planned doses.",
        (5., 57.), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"], "cABN",
        PATIENT_DOSE_DAYS, 1., (1.,), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"]),
    "alternative-regimen-chemo": BenchmarkCase(
        "alternative-regimen-chemo", "Same tutorial twin and past doses; remaining weekly doses rescaled.",
        (5., 57.), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"], "cABN",
        PATIENT_DOSE_DAYS, 1., (.5, 1., 1.5, 2.), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"]),
    "alternative-regimen-logistic-trend-chemo": BenchmarkCase(
        "alternative-regimen-logistic-trend-chemo",
        "As alternative-regimen-chemo, with the mid-course model: projected logistic form and input-aware GP.",
        (5., 57.), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"], "cABN",
        PATIENT_DOSE_DAYS, 1., (.5, 1., 1.5, 2.), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"], structure="galerkin_logistic", input_aware_gp=True),
    "standard-regimen-trend-chemo": BenchmarkCase(
        "standard-regimen-trend-chemo", "As standard-regimen-chemo (free ROM), with the input-aware Gaussian process.",
        (5., 57.), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"], "cABN",
        PATIENT_DOSE_DAYS, 1., (1.,), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"], input_aware_gp=True),
    **{name: BenchmarkCase(
        name, description, (5., end), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"],
        "cABN", PATIENT_DOSE_DAYS, 1., (0., .25, .5, .75, 1.), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"], input_aware_gp=True)
       for name, end, description in (
           ("reduced-regimen-trend-chemo", 57.,
            "Tutorial twin, data to the second MRI; remaining weekly doses kept, reduced or stopped "
            "(free ROM, input-aware GP)."),
           ("midcourse-reduced-trend-chemo", 84.,
            "Tutorial twin, eight weekly doses seen; last four kept, reduced or stopped (free ROM, input-aware GP)."))},
    **{name: BenchmarkCase(
        name, description, (5., 84.), 119., 120, .01, TUTORIAL_TWIN["growth_rate"], TUTORIAL_TWIN["diffusion"],
        "cABN", PATIENT_DOSE_DAYS, 1., (.5, 1., 1.5, 2.), PODSettings(4, "observed_training", "none"),
        sensitivity=TUTORIAL_TWIN["sensitivity"], structure=structure, input_aware_gp=trend)
       for name, structure, trend, description in (
           ("midcourse-regimen-chemo", "free", False,
            "Tutorial twin, eight standard weekly doses observed; last four doses rescaled (free ROM)."),
           ("midcourse-galerkin-chemo", "galerkin", False,
            "As midcourse-regimen-chemo; ROM with the projected PDE's form (no source, diagonal drug term)."),
           ("midcourse-logistic-chemo", "galerkin_logistic", False,
            "As midcourse-galerkin-chemo, plus the projected logistic crowding term."),
           ("midcourse-regimen-trend-chemo", "free", True,
            "As midcourse-regimen-chemo, with the input-aware Gaussian process."),
           ("midcourse-logistic-trend-chemo", "galerkin_logistic", True,
            "As midcourse-logistic-chemo, with the input-aware Gaussian process."))},
}
# Realistic scan schedules (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "Realistic scan schedules"): the free-ROM
# dose-modification cases with fixed, evenly spaced scan days instead of 120 random ones. Everything else is the
# base case's. The POD rank is the base's 4 unless fewer modes clear the scan-noise threshold on every acquisition
# (45 and 48-50, checked on the noisy scans alone before any fit). Two scans are not supported: the Gaussian
# process's lengthscale prior (median at the scan spacing, 99th percentile at the window) collapses.
SCAN_SPACING_DAYS = {"weekly": 7., "fortnightly": 14., "4weekly": 28.}
SCAN_SCHEDULE_RANKS = {
    "midcourse-reduced-trend-chemo": {"weekly": 4, "fortnightly": 4, "4weekly": 3, "regular": 4},
    "reduced-regimen-trend-chemo": {"weekly": 3, "fortnightly": 3, "4weekly": 3},
}


def _scan_schedule_case(base: str, schedule: str) -> BenchmarkCase:
    case = CASES[base]
    if schedule in SCAN_SPACING_DAYS:
        days = regular_scan_days(case.training_span, spacing=SCAN_SPACING_DAYS[schedule])
        label = f"about every {SCAN_SPACING_DAYS[schedule]:g} days, whole days"
    else:
        days, label = regular_scan_days(case.training_span, count=case.observations), "evenly spaced"
    return replace(case, name=f"{base}-{schedule}-scans", observations=len(days), scan_days=days,
                   description=f"As {base}, with {len(days)} fixed scans ({label}).",
                   pod=replace(case.pod, rank=SCAN_SCHEDULE_RANKS[base][schedule]))


SCAN_SCHEDULE_CASES = tuple(f"{base}-{schedule}-scans" for base, ranks in SCAN_SCHEDULE_RANKS.items()
                            for schedule in ranks)
CASES.update({f"{base}-{schedule}-scans": _scan_schedule_case(base, schedule)
              for base, ranks in SCAN_SCHEDULE_RANKS.items() for schedule in ranks})
# Fixing sparse scans (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "Fixing sparse scans"), declared before any fit: the
# weekly-count schedules with the scans' known measurement noise as the centre of the GP noise prior ("known-noise")
# and/or phase-diverse timing ("every8": a scan every 8 days, so each scan falls one day later in the weekly dose
# cycle than the one before, and no more scans than weekly). The "-r2" cases repeat every8 with known noise at POD
# rank 2 (side check: fewer unknowns); the regular-120 control checks that the known-noise prior does not hurt dense
# scans. Other ranks are the weekly schedules', checked on the noisy scans alone (acquisitions 45 and 48-50).
SCAN_STRIDE_DAYS = {"every8": 8.}
SPARSE_FIX_DESIGNS = {  # (schedule, known measurement noise, POD rank, name suffix)
    "midcourse-reduced-trend-chemo": (
        ("weekly", True, 4, "weekly-scans-known-noise"), ("every8", False, 4, "every8-scans"),
        ("every8", True, 4, "every8-scans-known-noise"), ("every8", True, 2, "every8-scans-known-noise-r2"),
        ("regular", True, 4, "regular-scans-known-noise")),
    "reduced-regimen-trend-chemo": (
        ("weekly", True, 3, "weekly-scans-known-noise"), ("every8", False, 3, "every8-scans"),
        ("every8", True, 3, "every8-scans-known-noise"), ("every8", True, 2, "every8-scans-known-noise-r2")),
}


def _sparse_fix_case(base: str, schedule: str, known_noise: bool, rank: int, suffix: str) -> BenchmarkCase:
    if schedule in SCAN_STRIDE_DAYS:
        stride, case = SCAN_STRIDE_DAYS[schedule], CASES[base]
        days = stride_scan_days(case.training_span, stride)
        case = replace(case, observations=len(days), scan_days=days)
        description = (f"As {base}, with {len(days)} fixed scans every {stride:g} days from the first training day "
                       "plus one on the last (each a day later in the weekly dose cycle)")
    else:
        case = CASES[f"{base}-{schedule}-scans"]
        description = f"As {case.name}"
    description += ("; GP noise prior centred on the scans' known noise" if known_noise else "") + (
        f"; POD rank {rank}" if rank != case.pod.rank else "") + "."
    return replace(case, name=f"{base}-{suffix}", description=description, pod=replace(case.pod, rank=rank),
                   gp_noise_prior="measurement" if known_noise else "spectrum")


SPARSE_FIX_CASES = tuple(f"{base}-{design[-1]}" for base, designs in SPARSE_FIX_DESIGNS.items() for design in designs)
CASES.update({f"{base}-{design[-1]}": _sparse_fix_case(base, *design)
              for base, designs in SPARSE_FIX_DESIGNS.items() for design in designs})

# Learned trend scale (declared after the sparse-fix diagnosis, before any fit): the same acquisitions as the weekly,
# every-8-day and (mid-course) 120-scan cases with the default noise prior; only the GP trend's prior scale is learned.
LEARNED_TREND_SOURCES = {
    "midcourse-reduced-trend-chemo": ("weekly-scans", "every8-scans", "regular-scans"),
    "reduced-regimen-trend-chemo": ("weekly-scans", "every8-scans"),
}
LEARNED_TREND_CASES = tuple(f"{base}-{source}-learned-trend"
                            for base, sources in LEARNED_TREND_SOURCES.items() for source in sources)
CASES.update({f"{base}-{source}-learned-trend": replace(
    CASES[f"{base}-{source}"], name=f"{base}-{source}-learned-trend", gp_trend_scale="learned",
    description=f"As {base}-{source}; the GP trend's prior scale is learned with the other GP hyperparameters.")
    for base, sources in LEARNED_TREND_SOURCES.items() for source in sources})
# Population prior from past patients (declared before any fit): the four sparse learned-trend cases, with the POD
# basis of a virtual cohort's scans on the same imaging protocol (COHORT_DESIGN) instead of the target's own, at the
# same rank. "-cohort-basis" keeps the zero-centred operator prior (the control: the basis alone); "-cohort-prior"
# centres it on the population operator fitted to the cohort (benchmark_cohort.fit_cohort). "-cohort-pooled" (partial
# pooling, declared before any of its fits) fits the acquisition jointly with the members' scans: each patient's
# operator is the population operator plus its own deviation, with the deviations' spread learned from all of them.
COHORT_SOURCES = {
    "midcourse-reduced-trend-chemo-weekly-scans-learned-trend": "weekly",
    "midcourse-reduced-trend-chemo-every8-scans-learned-trend": "every8",
    "reduced-regimen-trend-chemo-weekly-scans-learned-trend": "weekly",
    "reduced-regimen-trend-chemo-every8-scans-learned-trend": "every8",
}
COHORT_ARMS = {"cohort-basis": "zero", "cohort-prior": "cohort", "cohort-pooled": "pooled"}
COHORT_CASES = tuple(f"{source}-{arm}" for source in COHORT_SOURCES for arm in COHORT_ARMS)
POOLED_CASES = tuple(f"{source}-cohort-pooled" for source in COHORT_SOURCES)
CASES.update({f"{source}-{arm}": replace(
    CASES[source], name=f"{source}-{arm}", cohort=protocol, operator_prior_mean=mean,
    pod=replace(CASES[source].pod, source="cohort_training"),
    description=(f"As {source}, on the POD basis of a virtual cohort's {protocol} scans"
                 + {"zero": " (zero-centred operator prior).",
                    "cohort": "; operator prior centred on the cohort's population operator.",
                    "pooled": ("; fitted jointly with the cohort's scans, each patient's operator around a learned "
                               "population operator (partial pooling).")}[mean]))
    for source, protocol in COHORT_SOURCES.items() for arm, mean in COHORT_ARMS.items()})
# Cohort-size sweep (declared before any fit; COHORT_SWEEP_DESIGN): the weekly day-84 case with the first K past
# patients of the 64-member cohort, K in COHORT_SWEEP_DESIGN["sizes"]. "-cohort{K}-pooled" fits the acquisition jointly
# with their scans (partial pooling); "-cohort{K}-basis" is the control, the acquisition alone on the same basis.
COHORT_SWEEP_SOURCE = "midcourse-reduced-trend-chemo-weekly-scans-learned-trend"
COHORT_SWEEP_ARMS = {"basis": "zero", "pooled": "pooled"}
COHORT_SWEEP_CASES = tuple(f"{COHORT_SWEEP_SOURCE}-cohort{size}-{arm}"
                           for size in COHORT_SWEEP_DESIGN["sizes"] for arm in COHORT_SWEEP_ARMS)
CASES.update({f"{COHORT_SWEEP_SOURCE}-cohort{size}-{arm}": replace(
    CASES[COHORT_SWEEP_SOURCE], name=f"{COHORT_SWEEP_SOURCE}-cohort{size}-{arm}", cohort="weekly", cohort_size=size,
    operator_prior_mean=mean, pod=replace(CASES[COHORT_SWEEP_SOURCE].pod, source="cohort_training"),
    description=(f"As {COHORT_SWEEP_SOURCE}, on the POD basis of the weekly scans of the first {size} past patients of "
                 "the cohort-size sweep" + {"zero": " (zero-centred operator prior: the basis alone).",
                                             "pooled": "; fitted jointly with their scans (partial pooling)."}[mean]))
    for size in COHORT_SWEEP_DESIGN["sizes"] for arm, mean in COHORT_SWEEP_ARMS.items()})
# Variant round (TUMOR_TUTORIAL_SCHEDULE_CHEMO.md, "Variant round", declared before any fit): the cohort-size sweep's
# model and variants of it, for the tutorial twin and the virtual patients, on four of the new patient's imaging
# schedules up to the treatment decision (the past patients are the sweep's, weekly, days 5-119). Arms: the new
# patient alone on the past patients' basis ("basis") or pooled with them ("pooled"), each in the free form or the
# projected PDE form ("projected-": galerkin_logistic), and pooled with every patient's own learned drug factor
# ("-drug"). The fortnightly and 4-weekly sources are the fixed-schedule cases with the learned trend scale.
VARIANT_LEARNED_TREND_CASES = tuple(f"midcourse-reduced-trend-chemo-{schedule}-scans-learned-trend"
                                    for schedule in ("fortnightly", "4weekly"))
CASES.update({name: replace(
    CASES[name.removesuffix("-learned-trend")], name=name, gp_trend_scale="learned",
    description=f"As {name.removesuffix('-learned-trend')}; the GP trend's prior scale is learned with the other GP "
                "hyperparameters.") for name in VARIANT_LEARNED_TREND_CASES})
VARIANT_SOURCES = {
    COHORT_SWEEP_SOURCE: "weekly scans to day 84",
    VARIANT_LEARNED_TREND_CASES[0]: "fortnightly scans to day 84",
    VARIANT_LEARNED_TREND_CASES[1]: "4-weekly scans to day 84",
    "reduced-regimen-trend-chemo-weekly-scans-learned-trend": "weekly scans to day 57",
}
VARIANT_ARMS = {  # arm: (structure, operator prior mean, learned drug factor)
    "basis": ("free", "zero", False),
    "pooled": ("free", "pooled", False),
    "pooled-drug": ("free", "pooled", True),
    "projected-basis": ("galerkin_logistic", "zero", False),
    "projected-pooled": ("galerkin_logistic", "pooled", False),
    "projected-pooled-drug": ("galerkin_logistic", "pooled", True),
}
VARIANT_SIZE_ARMS = {8: tuple(VARIANT_ARMS), 64: ("pooled-drug",)}
VARIANT_PATIENTS = ("",) + tuple(VIRTUAL_PATIENTS)


def variant_name(source: str, size: int, arm: str, patient: str = "") -> str:
    return f"{source}-cohort{size}-{arm}" + (f"-{patient}" if patient else "")


def _variant_case(source: str, size: int, arm: str, patient: str) -> BenchmarkCase:
    structure, mean, drug = VARIANT_ARMS[arm]
    base = CASES[source]
    changes = dict(VIRTUAL_PATIENTS[patient], patient=patient,
                   input_sensitivity=COHORT_SWEEP_DESIGN["input_sensitivity"]) if patient else {}
    who = (f"virtual patient {patient} (k, D, sensitivity {tuple(VIRTUAL_PATIENTS[patient].values())}; ROM inputs "
           "in the cohort's units)" if patient else "the tutorial twin")
    fit = {"zero": "alone (zero-centred operator prior)", "pooled": "jointly with their scans (partial pooling)"}[mean]
    return replace(
        base, name=variant_name(source, size, arm, patient), cohort="weekly", cohort_size=size,
        operator_prior_mean=mean, structure=structure, drug_factor=drug,
        pod=replace(base.pod, source="cohort_training"), **changes,
        description=(f"As {source}, for {who}: on the POD basis of the weekly scans of the first {size} past patients "
                     f"of the cohort-size sweep, fitted {fit}"
                     + (", projected PDE form (galerkin_logistic)" if structure != "free" else "")
                     + ("; every patient's exposure input scaled by its own learned drug factor" if drug else "")
                     + "."))


VARIANT_CASES = tuple(variant_name(source, size, arm, patient) for source in VARIANT_SOURCES
                      for patient in VARIANT_PATIENTS for size, arms in VARIANT_SIZE_ARMS.items() for arm in arms)
# The cohort-size sweep's own cases (the tutorial twin, weekly to day 84) keep their definitions.
CASES.update({variant_name(*key): _variant_case(*key)
              for key in ((source, size, arm, patient) for source in VARIANT_SOURCES for patient in VARIANT_PATIENTS
                          for size, arms in VARIANT_SIZE_ARMS.items() for arm in arms)
              if variant_name(*key) not in CASES})
# The three benchmarks reported in the paper (the default when a runner is given no case).
REPORTED_CASES = ("untreated-growth", "single-dose-chemo", "multi-dose-chemo")

_CHEMO_SELECTION = ("chosen on development acquisition 45 among observed mean/none x rank 4/3 by the multi-dose rule "
                    "of experiments/TUMOR_BENCHMARK_POD_COMPARISON.md (lowest mean production error over the four "
                    "future strengths; centred bases failed)")
DEFAULT_POD_PROVENANCE = {
    "untreated-growth": ("Mean-centred POD of the noisy training scans; rank 3 because the fourth singular value "
                         "lies inside the noise bulk, below the scan-noise threshold, on every acquisition."),
    "single-dose-chemo": f"Uncentred rank-4 POD of the noisy training scans (as multi-dose); {_CHEMO_SELECTION}.",
    "multi-dose-chemo": f"Uncentred rank-4 POD of the noisy training scans; {_CHEMO_SELECTION}.",
    "standard-regimen-chemo": ("Uncentred rank-4 POD of the noisy training scans (as alternative-regimen), selected on "
                               "development acquisition 45 of alternative-regimen-chemo."),
    "alternative-regimen-chemo": ("Uncentred rank-4 POD of the noisy training scans, selected on development acquisition "
                                  "45 by the lowest mean production forecast_57_119 error over the four future strengths "
                                  "(none r4 18.2%, none r3 22.5%; centred bases diverged at the higher strengths); "
                                  "see experiments/TUMOR_TUTORIAL_SCHEDULE_CHEMO.md."),
    "alternative-regimen-logistic-trend-chemo": ("Uncentred rank-4 POD of the noisy training scans, as "
                                                 "alternative-regimen-chemo (the same acquisitions; not reselected)."),
    **{name: ("Uncentred rank-4 POD of the noisy training scans, as alternative-regimen-chemo (the same "
              "acquisitions; not reselected).")
       for name in ("standard-regimen-trend-chemo", "reduced-regimen-trend-chemo")},
    "midcourse-reduced-trend-chemo": ("Uncentred rank-4 POD of the noisy training scans, as the midcourse-*-chemo "
                                      "cases (carried over from alternative-regimen-chemo; not reselected)."),
    **{name: ("Uncentred rank-4 POD of the noisy training scans, carried over from alternative-regimen-chemo (not "
              "reselected); the Galerkin structure needs an uncentred basis, and preparation checks that rank 4 clears "
              "the scan-noise threshold.")
       for name in ("midcourse-regimen-chemo", "midcourse-galerkin-chemo", "midcourse-logistic-chemo",
                    "midcourse-regimen-trend-chemo", "midcourse-logistic-trend-chemo")},
    **{f"{base}-{schedule}-scans": (f"Uncentred rank-{rank} POD of the noisy training scans: {base}'s rank 4, "
                                    "lowered to the modes that clear the scan-noise threshold on acquisitions 45 and "
                                    "48-50 (checked on the scans alone, before any fit).")
       for base, ranks in SCAN_SCHEDULE_RANKS.items() for schedule, rank in ranks.items()},
    **{f"{base}-{suffix}": (f"Uncentred rank-{rank} POD of the noisy training scans: "
                            + ("deliberately two modes (side check: fewer unknowns)." if rank == 2 else
                               f"as {base}-{schedule}-scans." if schedule not in SCAN_STRIDE_DAYS else
                               f"the weekly schedule's rank {rank}, which also clears the scan-noise threshold here "
                               "on acquisitions 45 and 48-50 (checked on the scans alone, before any fit)."))
       for base, designs in SPARSE_FIX_DESIGNS.items() for schedule, _, rank, suffix in designs},
    **{f"{base}-{source}-learned-trend": f"As {base}-{source} (the same acquisitions and basis; not reselected)."
       for base, sources in LEARNED_TREND_SOURCES.items() for source in sources},
    **{f"{source}-{arm}": (f"Uncentred POD of the virtual cohort's noisy segmented {protocol} scans (8 past patients, "
                           f"days 5-119; benchmark_cohort.py), first {CASES[source].pod.rank} modes: {source}'s rank, "
                           "within the modes that clear the pooled scans' noise threshold. This acquisition's scans "
                           "are only projected onto it.")
       for source, protocol in COHORT_SOURCES.items() for arm in COHORT_ARMS},
    **{f"{COHORT_SWEEP_SOURCE}-cohort{size}-{arm}": (
        f"Uncentred POD of the noisy segmented weekly scans of the first {size} past patients of the cohort-size sweep "
        f"(days 5-119; benchmark_cohort.py build-sweep), first {CASES[COHORT_SWEEP_SOURCE].pod.rank} modes: "
        f"{COHORT_SWEEP_SOURCE}'s rank, checked against the pooled scans' noise threshold when the cohort is built. "
        "This acquisition's scans are only projected onto it.")
       for size in COHORT_SWEEP_DESIGN["sizes"] for arm in COHORT_SWEEP_ARMS},
    **{name: f"As {name.removesuffix('-learned-trend')} (the same acquisitions and basis; not reselected)."
       for name in VARIANT_LEARNED_TREND_CASES},
    **{name: (f"Uncentred POD of the noisy segmented weekly scans of the first {CASES[name].cohort_size} past patients "
              f"of the cohort-size sweep (as {COHORT_SWEEP_SOURCE}-cohort{CASES[name].cohort_size}-*), first "
              f"{CASES[name].pod.rank} modes: the source schedule's rank. This acquisition's scans are only projected "
              "onto it.")
       for name in VARIANT_CASES if name not in COHORT_SWEEP_CASES},
}

# The earlier oracle-masked design, kept reproducible under its original output paths.
LEGACY_PODS = {
    "untreated-growth": PODSettings(4, "observed_training", "mean"),
    "single-dose-chemo": PODSettings(),
    "multi-dose-chemo": PODSettings(4, "matched_training", "none"),
}
LEGACY_POD_PROVENANCE = {
    "untreated-growth": "Preserved observed-training basis from the untreated-growth comparison.",
    "single-dose-chemo": "Preserved nominal-training control from the fixed-strength comparison.",
    "multi-dose-chemo": ("Selected on development acquisition 45 and confirmed on fresh acquisitions 48-50; "
                         "see experiments/TUMOR_BENCHMARK_POD_COMPARISON.md."),
}


def get_case(name: str, observation: str | None = None) -> BenchmarkCase:
    try:
        case = CASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tumor benchmark {name!r}; choose from {tuple(CASES)}.") from exc
    if observation is None or observation == case.observation:
        return case
    if observation == LEGACY_OBSERVATION:
        if name not in LEGACY_PODS:
            raise ValueError(f"The {observation} design predates {name!r}.")
        return replace(case, observation=observation, pod=LEGACY_PODS[name])
    raise ValueError(f"Unknown observation model: {observation}")


@lru_cache(maxsize=2)
def _adapter(chemo: bool):
    folder = Path(__file__).resolve().parent
    filename = "04_unified_chemo.py" if chemo else "04_unified.py"
    old_path = sys.path[:]
    previous = sys.modules.get("config")
    try:
        sys.path[:0] = [str(folder), str(folder.parents[1])]
        config_spec = importlib.util.spec_from_file_location("_tumor_benchmark_config", folder / "config.py")
        config = importlib.util.module_from_spec(config_spec)
        sys.modules["config"] = config
        config_spec.loader.exec_module(config)
        spec = importlib.util.spec_from_file_location("_tumor_benchmark_" + filename[:-3], folder / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if previous is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous


def galerkin_operators(case: BenchmarkCase, pod: PODSettings) -> str:
    """ROM blocks implied by Galerkin projection of the kill term -alpha(t) u.

    The term projects to N alpha q with N = -I and, for a mean-centred basis u = ubar + V q,
    to a direct input term B alpha with B = -V^T ubar. B vanishes for an uncentred basis and
    is omitted there: a learned B could only absorb misfit that is correlated with the training
    regimen and would carry it to the changed doses.
    """
    if not case.dose_days or pod.centering == "mean":
        return case.operators
    return case.operators.replace("B", "")


def production_config(case: BenchmarkCase, pod: PODSettings | None = None):
    settings = case.pod if pod is None else pod
    if case.structure != "free" and settings.centering != "none":
        raise ValueError("The Galerkin structure holds for a chemotherapy ROM on an uncentred POD basis.")
    schema = {
        "name": case.name, "label": case.description, "NUM_SAMPLES": case.observations,
        "NOISE_LEVEL": case.noise_level, "NUM_EVAL_POINTS": 200,
    }
    adapter = _adapter(bool(case.dose_days))
    cfg = adapter.make_config(schema, profile="historical") if case.dose_days else adapter.make_config(schema)
    if case.observation == LEGACY_OBSERVATION:
        # The earlier design was fitted with the dimensional operator prior, relative GP nugget,
        # dimensional slack, its hand-set weights and the ridge-regularised normal equations.
        cfg = replace(cfg, sigma_O=5.0, gp_jitter_rel=1e-3, closure="slack", gamma2=0.035, weak_slack="legacy",
                      mll_weight=0.1, weakform_weight=8.0, operator_solver="normal")
        return replace(cfg, num_modes=settings.rank)
    operators = STRUCTURED_OPERATORS.get(case.structure) or galerkin_operators(case, settings)
    return replace(cfg, num_modes=settings.rank, operators=operators, gp_input_trend=case.input_aware_gp,
                   gp_noise_prior=case.gp_noise_prior, gp_trend_scale=case.gp_trend_scale)


def describe(case: BenchmarkCase, pod: PODSettings | None = None) -> dict:
    settings = case.pod if pod is None else pod
    legacy = case.observation == LEGACY_OBSERVATION
    fields = asdict(case)
    config = asdict(production_config(case, settings))
    if fields["sensitivity"] == HISTORICAL_SENSITIVITY:
        # Recipes from before the per-case drug sensitivity keep their recorded fingerprints.
        fields.pop("sensitivity")
    if fields["structure"] == "free":
        # Likewise for recipes from before the structured ROMs.
        fields.pop("structure")
    if not fields["input_aware_gp"]:
        fields.pop("input_aware_gp")
    if not fields["scan_days"]:
        # Random scan times (every case before the fixed schedules) keep their recorded fingerprints.
        fields.pop("scan_days")
    if fields["gp_noise_prior"] == "spectrum":
        # Likewise for recipes from before the measurement-noise prior.
        fields.pop("gp_noise_prior")
    if fields["gp_trend_scale"] == "fixed":
        # And from before the learned trend scale, whose config field is likewise left out.
        fields.pop("gp_trend_scale")
        config.pop("gp_trend_scale")
    if fields["cohort"]:
        # The cohort's declared design is part of the recipe (benchmark_cohort.py builds it from these values).
        fields["cohort_design"] = COHORT_SWEEP_DESIGN if case.cohort_size else COHORT_DESIGN
    else:
        fields.pop("cohort")
    if not fields["cohort_size"]:
        # The declared eight-member cohort's recipes keep their recorded fingerprints.
        fields.pop("cohort_size")
    if fields["operator_prior_mean"] == "zero":
        fields.pop("operator_prior_mean")
    # The tutorial twin with inputs in its own units and no learned drug factor: every recipe before the variant round.
    for key, default in (("patient", ""), ("input_sensitivity", 0.), ("drug_factor", False)):
        if fields[key] == default:
            fields.pop(key)
    if fields.get("patient"):
        fields["virtual_patient_design"] = VIRTUAL_PATIENT_DESIGN
    if legacy:
        # Keeps the fingerprints recorded by the earlier design reproducible.
        fields.pop("observation")
        for later in ("gamma2_nd", "weak_slack", "closure"):
            config.pop(later)
    recipe = {**fields, "pod": asdict(settings), "production_config": config}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    default = LEGACY_PODS[case.name] if legacy else CASES[case.name].pod
    return {
        **recipe, "observation": case.observation, "recipe_fingerprint": fingerprint,
        "output_namespace": f"tumor_benchmarks_v1/{case.name}/{fingerprint[:16]}",
        "future_pulse_coefficients": {str(x): case.pulse_coefficients(x) for x in case.future_strengths},
        "POD_fitting_may_not_use_future_fields": True,
        "both_methods_must_share_observations_decoder_and_inputs": True,
        "description_only_does_not_run_or_claim_a_successful_experiment": True,
        "uses_default_POD": settings == default,
        "default_POD_provenance": (LEGACY_POD_PROVENANCE if legacy else DEFAULT_POD_PROVENANCE)[case.name],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=tuple(CASES))
    parser.add_argument("--observation", choices=OBSERVATION_MODELS)
    parser.add_argument("--pod-rank", type=int)
    parser.add_argument("--pod-source", choices=("observed_training", "nominal_training", "matched_training"))
    parser.add_argument("--pod-centering", choices=("mean", "none"))
    args = parser.parse_args()
    if args.case is None:
        if any(value is not None for value in (args.pod_rank, args.pod_source, args.pod_centering,
                                               args.observation)):
            parser.error("Select a benchmark before overriding its observation or POD settings.")
        print(json.dumps({key: value.description for key, value in CASES.items()}, indent=2))
        return
    case = get_case(args.case, args.observation)
    changes = {key: value for key, value in (
        ("rank", args.pod_rank), ("source", args.pod_source),
        ("centering", args.pod_centering)) if value is not None}
    try:
        pod = replace(case.pod, **changes)
        replace(case, pod=pod)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(describe(case, pod), indent=2))


if __name__ == "__main__":
    main()
