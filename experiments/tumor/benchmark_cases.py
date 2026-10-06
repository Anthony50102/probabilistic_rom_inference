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
        if self.source not in ("observed_training", "nominal_training", "matched_training"):
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

    def __post_init__(self):
        if self.observation not in OBSERVATION_MODELS:
            raise ValueError(f"Unknown observation model: {self.observation}")
        if self.observation == "segmented" and self.pod.source != "observed_training":
            raise ValueError("Segmented acquisitions fit their POD to the noisy training scans "
                             "(observed_training); simulation bases belong to the oracle_masked design.")
        if self.structure not in STRUCTURES:
            raise ValueError(f"Unknown reduced-model structure: {self.structure}")
        if self.structure != "free" and not (self.dose_days and self.pod.centering == "none"):
            raise ValueError("The Galerkin structure holds for a chemotherapy ROM on an uncentred POD basis.")
        if self.input_aware_gp and not self.dose_days:
            raise ValueError("The input-aware Gaussian process needs a chemotherapy exposure table.")

    def pulse_coefficients(self, future_strength: float) -> tuple[float, ...]:
        if future_strength not in self.future_strengths:
            raise ValueError(f"{future_strength} is not a declared strength for {self.name}.")
        return tuple(self.sensitivity * (self.training_strength if day <= self.training_span[1]
                                         else future_strength) for day in self.dose_days)


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
    return replace(cfg, num_modes=settings.rank, operators=operators, gp_input_trend=case.input_aware_gp)


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
