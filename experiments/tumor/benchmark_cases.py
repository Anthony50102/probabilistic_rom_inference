"""Explicit tumor benchmark recipes, separate from historical entry-point defaults.

Here single-dose means one fixed dose strength, not one administration.
Both chemotherapy cases use repeated pulses; multi-dose changes future strengths.

Observation models
------------------
``segmented`` (reported): every scan, including the baseline, adds Gaussian
noise to every breast-tissue voxel. The lesion is segmented from the noisy
scan itself, and cellularity is reported only inside that region (clipped to
[0, theta]) and is zero elsewhere, as in TumorTwin's ADC-to-cellularity step.
The segmentation smooths the scan with a 1 mm Gaussian and thresholds it at a
level that holds the family-wise false-positive rate per scan at 5%. Each
POD is fitted to the acquisition's own noisy training scans.

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

    def __post_init__(self):
        if self.observation not in OBSERVATION_MODELS:
            raise ValueError(f"Unknown observation model: {self.observation}")
        if self.observation == "segmented" and self.pod.source != "observed_training":
            raise ValueError("Segmented acquisitions fit their POD to the noisy training scans "
                             "(observed_training); simulation bases belong to the oracle_masked design.")

    def pulse_coefficients(self, future_strength: float) -> tuple[float, ...]:
        if future_strength not in self.future_strengths:
            raise ValueError(f"{future_strength} is not a declared strength for {self.name}.")
        return tuple(.5 * (self.training_strength if day <= self.training_span[1]
                           else future_strength) for day in self.dose_days)


CASES = {
    "untreated-growth": BenchmarkCase(
        "untreated-growth", "Untreated faster-spreading tumor: next-month growth.",
        (5., 60.), 90., 40, .01, .05, .1, "cA", (), 0., (0.,),
        PODSettings(4, "observed_training", "mean")),
    "single-dose-chemo": BenchmarkCase(
        "single-dose-chemo", "Fixed half-strength regimen; repeated pulses, one strength.",
        (5., 70.), 110., 120, .01, .025, .05, "cABN",
        (20., 40., 60., 80., 100.), .5, (.5,), PODSettings(4, "observed_training", "mean")),
    "multi-dose-chemo": BenchmarkCase(
        "multi-dose-chemo", "Same half-strength past; different strengths of future pulses.",
        (5., 70.), 110., 120, .01, .025, .05, "cABN",
        (20., 40., 60., 80., 100.), .5, (.25, .5, .75, 1.),
        PODSettings(4, "observed_training", "mean")),
}

DEFAULT_POD_PROVENANCE = {
    "untreated-growth": "Production POD of the noisy training scans, as in the untreated-growth comparison.",
    "single-dose-chemo": "Production (mean-centred, rank 4) POD of the noisy training scans, declared in advance.",
    "multi-dose-chemo": "Production (mean-centred, rank 4) POD of the noisy training scans.",
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


def production_config(case: BenchmarkCase, pod: PODSettings | None = None):
    settings = case.pod if pod is None else pod
    schema = {
        "name": case.name, "label": case.description, "NUM_SAMPLES": case.observations,
        "NOISE_LEVEL": case.noise_level, "NUM_EVAL_POINTS": 200,
    }
    adapter = _adapter(bool(case.dose_days))
    cfg = adapter.make_config(schema, profile="historical") if case.dose_days else adapter.make_config(schema)
    return replace(cfg, num_modes=settings.rank)


def describe(case: BenchmarkCase, pod: PODSettings | None = None) -> dict:
    settings = case.pod if pod is None else pod
    legacy = case.observation == LEGACY_OBSERVATION
    fields = asdict(case)
    if legacy:
        # Keeps the fingerprints recorded by the earlier design reproducible.
        fields.pop("observation")
    recipe = {**fields, "pod": asdict(settings), "production_config": asdict(production_config(case, settings))}
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
