"""Explicit tumor benchmark recipes, separate from historical entry-point defaults.

Here single-dose means one fixed dose strength, not one administration.
Both chemotherapy cases use repeated pulses; multi-dose changes future strengths.
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
        (20., 40., 60., 80., 100.), .5, (.5,), PODSettings()),
    "multi-dose-chemo": BenchmarkCase(
        "multi-dose-chemo", "Same half-strength past; different strengths of future pulses.",
        (5., 70.), 110., 120, .01, .025, .05, "cABN",
        (20., 40., 60., 80., 100.), .5, (.25, .5, .75, 1.), PODSettings()),
}


def get_case(name: str) -> BenchmarkCase:
    try:
        return CASES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tumor benchmark {name!r}; choose from {tuple(CASES)}.") from exc


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
    recipe = {**asdict(case), "pod": asdict(settings), "production_config": asdict(production_config(case, settings))}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return {
        **recipe, "recipe_fingerprint": fingerprint,
        "output_namespace": f"tumor_benchmarks_v1/{case.name}/{fingerprint[:16]}",
        "future_pulse_coefficients": {str(x): case.pulse_coefficients(x) for x in case.future_strengths},
        "POD_fitting_may_not_use_future_fields": True,
        "both_methods_must_share_observations_decoder_and_inputs": True,
        "description_only_does_not_run_or_claim_a_successful_experiment": True,
        "default_single_and_multi_dose_POD_is_the_preserved_control_not_an_outcome_selected_winner": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=tuple(CASES))
    parser.add_argument("--pod-rank", type=int)
    parser.add_argument("--pod-source", choices=("observed_training", "nominal_training", "matched_training"))
    parser.add_argument("--pod-centering", choices=("mean", "none"))
    args = parser.parse_args()
    if args.case is None:
        if any(value is not None for value in (args.pod_rank, args.pod_source, args.pod_centering)):
            parser.error("Select a benchmark before overriding its POD settings.")
        print(json.dumps({key: value.description for key, value in CASES.items()}, indent=2))
        return
    case = get_case(args.case)
    changes = {key: value for key, value in (
        ("rank", args.pod_rank), ("source", args.pod_source),
        ("centering", args.pod_centering)) if value is not None}
    try:
        pod = replace(case.pod, **changes)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(describe(case, pod), indent=2))


if __name__ == "__main__":
    main()
