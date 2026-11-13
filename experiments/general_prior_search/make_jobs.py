"""Write the predeclared development or frozen-shortlist confirmation queues."""

import argparse
import json
from pathlib import Path

from experiments.general_prior_search.run import CANDIDATES


DEVELOPMENT = [
    dict(system="cylinder", schema="dense_low_noise", modes=25),
    dict(system="heat", schema="sparse_medium_noise"),
    dict(system="euler", schema="dense_low_noise"),
    dict(system="burgers_2d", schema="dense_medium_noise"),
]


def jobs(stage, shortlist):
    if stage == "development":
        cases = [dict(case, seed=42) for case in DEVELOPMENT]
        candidates, draws, long_draws = list(CANDIDATES), 200, 20
    elif stage == "confirmation":
        if not shortlist or "original" in shortlist or len(shortlist) > 3 or len(set(shortlist)) != len(shortlist):
            raise ValueError("Supply one to three distinct shortlisted candidates, excluding original")
        if any(name not in CANDIDATES for name in shortlist):
            raise ValueError("Unknown shortlisted candidate")
        cases = [
            dict(system="cylinder", schema="dense_low_noise", modes=29, seed=42),
            dict(system="cylinder", schema="dense_low_noise", modes=25, seed=43),
            dict(system="cylinder", schema="dense_low_noise", modes=25, seed=47, window_start=20.),
            dict(system="heat", schema="sparse_low_noise", seed=42),
            dict(system="heat", schema="sparse_high_noise", seed=42),
            dict(system="euler", schema="sparse_low_noise", seed=42),
            dict(system="euler", schema="dense_high_noise", seed=42),
        ] + [dict(case, seed=47) for case in DEVELOPMENT]
        candidates, draws, long_draws = ["original", *shortlist], 500, 50
    else:
        raise ValueError(f"Unknown stage: {stage}")
    output = []
    for case in cases:
        for candidate in candidates:
            arguments = dict(case, candidate=candidate, steps=12000, draws=draws, long_draws=long_draws)
            name = (f"{case['system']}-{case['schema']}-r{case.get('modes', 'default')}-s{case['seed']}"
                    f"-w{case.get('window_start', 16):g}-{candidate}")
            output.append(dict(id=name, arguments=arguments))
    return output


def main(args):
    queue = jobs(args.stage, args.shortlist.split(",") if args.shortlist else [])
    args.output.mkdir(parents=True, exist_ok=True)
    for path in [args.output / "all_jobs.json", *[args.output / f"prior-{host}.json" for host in "abcd"]]:
        if path.exists():
            raise FileExistsError(f"Do not overwrite a declared queue: {path}")
    (args.output / "all_jobs.json").write_text(json.dumps(queue, indent=2) + "\n")
    for index, host in enumerate("abcd"):
        (args.output / f"prior-{host}.json").write_text(json.dumps(queue[index::4], indent=2) + "\n")
    print(f"{len(queue)} jobs, {[len(queue[i::4]) for i in range(4)]} per worker")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--shortlist", default="")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
