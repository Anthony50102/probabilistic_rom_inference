"""Bounded remote job queue with explicit CPU affinity and durable failures."""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


def main(args):
    jobs = json.loads(args.jobs.read_text())
    if args.slots < 1 or args.cpus_per_slot < 1:
        raise ValueError("Positive slot and CPU counts are required")
    available = sorted(os.sched_getaffinity(0))
    if args.slots * args.cpus_per_slot > len(available):
        raise ValueError("Requested CPU slots exceed the process affinity")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = queue.Queue()
    for job in jobs:
        tasks.put(job)
    statuses, lock = [], threading.Lock()

    def worker(slot):
        cpus = available[slot*args.cpus_per_slot:(slot+1)*args.cpus_per_slot]
        while True:
            try:
                job = tasks.get_nowait()
            except queue.Empty:
                return
            destination = args.output / job["id"]
            destination.mkdir(parents=True, exist_ok=True)
            record_path = destination / "job.json"
            if record_path.exists():
                previous = json.loads(record_path.read_text())
                if previous["job"] != job or previous["source_commit"] != args.source_commit:
                    raise ValueError(f"Existing job {job['id']} belongs to another request")
                if previous.get("returncode") == 0:
                    with lock:
                        statuses.append(previous)
                    continue
            command = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable,
                       str(Path(__file__).with_name("run.py")), "--output", str(destination),
                       "--source-commit", args.source_commit]
            for name, value in job["arguments"].items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
            record = dict(job=job, source_commit=args.source_commit, cpus=cpus, started=time.time())
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"START {job['id']} slot={slot}", flush=True)
            with (destination / "run.log").open("a") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(finished=time.time(), returncode=result.returncode)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"END {job['id']} rc={result.returncode}", flush=True)
            with lock:
                statuses.append(record)

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(args.slots)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (args.output / "batch.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if len(statuses) != len(jobs) or any(row["returncode"] for row in statuses):
        raise SystemExit("Some jobs failed; inspect each job.json and run.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--cpus-per-slot", type=int, default=4)
    main(parser.parse_args())
