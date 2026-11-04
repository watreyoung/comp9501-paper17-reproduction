"""Validation-selected, resumable public-data evaluation with saved predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from mil_pi.demo import default_parameters
from .data import make_task, export_mat_task
from .models import prepare_problem, fit_predict


def parameters_for(method, scale, config):
    base = "sMIL-PI" if method.endswith("-shuffled") else method
    p = {} if base in {"SVM", "sMIL"} else default_parameters(base, config["bag_size"])
    if base in {"SVM", "sMIL"}:
        p["baseline_C"] = 100.0*scale
    elif base.endswith("-DA"):
        p["svm_C1"] *= scale
    else:
        p["svm_C"] *= scale
    p.update(config.get("method_parameters", {}).get(base, {}))
    p.update(rho=config["sigma"], pos_ratio=config["sigma"],
             max_iter=config["outer_max_iter"], mkl_max_iter=config["mkl_max_iter"],
             solver_max_iter=config["solver_max_iter"])
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def store_record(log, records, record):
    """Keep one current record per fit; archive superseded failed attempts."""
    previous = next((r for r in records if r["record_id"] == record["record_id"]), None)
    if previous is not None:
        with log.with_name("superseded_attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(previous, allow_nan=False) + "\n")
        records[:] = [r for r in records if r["record_id"] != record["record_id"]]
    records.append(record)
    if previous is None:
        with log.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    else:
        temporary = log.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        temporary.replace(log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/msrvtt.json"))
    parser.add_argument("--output", type=Path, default=Path("results/msrvtt"))
    parser.add_argument("--phase", choices=["all", "tune", "evaluate"], default="all")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry failed records; preserve superseded attempts separately")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    predictions = output/"predictions"
    predictions.mkdir(exist_ok=True)
    # NpzFile is lazy: accessing it in each bag would repeatedly decompress
    # the complete feature matrix. Materialize once for the whole experiment.
    with np.load(config["cache"]) as stored:
        cache = {k: stored[k] for k in stored.files}
    if set(np.unique(cache["splits"])) != {"train", "validate", "test"}:
        raise ValueError("Dataset must have disjoint train/validation/test partitions")
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for d in [Path("mil_pi"), Path("experiments")]
                     for p in sorted(d.glob("*.py"))}
    identity = {"config": config, "source_hashes": source_hashes,
                "data_manifest_sha256": sha256_file("data/prepared/msrvtt_manifest.json"),
                "cache_sha256": sha256_file(config["cache"])}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    snapshot_path = output/"run_manifest.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Code/config/data changed: use a new output directory to avoid mixing results")
    if not snapshot_path.exists():
        write_json(snapshot_path, {**identity, "fingerprint": fingerprint,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                    "command": sys.argv, "started_unix": time.time()})
    log = output/"records.jsonl"
    records = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing = {r["record_id"]: r for r in records}

    def run_one(method, class_id, seed, noise, parameters, split, candidate):
        key = dict(method=method, class_id=class_id, seed=seed, noise=noise,
                   split=split, candidate=candidate, parameters=parameters)
        rid = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
        if rid in existing and (existing[rid]["status"] == "ok" or not args.retry_failed):
            return existing[rid]
        started = time.perf_counter()
        record = {**key, "record_id": rid}
        try:
            dataset, task_manifest = make_task(cache, class_id, seed=seed, noise=noise,
                n_bags=config["n_bags_per_sign"], bag_size=config["bag_size"], evaluation_split=split)
            problem = prepare_problem(dataset)
            scores, diagnostics = fit_predict(method, problem, parameters=parameters,
                backend=config["qp_backend"], seed=seed)
            # Held-out labels enter only metric calculation, after prediction.
            ap = float(average_precision_score(dataset.test_labels > 0, scores))
            record.update(status="ok", average_precision=ap, n_test=int(scores.size),
                          n_test_positive=int((dataset.test_labels > 0).sum()),
                          kernel_gamma=problem["kernel_gamma"], **diagnostics)
            np.savez_compressed(predictions/(rid+".npz"), scores=scores,
                labels=dataset.test_labels, video_ids=np.asarray(task_manifest["evaluation_video_ids"]))
            if method == config["methods"][0]:
                write_json(output/(rid+"_bags.json"), task_manifest)
        except Exception as error:
            record.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                          seconds=time.perf_counter()-started)
        store_record(log, records, record)
        existing[rid] = record
        ap_text = f"AP={record['average_precision']:.4f}" if record["status"] == "ok" else record["error"]
        print(f"{split} {method:20s} class={class_id:2d} seed={seed} noise={noise:.1f} scale={candidate}: {ap_text}", flush=True)
        return record

    with threadpool_limits(limits=1):
        if args.phase in {"all", "tune"}:
            chosen = {}
            validation = {}
            for method in [m for m in config["methods"] if not m.endswith("-shuffled")]:
                candidates = []
                for scale in config["regularization_scales"]:
                    p = parameters_for(method, scale, config)
                    rows = [run_one(method, c, config["validation_seed"], config["validation_noise"],
                                    p, "validate", scale) for c in config["classes"]]
                    candidate = {"scale": scale, "complete": all(r["status"] == "ok" for r in rows),
                                 "failures": sum(r["status"] != "ok" for r in rows)}
                    if candidate["complete"]:
                        candidate["mAP"] = float(np.mean([r["average_precision"] for r in rows]))
                    candidates.append(candidate)
                complete = [x for x in candidates if x["complete"]]
                validation[method] = candidates
                write_json(output/"validation_candidates.json", validation)
                if not complete:
                    raise RuntimeError(f"No fully successful validation candidate for {method}; see failed QPs")
                best = max(complete, key=lambda x: x["mAP"])
                chosen[method] = parameters_for(method, best["scale"], config)
            if "sMIL-PI-shuffled" in config["methods"]:
                chosen["sMIL-PI-shuffled"] = chosen["sMIL-PI"].copy()
            write_json(output/"selected_parameters.json", chosen)

        if args.phase in {"all", "evaluate"}:
            chosen = json.loads((output/"selected_parameters.json").read_text())
            for noise in config["noise_levels"]:
                for seed in config["seeds"]:
                    for c in config["classes"]:
                        for method in config["methods"]:
                            run_one(method, c, seed, noise, chosen[method], "test", "selected_on_validation")
            # Export one self-contained task in the exact original four-MAT layout.
            task, manifest = make_task(cache, config["classes"][0], seed=config["seeds"][0],
                noise=config["noise_levels"][-1], n_bags=config["n_bags_per_sign"],
                bag_size=config["bag_size"])
            export_mat_task(task, manifest, output/"mat_example")

    summary = []
    for noise in config["noise_levels"]:
        for method in config["methods"]:
            seed_maps = []
            failures = 0
            for seed in config["seeds"]:
                rows = [r for r in records if r["split"] == "test" and r["method"] == method
                        and r["noise"] == noise and r["seed"] == seed]
                failures += sum(r["status"] != "ok" for r in rows)
                if len(rows) == len(config["classes"]) and all(r["status"] == "ok" for r in rows):
                    seed_maps.append(float(np.mean([r["average_precision"] for r in rows])))
            row = {"noise": noise, "method": method, "complete_seeds": len(seed_maps), "failed_fits": failures}
            if seed_maps:
                row.update(mean_mAP=float(np.mean(seed_maps)),
                           std_mAP=float(np.std(seed_maps, ddof=1)) if len(seed_maps)>1 else 0.0,
                           seed_mAPs=seed_maps)
            summary.append(row)
    write_json(output/"summary.json", summary)
    print("Saved", output/"summary.json", flush=True)


if __name__ == "__main__":
    main()
