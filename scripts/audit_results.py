"""Independently recompute all saved AP values and check the delivered protocol."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np
from sklearn.metrics import average_precision_score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/msrvtt"))
    args=parser.parse_args()
    root=args.results
    run=json.loads((root/"run_manifest.json").read_text())
    config=run["config"]
    rows=[json.loads(s) for s in (root/"records.jsonl").read_text().splitlines()]
    assert len({r["record_id"] for r in rows})==len(rows), "duplicate fit records"
    assert all(r["status"]=="ok" for r in rows), "unsuccessful fits"
    for name,digest in run["source_hashes"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, f"source changed: {name}"
    assert hashlib.sha256(Path("data/prepared/msrvtt_manifest.json").read_bytes()).hexdigest()==run["data_manifest_sha256"]
    if "cache_sha256" in run:
        digest = hashlib.sha256()
        with Path(config["cache"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == run["cache_sha256"], "feature cache changed"
    for r in rows:
        if r["method"] in {"MI-SVM-PI", "MI-SVM-PI-DA"} and "outer_iterations" in r:
            assert r["outer_iterations"] == len(r["outer_objective_history"]) == len(r["label_changes"])
            assert r["outer_iterations"] == r["qp_solves"] > 0
            assert r["stop_reason"] in {"labels_stable", "label_change_tolerance", "max_iter", "no_support_vectors"}
    expected_test_keys={(m,c,s,n) for m in config["methods"] for c in config["classes"]
                        for s in config["seeds"] for n in config["noise_levels"]}
    actual_test_keys=[(r["method"],r["class_id"],r["seed"],r["noise"])
                      for r in rows if r["split"]=="test"]
    assert len(actual_test_keys)==len(expected_test_keys) and set(actual_test_keys)==expected_test_keys
    expected_validation_keys={(m,c,scale) for m in config["methods"] if not m.endswith("-shuffled")
                              for c in config["classes"] for scale in config["regularization_scales"]}
    actual_validation_keys=[(r["method"],r["class_id"],r["candidate"])
                            for r in rows if r["split"]=="validate"]
    assert len(actual_validation_keys)==len(expected_validation_keys) and set(actual_validation_keys)==expected_validation_keys
    selected=json.loads((root/"selected_parameters.json").read_text())
    candidates=json.loads((root/"validation_candidates.json").read_text())
    for method,options in candidates.items():
        for option in options:
            fits=[r for r in rows if r["split"]=="validate" and r["method"]==method
                  and r["candidate"]==option["scale"]]
            assert len(fits)==len(config["classes"]) and option["complete"]
            assert abs(np.mean([r["average_precision"] for r in fits])-option["mAP"])<1e-12
        best=max(options,key=lambda x:x["mAP"])
        corresponding=next(r for r in rows if r["split"]=="validate" and r["method"]==method
                           and r["candidate"]==best["scale"])
        # Runtime adds derived pos_num; compare the persisted requested parameters.
        assert all(corresponding["parameters"][k]==v for k,v in selected[method].items())
    assert selected["sMIL-PI-shuffled"]==selected["sMIL-PI"]
    for r in rows:
        if r["split"]=="test":
            assert all(r["parameters"][k]==v for k,v in selected[r["method"]].items())
    with np.load(config["cache"]) as stored:cache={k:stored[k] for k in stored.files}
    groups={split:set(cache["video_ids"][cache["splits"]==split]) for split in ["train","validate","test"]}
    assert [len(groups[s]) for s in ["train","validate","test"]]==[6513,497,2990]
    assert not (groups["train"]&groups["validate"] or groups["train"]&groups["test"] or groups["validate"]&groups["test"])
    assert np.count_nonzero(cache["text"][cache["splits"]!="train"])==0
    category_by_id=dict(zip(cache["video_ids"],cache["labels"]))
    max_ap_difference=0.0
    for r in rows:
        with np.load(root/"predictions"/(r["record_id"]+".npz")) as p:
            assert set(p["video_ids"])==groups[r["split"]]
            assert p["scores"].size==len(groups[r["split"]])
            expected_labels=np.array([1 if category_by_id[v]==r["class_id"] else -1 for v in p["video_ids"]])
            np.testing.assert_array_equal(p["labels"],expected_labels)
            assert np.isfinite(p["scores"]).all()
            ap=average_precision_score(p["labels"]>0,p["scores"])
            diff=abs(float(ap)-r["average_precision"])
            assert diff<1e-12
            max_ap_difference=max(max_ap_difference,diff)
        assert all(x["success"] and x["constraint_violation"]<=1e-6 for x in r["qp_diagnostics"])
    bag_files=list(root.glob("*_bags.json"))
    expected_bag_files=len(config["classes"])*(len(config["noise_levels"])*len(config["seeds"])
                                              +len(config["regularization_scales"]))
    assert len(bag_files)==expected_bag_files
    for path in bag_files:
        m=json.loads(path.read_text())
        pos=np.array(m["positive_video_ids"])
        neg=np.array(m["negative_video_ids"])
        source=set(pos.ravel())|set(neg.ravel())
        assert len(source)==2*config["n_bags_per_sign"]*config["bag_size"]
        assert source<=groups["train"]
        assert all(category_by_id[v]!=m["class_id"] for v in neg.ravel())
        actual_counts=[sum(category_by_id[v]==m["class_id"] for v in b) for b in pos]
        assert actual_counts==m["positive_bag_true_positive_counts"]
        assert set(m["evaluation_video_ids"])==groups[m["evaluation_split"]]
        assert all(x==config["bag_size"]-round(m["noise"]*config["bag_size"])
                   for x in m["positive_bag_true_positive_counts"])
    expected_test=len(config["classes"])*len(config["noise_levels"])*len(config["seeds"])*len(config["methods"])
    assert sum(r["split"]=="test" for r in rows)==expected_test
    summary=json.loads((root/"summary.json").read_text())
    for s in summary:
        seed_maps=[]
        for seed in config["seeds"]:
            subset=[r for r in rows if r["split"]=="test" and r["method"]==s["method"]
                    and r["noise"]==s["noise"] and r["seed"]==seed]
            assert len(subset)==len(config["classes"])
            seed_maps.append(np.mean([r["average_precision"] for r in subset]))
        assert abs(np.mean(seed_maps)-s["mean_mAP"])<1e-12
        assert abs(np.std(seed_maps,ddof=1)-s["std_mAP"])<1e-12
    report={"status":"passed","fit_records":len(rows),"test_fits":expected_test,
            "validation_fits":len(actual_validation_keys),"full_experiment_grid":"matched",
            "validation_selection_independently_recomputed":True,
            "prediction_APs_recomputed":len(rows),"max_AP_difference":max_ap_difference,
            "bag_manifests_checked":len(bag_files),"source_code_hashes":"matched",
            "cache_sha256_checked": "cache_sha256" in run,
            "train_validation_test_disjoint":True,"heldout_text_rows_zero":True,
            "fit_status_counts":dict(Counter(r["status"] for r in rows))}
    (root/"delivery_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
