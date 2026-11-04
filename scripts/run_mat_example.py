"""Exercise the original four-MAT API and compare with the recorded benchmark."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from mil_pi.data import load_mat_dataset
from mil_pi.demo import METHODS, run_method
from mil_pi.qp import qp_session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/msrvtt"))
    args=parser.parse_args()
    root=args.results
    config=json.loads((root/"run_manifest.json").read_text())["config"]
    parameters=json.loads((root/"selected_parameters.json").read_text())
    records=[json.loads(s) for s in (root/"records.jsonl").read_text().splitlines()]
    data=load_mat_dataset(root/"mat_example" if (root/"mat_example").exists() else "data/tasks/msrvtt_example")
    checks=[]
    with threadpool_limits(limits=1):
        for method in METHODS:
            with qp_session("osqp"):
                _,scores,legacy_ap=run_method(method,data,overrides=parameters[method])
            expected=next(r for r in records if r["split"]=="test" and r["class_id"]==config["classes"][0]
                          and r["seed"]==config["seeds"][0] and r["noise"]==config["noise_levels"][-1] and r["method"]==method)
            with np.load(root/"predictions"/(expected["record_id"]+".npz")) as stored:
                np.testing.assert_allclose(scores,stored["scores"],atol=1e-10,rtol=1e-8)
                maximum=float(np.max(np.abs(scores-stored["scores"])))
            ap=float(average_precision_score(data.test_labels>0,scores))
            assert abs(ap-expected["average_precision"])<1e-12
            assert abs(legacy_ap-ap)<1e-12
            row={"method":method,"AP":ap,"original_API_AP":legacy_ap,
                 "max_score_difference":maximum,"status":"matched"}
            checks.append(row)
            print(json.dumps(row),flush=True)
    (root/"mat_interface_check.json").write_text(json.dumps(checks,indent=2)+"\n")


if __name__=="__main__":main()
