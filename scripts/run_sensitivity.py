import argparse, csv, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from controller import AgenticController, ControllerConfig, ControllerState


SWEEP = {
    "tau_infer":   [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "tau_enhance": [0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65],
    "tau_adapt":   [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "tau_q":       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "tau_max":     [40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 100.0],
    "E_max":       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "B_max":       [512, 700, 800, 1024, 1500, 2048, 4096],
}

NOMINAL = dict(tau_infer=0.25, tau_enhance=0.50, tau_adapt=0.75, tau_q=0.25,
               tau_max=60.0, E_max=0.25, M_max=512.0, B_max=1024.0)


def run_policy(posteriors, labels, cfg_kwargs: dict) -> dict:
    cfg  = ControllerConfig(**cfg_kwargs)
    ctrl = AgenticController(cfg)
    preds, deferred = [], []
    rng = np.random.default_rng(42)
    for i in range(len(labels)):
        p = posteriors[i]
        u = -np.sum(p * np.log(p + 1e-9)) / math.log(2)
        q = rng.uniform(0, 0.5)
        s = ControllerState(u_t=u, q_t=q, tau_t=22.0, E_t=0.17, M_t=460.0, B_t=800.0)
        a = ctrl.step(s)
        deferred.append(a == "Defer")
        preds.append(int(np.argmax(p)))

    preds    = np.array(preds)
    deferred = np.array(deferred)
    resolved = ~deferred
    mf1 = 0.0
    if resolved.sum() > 0:
        from evaluate import macro_f1_score
        mf1 = macro_f1_score(preds[resolved], labels[resolved])
    fnr = ((preds != 1) | deferred)[labels == 1].mean() if (labels == 1).sum() > 0 else 0.0
    return {
        "macro_f1":      round(mf1, 4),
        "fall_fnr_pct":  round(fnr * 100, 2),
        "deferred_pct":  round(deferred.mean() * 100, 2),
    }


def main(args):
    posteriors = np.load(args.posteriors)
    labels     = np.load(args.labels)

    rows = [["param", "value", "macro_f1", "fall_fnr_pct", "deferred_pct"]]
    for param, values in SWEEP.items():
        for val in values:
            kwargs = {**NOMINAL, param: val}
            res    = run_policy(posteriors, labels, kwargs)
            rows.append([param, val, res["macro_f1"], res["fall_fnr_pct"], res["deferred_pct"]])
        print(f"  {param} swept ({len(values)} values)")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "sensitivity.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posteriors", default="data/posteriors_test.npy")
    p.add_argument("--labels",     default="data/labels_test.npy")
    main(p.parse_args())
