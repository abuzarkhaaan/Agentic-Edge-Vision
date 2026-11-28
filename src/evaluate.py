import argparse, csv, math, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import bootstrap

from model      import AgenticViT
from dataset    import (WindowDataset, build_windows, load_split_file,
                        eval_transform, apply_corruption)
from controller import AgenticController, ControllerConfig, ControllerState


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    conf   = probs.max(axis=1)
    preds  = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc       = correct[mask].mean()
        mean_conf = conf[mask].mean()
        ece      += mask.sum() / len(probs) * abs(acc - mean_conf)
    return ece


def macro_f1_score(preds: np.ndarray, labels: np.ndarray, K: int = 2) -> float:
    f1s = []
    for k in range(K):
        tp = ((preds == k) & (labels == k)).sum()
        fp = ((preds == k) & (labels != k)).sum()
        fn = ((preds != k) & (labels == k)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def fall_fnr(preds: np.ndarray, labels: np.ndarray,
             deferred_mask: np.ndarray | None = None) -> float:
    fall_mask = (labels == 1)
    if deferred_mask is None:
        deferred_mask = np.zeros(len(labels), dtype=bool)
    missed = ((preds != 1) | deferred_mask) & fall_mask
    return float(missed.sum() / fall_mask.sum()) if fall_mask.sum() > 0 else 0.0


def aurc(scores: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> float:
    order  = np.argsort(scores)
    errors = (preds[order] != labels[order]).astype(float)
    risks  = np.cumsum(errors) / (np.arange(len(errors)) + 1)
    return float(risks.mean())


def bootstrap_ci(metric_fn, *arrays, n_resamples: int = 2000, ci: float = 0.95):
    result = bootstrap(arrays, metric_fn, n_resamples=n_resamples,
                       confidence_level=ci, method="percentile")
    return result.confidence_interval.low, result.confidence_interval.high


def evaluate_policy(policy: str, probs: np.ndarray, labels: np.ndarray,
                    luminance: np.ndarray, cfg: ControllerConfig) -> dict:
    ctrl = AgenticController(cfg)
    preds_out, deferred = [], []

    for i in range(len(labels)):
        pred = int(np.argmax(probs[i]))
        u_t  = -np.sum(probs[i] * np.log(probs[i] + 1e-9)) / math.log(2)
        q_t  = float(luminance[i])
        state = ControllerState(u_t=u_t, q_t=q_t,
                                tau_t=22.0, E_t=0.17, M_t=460.0, B_t=800.0)

        if policy == "P0":
            action = "Infer"
        elif policy == "P1":
            action = "Enhance" if q_t < cfg.tau_q else "Infer"
        elif policy == "P2":
            action = "Infer" if u_t <= cfg.tau_enhance else "Defer"
        elif policy == "P3":
            action = "Infer"
        elif policy == "P4":
            action = "FeatureSend"
        elif policy == "P5":
            action = np.random.choice(
                ["Infer", "Enhance", "FeatureSend", "Defer", "Adapt"],
                p=[0.7358, 0.1038, 0.0943, 0.0377, 0.0284])
        elif policy == "P6":
            action = ctrl.step(state)
        else:
            action = "Infer"

        deferred.append(action == "Defer")
        preds_out.append(pred)

    preds_out = np.array(preds_out)
    deferred  = np.array(deferred)
    resolved  = ~deferred
    mf1 = macro_f1_score(preds_out[resolved], labels[resolved]) if resolved.sum() > 0 else 0.0
    fnr = fall_fnr(preds_out, labels, deferred)
    return {
        "macro_f1": round(mf1, 3),
        "fall_fnr_pct": round(fnr * 100, 2),
        "unresolved_pct": round(deferred.mean() * 100, 2),
    }


@torch.no_grad()
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AgenticViT(num_classes=2, L=16)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    test_files  = load_split_file(os.path.join(args.splits_dir, "test_files.txt"))
    test_labels = [1 if "fall_" in f and "nofall" not in f else 0 for f in test_files]
    test_paths  = [os.path.join(args.data_root, f) for f in test_files]
    test_wins   = build_windows(test_paths, test_labels)
    test_ds     = WindowDataset(test_wins, test_paths, transform=eval_transform(),
                                corruption=args.corruption, severity=args.severity)
    loader = DataLoader(test_ds, batch_size=16, num_workers=2)

    all_probs, all_labels = [], []
    for frames, labels in loader:
        _, probs, _ = model(frames.to(device))
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    probs  = np.vstack(all_probs)
    labels = np.array(all_labels)

    n_images  = len(test_paths)
    img_probs = np.zeros((n_images, 2))
    img_count = np.zeros(n_images)
    for w_idx, win in enumerate(test_wins):
        p = probs[w_idx] if w_idx < len(probs) else np.array([0.5, 0.5])
        for img_idx in win["image_indices"]:
            img_probs[img_idx] += p
            img_count[img_idx] += 1
    img_probs /= np.maximum(img_count[:, None], 1)
    img_preds  = img_probs.argmax(axis=1)
    img_labels = np.array(test_labels)

    acc = (img_preds == img_labels).mean()
    mf1 = macro_f1_score(img_preds, img_labels)
    ece = expected_calibration_error(img_probs, img_labels)
    fnr = fall_fnr(img_preds, img_labels)
    u_scores = -np.sum(img_probs * np.log(img_probs + 1e-9), axis=1) / math.log(2)
    aur = aurc(u_scores, img_labels, img_preds)

    print(f"\n{'=' * 55}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Corruption : {args.corruption or 'none'}  severity={args.severity}")
    print(f"{'─' * 55}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro-F1   : {mf1:.4f}")
    print(f"ECE        : {ece:.4f}")
    print(f"Fall FNR   : {fnr * 100:.2f}%")
    print(f"AURC       : {aur:.4f}")
    print(f"{'=' * 55}\n")

    if args.output_csv:
        rows = [[i, int(img_labels[i]), int(img_preds[i]),
                 round(float(img_probs[i, 0]), 4),
                 round(float(u_scores[i]), 4)]
                for i in range(n_images)]
        with open(args.output_csv, "w", newline="") as f:
            csv.writer(f).writerows(
                [["image_id", "true_label", "pred_label", "p_nofall", "uncertainty"]] + rows)
        print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/agentic_vit_seed42.pt")
    p.add_argument("--data_root",   default="data/fall_detection")
    p.add_argument("--splits_dir",  default="data/splits")
    p.add_argument("--corruption",  default=None,
                   choices=[None, "blur", "jpeg", "lowlight", "noise"])
    p.add_argument("--severity",    type=int, default=0)
    p.add_argument("--output_csv",  default="results/eval_output.csv")
    run_evaluation(p.parse_args())
