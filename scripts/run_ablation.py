import argparse, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from model   import AgenticViT
from dataset import WindowDataset, build_windows, load_split_file, eval_transform
from evaluate import macro_f1_score, expected_calibration_error, fall_fnr

SEEDS = [42, 43, 44, 45, 46]

VARIANTS = {
    "Full_Agentic_ViT":          {},
    "No_temporal_module":        {"no_temporal": True},
    "No_adapter_pathway":        {"no_adapters": True},
    "No_adaptive_token_budget":  {"fixed_k_max": True},
    "No_low_rank_attention":     {"full_rank": True},
}


def build_model(variant: dict, seed: int) -> AgenticViT:
    torch.manual_seed(seed)
    return AgenticViT(
        num_classes=2,
        L=16,
        k_min=64 if not variant.get("fixed_k_max") else 160,
        k_max=160,
        rank=192 if variant.get("full_rank") else 48,
    )


def run_seed(model, test_wins, test_paths, test_labels, device):
    ds  = WindowDataset(test_wins, test_paths, transform=eval_transform())
    ldr = DataLoader(ds, batch_size=16, num_workers=0)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for frames, labels in ldr:
            _, probs, _ = model(frames.to(device))
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    probs  = np.vstack(all_probs)
    labels = np.array(all_labels)
    preds  = probs.argmax(axis=1)
    acc    = (preds == labels).mean()
    mf1    = macro_f1_score(preds, labels)
    ece    = expected_calibration_error(probs, labels)
    fnr    = fall_fnr(preds, labels)
    return acc, mf1, ece, fnr


def main(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_files = load_split_file(os.path.join(args.splits_dir, "test_files.txt"))
    test_labels = [1 if "fall_" in f and "nofall" not in f else 0 for f in test_files]
    test_paths  = [os.path.join(args.data_root, f) for f in test_files]
    test_wins   = build_windows(test_paths, test_labels)

    rows = [["variant", "seed", "accuracy", "macro_f1", "ece", "fall_fnr_pct"]]
    for name, variant in VARIANTS.items():
        accs, f1s, eces, fnrs = [], [], [], []
        for seed in SEEDS:
            model = build_model(variant, seed).to(device)
            acc, mf1, ece, fnr = run_seed(model, test_wins, test_paths, test_labels, device)
            accs.append(acc)
            f1s.append(mf1)
            eces.append(ece)
            fnrs.append(fnr)
            rows.append([name, seed, round(acc, 4), round(mf1, 4),
                         round(ece, 4), round(fnr * 100, 2)])
        print(f"{name:40s}  F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_run.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/fall_detection")
    p.add_argument("--splits_dir", default="data/splits")
    main(p.parse_args())
