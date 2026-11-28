import argparse, math, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model   import AgenticViT
from dataset import (WindowDataset, build_windows, load_split_file,
                     train_transform, eval_transform)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calibrate_temperature(model: AgenticViT, val_loader: DataLoader,
                           device: torch.device) -> float:
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            logits, _, _ = model(frames, calibrated=False)
            logits_all.append(logits.cpu())
            labels_all.append(labels)
    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    T = torch.nn.Parameter(torch.tensor(1.5))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def _eval():
        optimizer.zero_grad()
        loss = ce(logits_all / T.clamp(0.5, 5.0), labels_all)
        loss.backward()
        return loss

    for _ in range(50):
        optimizer.step(_eval)

    T_cal = float(T.clamp(0.5, 5.0).item())
    model.T_cal.data.fill_(T_cal)
    return T_cal


def macro_f1(preds: list[int], labels: list[int], K: int = 2) -> float:
    f1s = []
    for k in range(K):
        tp = sum(p == k and l == k for p, l in zip(preds, labels))
        fp = sum(p == k and l != k for p, l in zip(preds, labels))
        fn = sum(p != k and l == k for p, l in zip(preds, labels))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return sum(f1s) / K


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(frames, calibrated=False)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    for frames, labels in loader:
        frames = frames.to(device)
        _, probs, _ = model(frames)
        preds.extend(probs.argmax(dim=-1).cpu().tolist())
        labs.extend(labels.tolist())
    acc = sum(p == l for p, l in zip(preds, labs)) / len(labs)
    mf1 = macro_f1(preds, labs)
    return acc, mf1


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {args.seed}")

    train_files = load_split_file(os.path.join(args.splits_dir, "train_files.txt"))
    val_files   = load_split_file(os.path.join(args.splits_dir, "val_files.txt"))

    train_labels = [1 if "fall_" in f and "nofall" not in f else 0 for f in train_files]
    val_labels   = [1 if "fall_" in f and "nofall" not in f else 0 for f in val_files]

    train_paths = [os.path.join(args.data_root, f) for f in train_files]
    val_paths   = [os.path.join(args.data_root, f) for f in val_files]

    train_wins = build_windows(train_paths, train_labels)
    val_wins   = build_windows(val_paths,   val_labels)

    train_ds = WindowDataset(train_wins, train_paths, transform=train_transform())
    val_ds   = WindowDataset(val_wins,   val_paths,   transform=eval_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = AgenticViT(num_classes=2, L=16).to(device)

    for name, p in model.named_parameters():
        if "backbone.patch_embed" in name or "backbone.pos_embed" in name:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, patience_counter = 0.0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | loss {loss:.4f} | val_acc {val_acc:.4f} | val_F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"best_seed{args.seed}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, f"best_seed{args.seed}.pt")))
    T_cal = calibrate_temperature(model, val_loader, device)
    print(f"T_cal = {T_cal:.3f}")
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"calibrated_seed{args.seed}.pt"))
    print(f"Saved. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/fall_detection")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience",   type=int, default=8)
    p.add_argument("--output_dir", default="checkpoints")
    main(p.parse_args())
