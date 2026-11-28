from __future__ import annotations
from collections import deque
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeedbackBuffer:
    def __init__(self, maxlen: int = 20, tau_adapt: float = 0.75,
                 strict: bool = False):
        self._buf  = deque(maxlen=maxlen)
        self.tau   = tau_adapt
        self.strict = strict

    def add(self, embedding: torch.Tensor, label: int, uncertainty: float):
        if self.strict and uncertainty < self.tau:
            return
        self._buf.append((embedding.detach().cpu(), torch.tensor(label)))

    def clear(self):
        self._buf.clear()

    def as_dataset(self) -> TensorDataset | None:
        if not self._buf:
            return None
        embs, lbls = zip(*self._buf)
        return TensorDataset(torch.stack(embs), torch.stack(lbls))

    def __len__(self):
        return len(self._buf)


class AdapterUpdater:
    def __init__(self, model: nn.Module,
                 n_steps: int = 10, lr: float = 1e-4,
                 eps_rollback: float = 0.01,
                 replay_buffer: list | None = None):
        self.model   = model
        self.n_steps = n_steps
        self.lr      = lr
        self.eps_rb  = eps_rollback
        self.replay  = replay_buffer or []

    def _adapter_params(self):
        return [p for name, p in self.model.named_parameters()
                if "adapter" in name and p.requires_grad]

    def _macro_f1(self, loader: DataLoader, device: torch.device) -> float:
        self.model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for frames, labels in loader:
                _, probs, _ = self.model(frames.to(device))
                preds.extend(probs.argmax(-1).cpu().tolist())
                labs.extend(labels.tolist())
        K = 2
        f1s = []
        for k in range(K):
            tp = sum(p == k and l == k for p, l in zip(preds, labs))
            fp = sum(p == k and l != k for p, l in zip(preds, labs))
            fn = sum(p != k and l == k for p, l in zip(preds, labs))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1s.append(2 * prec * rec / (prec + rec + 1e-9))
        return sum(f1s) / K

    def update(self, feedback_ds: TensorDataset,
               check_ds: TensorDataset | None = None,
               device: torch.device | None = None,
               guarded: bool = False) -> bool:
        if feedback_ds is None:
            return False
        device = device or torch.device("cpu")
        params = self._adapter_params()
        if not params:
            return False

        snapshot = {id(p): p.data.clone() for p in params}

        f1_before = None
        if guarded and check_ds:
            ldr = DataLoader(check_ds, batch_size=16)
            f1_before = self._macro_f1(ldr, device)

        all_items = list(feedback_ds)
        for item in self.replay[-50:]:
            all_items.append(item)
        combined = TensorDataset(*[torch.stack([x[i] for x in all_items])
                                   for i in range(2)])

        loader = DataLoader(combined, batch_size=min(16, len(combined)), shuffle=True)
        opt    = torch.optim.Adam(params, lr=self.lr)
        ce     = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(self.n_steps):
            for embs, lbls in loader:
                embs, lbls = embs.to(device), lbls.to(device)
                opt.zero_grad()
                logits = self.model.head(embs)
                ce(logits, lbls).backward()
                opt.step()

        if guarded and check_ds and f1_before is not None:
            ldr      = DataLoader(check_ds, batch_size=16)
            f1_after = self._macro_f1(ldr, device)
            if f1_after < f1_before - self.eps_rb:
                for p in params:
                    p.data.copy_(snapshot[id(p)])
                return False

        return True
