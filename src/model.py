import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenScorer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.w(tokens).squeeze(-1)


class LowRankAttention(nn.Module):
    def __init__(self, d: int = 192, r: int = 48):
        super().__init__()
        self.r = r
        self.Wq = nn.Linear(d, r, bias=False)
        self.Wk = nn.Linear(d, r, bias=False)
        self.Wv = nn.Linear(d, r, bias=False)
        self.Wo = nn.Linear(r, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.r), dim=-1)
        return self.Wo(attn @ V)


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(tokens @ self.w, dim=-1)
        return (alpha.unsqueeze(-1) * tokens).sum(dim=1)


class LoRAAdapter(nn.Module):
    def __init__(self, d: int = 192, r: int = 4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d))

    def forward(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x @ W.T + x @ self.A @ self.B


class EdgeViTBackbone(nn.Module):
    D = 192
    N = 196

    def __init__(self, k_min: int = 64, k_max: int = 160, rank: int = 48,
                 pretrained: bool = False):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.patch_embed = nn.Conv2d(3, self.D, kernel_size=16, stride=16, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.D))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N + 1, self.D))
        self.scorer = TokenScorer(self.D)
        self.attn = LowRankAttention(self.D, rank)
        self.norm = nn.LayerNorm(self.D)
        self.adapter = LoRAAdapter(self.D)
        self.pool = AttentionPool(self.D)

    def compute_budget(self, tau_t: float, E_t: float, M_t: float,
                       tau_max: float = 60.0, E_max: float = 0.25,
                       M_max: float = 512.0, tau_ref: float = 19.6,
                       E_ref: float = 0.14, M_ref: float = 452.0) -> int:
        b = max(0.0, min(1.0, min(
            (tau_max - tau_t) / (tau_max - tau_ref),
            (E_max - E_t) / (E_max - E_ref),
            (M_max - M_t) / (M_max - M_ref),
        )))
        k = round(self.k_min + (self.k_max - self.k_min) * b)
        return int(max(self.k_min, min(self.k_max, k)))

    def forward(self, x: torch.Tensor, k_t: int | None = None) -> torch.Tensor:
        if k_t is None:
            k_t = self.k_max
        B = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = tokens[:, 1:]
        scores = self.scorer(tokens)
        idx = scores.topk(k_t, dim=-1).indices.sort(dim=-1).values
        tokens = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.D))
        tokens = tokens + self.attn(self.norm(tokens))
        return self.pool(tokens)


class TemporalFusion(nn.Module):
    def __init__(self, d: int = 192, L: int = 16, n_heads: int = 4):
        super().__init__()
        self.L = L
        enc = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", enc.unsqueeze(0))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z + self.pe
        H, _ = self.attn(Z, Z, Z)
        H = self.norm(Z + H)
        H = self.norm2(H + self.ff(H))
        return H.mean(dim=1)


class AgenticViT(nn.Module):
    def __init__(self, num_classes: int = 2, L: int = 16,
                 k_min: int = 64, k_max: int = 160, rank: int = 48):
        super().__init__()
        self.L = L
        self.backbone = EdgeViTBackbone(k_min=k_min, k_max=k_max, rank=rank)
        self.temporal = TemporalFusion(d=192, L=L)
        self.head = nn.Linear(192, num_classes)
        self.T_cal = nn.Parameter(torch.tensor(1.35), requires_grad=False)

    def encode_window(self, frames: torch.Tensor, k_t: int | None = None) -> torch.Tensor:
        B, L = frames.shape[:2]
        flat = frames.view(B * L, *frames.shape[2:])
        z = self.backbone(flat, k_t)
        return z.view(B, L, 192)

    def forward(self, frames: torch.Tensor, k_t: int | None = None,
                calibrated: bool = True):
        Z = self.encode_window(frames, k_t)
        h = self.temporal(Z)
        logits = self.head(h)
        T = self.T_cal if calibrated else torch.tensor(1.0)
        probs = F.softmax(logits / T, dim=-1)
        return logits, probs, h

    def uncertainty(self, probs: torch.Tensor) -> torch.Tensor:
        log_K = math.log(probs.shape[-1])
        return -(probs * (probs + 1e-9).log()).sum(dim=-1) / log_K
