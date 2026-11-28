from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Literal

Action = Literal["Infer", "Enhance", "FeatureSend", "Defer", "Adapt"]


@dataclass
class ControllerState:
    u_t: float
    q_t: float
    tau_t: float
    E_t: float
    M_t: float
    B_t: float
    uncertainty_history: list[float] = field(default_factory=list)


@dataclass
class ControllerConfig:
    tau_infer: float   = 0.25
    tau_enhance: float = 0.50
    tau_adapt: float   = 0.75
    tau_q: float       = 0.25
    tau_max: float     = 60.0
    E_max: float       = 0.25
    M_max: float       = 512.0
    B_max: float       = 1024.0
    m_d: int           = 6
    n_d: int           = 8


class AgenticController:
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()
        self._u_history: deque[float] = deque(maxlen=self.cfg.n_d)

    def _resource_feasible(self, s: ControllerState) -> bool:
        c = self.cfg
        return s.tau_t <= c.tau_max and s.E_t <= c.E_max and s.M_t <= c.M_max

    def _drift_indicator(self) -> bool:
        if len(self._u_history) < self.cfg.n_d:
            return False
        count = sum(u >= self.cfg.tau_adapt for u in self._u_history)
        return count >= self.cfg.m_d

    def step(self, s: ControllerState) -> Action:
        self._u_history.append(s.u_t)
        cfg = self.cfg
        R = self._resource_feasible(s)
        d_t = self._drift_indicator()

        if not R:
            return "Defer"

        if s.u_t >= cfg.tau_adapt and d_t:
            return "Adapt"

        if cfg.tau_enhance < s.u_t < cfg.tau_adapt:
            if s.B_t <= cfg.B_max:
                return "FeatureSend"
            return "Defer"

        if cfg.tau_infer < s.u_t <= cfg.tau_enhance and s.q_t < cfg.tau_q:
            return "Enhance"

        if s.u_t <= cfg.tau_infer:
            return "Infer"

        return "Defer"

    def compute_luminance(self, frames) -> float:
        import torch
        Y = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        return float(Y.mean().item())

    def reset(self):
        self._u_history.clear()


class ExtendedQualityController(AgenticController):
    def __init__(self, cfg: ControllerConfig | None = None,
                 tau_sharp: float = 0.35, tau_noise: float = 0.12,
                 tau_blk: float = 1.30):
        super().__init__(cfg)
        self.tau_sharp = tau_sharp
        self.tau_noise = tau_noise
        self.tau_blk = tau_blk

    def _quality_flags(self, frames) -> bool:
        import numpy as np
        from scipy.ndimage import laplace
        Y = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] +
             0.114 * frames[:, 2]).mean(0).cpu().numpy()
        lap_var = float(laplace(Y).var())
        hp = Y - Y.mean()
        noise_est = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
        low_lum = float(Y.mean()) < (self.cfg or ControllerConfig()).tau_q
        return low_lum or lap_var < self.tau_sharp or noise_est > self.tau_noise

    def step(self, s: ControllerState, frames=None) -> Action:
        self._u_history.append(s.u_t)
        cfg = self.cfg
        R = self._resource_feasible(s)
        d_t = self._drift_indicator()

        if not R:
            return "Defer"
        if s.u_t >= cfg.tau_adapt and d_t:
            return "Adapt"
        if cfg.tau_enhance < s.u_t < cfg.tau_adapt:
            return "FeatureSend" if s.B_t <= cfg.B_max else "Defer"
        if cfg.tau_infer < s.u_t <= cfg.tau_enhance:
            poor = self._quality_flags(frames) if frames is not None else s.q_t < cfg.tau_q
            if poor:
                return "Enhance"
        if s.u_t <= cfg.tau_infer:
            return "Infer"
        return "Defer"
