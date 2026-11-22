"""One Euro filter (Casiez et al. 2012), scaled by the real time per frame (dt).

All times are inputs in seconds. This module never reads a clock.

    tau   = 1 / (2 pi f_c)
    alpha = 1 / (1 + tau / dt)
    x_hat = alpha * x + (1 - alpha) * x_hat_prev

Speed uses the raw previous sample: dx = (x - x_prev) / dt, smoothed with
cutoff d_c. The cutoff is f_c = f_min + beta * ||dx_hat|| (one speed for
all axes: the norm of the vector).
"""
import math
from dataclasses import dataclass

import numpy as np


class FilterError(ValueError):
    """Refusal with a short machine code in `.code`."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class OneEuroConfig:
    min_cutoff_hz: float = 1.0
    beta: float = 0.0  # Hz per (px/s)
    d_cutoff_hz: float = 1.0
    max_gap_s: float = 0.5

    def __post_init__(self):
        for name in ("min_cutoff_hz", "beta", "d_cutoff_hz", "max_gap_s"):
            if not _finite(getattr(self, name)):
                raise FilterError("config", f"{name} must be a finite number")
        if self.min_cutoff_hz <= 0 or self.d_cutoff_hz <= 0:
            raise FilterError("config", "cutoffs must be > 0")
        if self.beta < 0:
            raise FilterError("config", "beta must be >= 0")
        if self.max_gap_s <= 0:
            raise FilterError("config", "max_gap_s must be > 0")


def alpha(cutoff_hz, dt_s):
    """Smoothing factor for a cutoff (Hz) and a real frame time (s)."""
    if not _finite(dt_s) or dt_s <= 0:
        raise FilterError("time", "dt must be finite and > 0")
    if not _finite(cutoff_hz) or cutoff_hz <= 0:
        raise FilterError("config", "cutoff must be finite and > 0")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuro:
    def __init__(self, config=OneEuroConfig(), dim=2):
        if not isinstance(config, OneEuroConfig):
            raise FilterError("config", "config must be a OneEuroConfig")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise FilterError("config", "dim must be an int >= 1")
        self.config = config
        self.dim = dim
        self.reset()

    def reset(self):
        self._t_prev = None
        self._x_prev = None
        self._x_hat = None
        self._dx_hat = None

    def step(self, t_s, x):
        try:
            t = float(t_s)
        except (TypeError, ValueError):
            raise FilterError("time", "t must be a number") from None
        if isinstance(t_s, bool) or not math.isfinite(t):
            raise FilterError("time", "t must be a finite number")
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise FilterError("shape", f"x must have shape ({self.dim},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise FilterError("nan", "x has NaN or inf")

        if self._t_prev is not None:
            dt = t - self._t_prev
            if dt <= 0:
                raise FilterError("time", f"time must rise (dt = {dt})")
            if dt > self.config.max_gap_s:
                self.reset()

        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x.copy()
            self._x_hat = x.copy()
            self._dx_hat = np.zeros(self.dim)
            return self._x_hat.copy()

        cfg = self.config
        dx = (x - self._x_prev) / dt  # raw previous sample, not the filtered one
        a_d = alpha(cfg.d_cutoff_hz, dt)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        speed = float(np.linalg.norm(self._dx_hat))
        a = alpha(cfg.min_cutoff_hz + cfg.beta * speed, dt)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        self._t_prev = t
        self._x_prev = x.copy()
        return self._x_hat.copy()


def filter_series(t_s, xs, config=OneEuroConfig()):
    """Run OneEuro over the rows of xs (n, dim); returns (n, dim) float64."""
    xs = np.asarray(xs, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    if xs.ndim != 2 or t_s.ndim != 1 or t_s.shape[0] != xs.shape[0]:
        raise FilterError("shape", "need t (n,) and xs (n, dim)")
    f = OneEuro(config, dim=xs.shape[1])
    out = np.empty_like(xs)
    for i in range(xs.shape[0]):
        out[i] = f.step(float(t_s[i]), xs[i])
    return out


def min_cutoff_for_noise(sigma_in, sigma_out, fps):
    """f_min (Hz) that turns white noise SD sigma_in into SD sigma_out at fps (beta 0)."""
    for v in (sigma_in, sigma_out, fps):
        if not _finite(v) or v <= 0:
            raise FilterError("value", "sigma_in, sigma_out, fps must be finite and > 0")
    r = sigma_out / sigma_in
    if not 0 < r < 1:
        raise FilterError("value", "need 0 < sigma_out / sigma_in < 1")
    a = 2.0 * r * r / (1.0 + r * r)
    dt = 1.0 / fps
    tau = dt * (1.0 - a) / a
    return 1.0 / (2.0 * math.pi * tau)


def ramp_lag_ms(config, speed_px_s):
    """Steady lag (ms) on a ramp of speed v: 1000 / (2 pi (f_min + beta v))."""
    if not _finite(speed_px_s) or speed_px_s < 0:
        raise FilterError("value", "speed must be finite and >= 0")
    return 1000.0 / (2.0 * math.pi * (config.min_cutoff_hz + config.beta * speed_px_s))
