from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .core import EPS, Geometry, Trace


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.05, EPS)
    return median, scale


def first_pulse_time(trace: Trace) -> float:
    combined = np.maximum(trace.ch1 - np.median(trace.ch1), 0) + np.maximum(trace.ch2 - np.median(trace.ch2), 0)
    _, scale = robust_location_scale(combined)
    threshold = np.median(combined) + 8.0 * scale
    indices = np.flatnonzero(combined > threshold)
    if len(indices):
        return float(trace.time_s[int(indices[0])])
    return float(trace.time_s[int(np.argmax(combined))])


def _point_energy(trace: Trace, centers: np.ndarray) -> float:
    indices = np.searchsorted(trace.time_s, centers)
    indices = np.clip(indices, 0, len(trace.time_s) - 1)
    baseline1 = np.median(trace.ch1)
    baseline2 = np.median(trace.ch2)
    return float(np.sum(np.maximum(trace.ch1[indices] - baseline1, 0) + np.maximum(trace.ch2[indices] - baseline2, 0)))


def fit_geometry(trace: Trace, nominal: Geometry, lattice_cfg: Mapping[str, Any]) -> Geometry:
    """Refine t0, mean round-trip time, and loop delay from early pulses."""
    layers = min(int(lattice_cfg.get("fit_early_layers", 18)), nominal.layers)
    t0_seed = nominal.t0_s if lattice_cfg.get("t0_s") is not None else first_pulse_time(trace)
    rt_fraction = float(lattice_cfg.get("fit_roundtrip_fraction", 0.002))
    spacing_fraction = float(lattice_cfg.get("fit_spacing_fraction", 0.03))

    best = (float("-inf"), t0_seed, nominal.roundtrip_s, nominal.site_spacing_s)
    t0_offsets = np.linspace(-0.55 * nominal.site_spacing_s, 0.55 * nominal.site_spacing_s, 17)
    roundtrips = nominal.roundtrip_s * (1.0 + np.linspace(-rt_fraction, rt_fraction, 9))
    spacings = nominal.site_spacing_s * (1.0 + np.linspace(-spacing_fraction, spacing_fraction, 9))
    for roundtrip in roundtrips:
        for spacing in spacings:
            base_times: list[float] = []
            for m in range(layers + 1):
                for n in range(-m, m + 1, 2):
                    base_times.append(t0_seed + m * roundtrip + 0.5 * n * spacing)
            base = np.asarray(base_times)
            for offset in t0_offsets:
                centers = base + offset
                valid = (centers >= trace.time_s[0]) & (centers <= trace.time_s[-1])
                if not np.any(valid):
                    continue
                score = _point_energy(trace, centers[valid])
                if score > best[0]:
                    best = (score, t0_seed + float(offset), float(roundtrip), float(spacing))

    _, t0, roundtrip, spacing = best
    for factor in (0.25, 0.08):
        candidates: list[tuple[float, float, float, float]] = []
        for dt0 in np.linspace(-factor, factor, 5) * nominal.site_spacing_s:
            for drt in np.linspace(-factor, factor, 5) * nominal.roundtrip_s * rt_fraction:
                for dsp in np.linspace(-factor, factor, 5) * nominal.site_spacing_s * spacing_fraction:
                    trial = Geometry(t0 + dt0, roundtrip + drt, spacing + dsp, nominal.pulse_window_s, layers)
                    centers, _, _ = trial.centers()
                    valid = (centers >= trace.time_s[0]) & (centers <= trace.time_s[-1])
                    candidates.append((_point_energy(trace, centers[valid]), trial.t0_s, trial.roundtrip_s, trial.site_spacing_s))
        _, t0, roundtrip, spacing = max(candidates, key=lambda item: item[0])
    return Geometry(float(t0), float(roundtrip), float(spacing), nominal.pulse_window_s, nominal.layers)


def window_integrals(time_s: np.ndarray, signal: np.ndarray, centers: np.ndarray, width_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = width_s / 2.0
    left = np.searchsorted(time_s, centers - half, side="left")
    right = np.searchsorted(time_s, centers + half, side="right")
    left = np.clip(left, 0, len(time_s))
    right = np.clip(right, 0, len(time_s))
    cumulative = np.concatenate(([0.0], np.cumsum(signal)))
    integral = cumulative[right] - cumulative[left]
    counts = right - left
    return integral, counts, np.stack((left, right), axis=1)


def off_grid_mask(length: int, windows: np.ndarray) -> np.ndarray:
    delta = np.zeros(length + 1, dtype=np.int32)
    for left, right in windows:
        if right > left:
            delta[int(left)] += 1
            delta[int(right)] -= 1
    return np.cumsum(delta[:-1]) == 0


def robust_quadratic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return linear and quadratic coefficients with two residual reweightings."""
    if len(x) < 4:
        return 0.0, 0.0
    x0 = x - np.mean(x)
    design = np.column_stack((np.ones_like(x0), x0, x0 * x0))
    weights = np.ones_like(y)
    coefficients = np.zeros(3)
    for _ in range(3):
        weighted = design * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=None)
        residual = y - design @ coefficients
        _, scale = robust_location_scale(residual)
        u = residual / (4.685 * scale)
        weights = np.square(np.maximum(1.0 - u * u, 0.0))
        if np.count_nonzero(weights) < 4:
            weights = np.ones_like(y)
    return float(coefficients[1]), float(coefficients[2])


def clipping_fraction(values: np.ndarray, low: float | None, high: float | None) -> float:
    if low is not None or high is not None:
        mask = np.zeros(len(values), dtype=bool)
        if low is not None:
            mask |= values <= low
        if high is not None:
            mask |= values >= high
        return float(np.mean(mask))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    tolerance = max((maximum - minimum) * 1e-9, EPS)
    return float(np.mean((np.abs(values - minimum) <= tolerance) | (np.abs(values - maximum) <= tolerance)))


def sigmoid_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    ratio = max(value, 0.0) / target
    return float(1.0 - math.exp(-ratio))
