from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .core import EPS, Geometry, Metrics, Trace
from .geometry import (
    clipping_fraction,
    off_grid_mask,
    robust_location_scale,
    robust_quadratic_fit,
    sigmoid_ratio,
    window_integrals,
)


def analyze_trace(trace: Trace, geometry: Geometry, config: Mapping[str, Any]) -> Metrics:
    detectors = config.get("detectors", {}) if isinstance(config.get("detectors", {}), Mapping) else {}
    targets = config.get("targets", {}) if isinstance(config.get("targets", {}), Mapping) else {}
    gain1 = float(detectors.get("gain_ch1", 1.0))
    gain2 = float(detectors.get("gain_ch2", 1.0))
    gain3 = float(detectors.get("gain_ch3", 1.0))
    if gain1 <= 0 or gain2 <= 0 or gain3 <= 0:
        raise ValueError("detector gains must be positive")

    centers, layers, _sites = geometry.centers()
    valid = (centers - geometry.pulse_window_s / 2 >= trace.time_s[0]) & (centers + geometry.pulse_window_s / 2 <= trace.time_s[-1])
    centers = centers[valid]
    layers = layers[valid]
    if not len(centers):
        raise ValueError("trace does not cover the configured lattice")

    _, _, windows = window_integrals(trace.time_s, np.zeros_like(trace.ch1), centers, geometry.pulse_window_s)
    background_mask = off_grid_mask(len(trace.time_s), windows)
    if np.count_nonzero(background_mask) < 16:
        background_mask = np.ones(len(trace.time_s), dtype=bool)
    base1, sigma1 = robust_location_scale(trace.ch1[background_mask])
    base2, sigma2 = robust_location_scale(trace.ch2[background_mask])
    corrected1 = (trace.ch1 - base1) * gain1
    corrected2 = (trace.ch2 - base2) * gain2
    energy1, counts, windows = window_integrals(trace.time_s, corrected1, centers, geometry.pulse_window_s)
    energy2, _, _ = window_integrals(trace.time_s, corrected2, centers, geometry.pulse_window_s)

    positive1, _, _ = window_integrals(trace.time_s, np.maximum(corrected1, 0.0), centers, geometry.pulse_window_s)
    positive2, _, _ = window_integrals(trace.time_s, np.maximum(corrected2, 0.0), centers, geometry.pulse_window_s)
    max_layer = geometry.layers
    layer_long = np.bincount(layers, weights=energy1, minlength=max_layer + 1)[:max_layer + 1]
    layer_short = np.bincount(layers, weights=energy2, minlength=max_layer + 1)[:max_layer + 1]
    layer_counts = np.bincount(layers, weights=counts, minlength=max_layer + 1)[:max_layer + 1]
    noise_per_sample = math.hypot(sigma1 * gain1, sigma2 * gain2)
    layer_noise = noise_per_sample * np.sqrt(np.maximum(layer_counts, 1))
    layer_signal = np.maximum(layer_long + layer_short, 0.0)
    layer_snr = layer_signal / np.maximum(layer_noise, EPS)

    min_layer_snr = float(targets.get("min_layer_snr", 4.0))
    usable = layer_snr >= min_layer_snr
    usable_layers = 0
    misses = 0
    for index, present in enumerate(usable):
        if present:
            misses = 0
            usable_layers = index
        else:
            misses += 1
            if misses >= 2:
                break
    target_layers = int(targets.get("layers", geometry.layers))

    fit_start = min(5, max_layer)
    fit_end = min(max_layer, max(12, usable_layers))
    fit_slice = np.arange(fit_start, fit_end + 1)
    total_layer = np.maximum(layer_long + layer_short, EPS)
    positive_fit = layer_snr[fit_slice] >= max(min_layer_snr, 2.0)
    fit_reliable = np.count_nonzero(positive_fit) >= 4
    if fit_reliable:
        slope, curvature = robust_quadratic_fit(fit_slice[positive_fit].astype(float), np.log(total_layer[fit_slice][positive_fit]))
    else:
        slope, curvature = 0.0, 0.0

    early_end = min(8, max_layer)
    early_long = float(np.sum(np.maximum(layer_long[1:early_end + 1], 0.0)))
    early_short = float(np.sum(np.maximum(layer_short[1:early_end + 1], 0.0)))
    early_split = early_long / max(early_long + early_short, EPS)
    target_split_value = targets.get("target_split_long")
    balance_known = target_split_value is not None
    target_split = 0.5 if target_split_value is None else float(target_split_value)
    balance_error = abs(early_split - target_split) if balance_known else 0.0

    in_window_positive = float(np.sum(positive1 + positive2))
    window_samples = max(int(np.sum(counts)) * 2, 1)
    off_positive = np.maximum(corrected1[background_mask], 0.0)
    off_positive = np.concatenate((off_positive, np.maximum(corrected2[background_mask], 0.0)))
    off_mean = float(np.mean(off_positive)) if len(off_positive) else 0.0
    in_mean = in_window_positive / window_samples
    concentration = float(np.clip((in_mean - off_mean) / max(in_mean, EPS), 0.0, 1.0))

    leakage_fraction = 0.0
    if trace.ch3 is not None:
        base3, _sigma3 = robust_location_scale(trace.ch3[background_mask])
        corrected3 = (trace.ch3 - base3) * gain3
        positive3, _, _ = window_integrals(trace.time_s, np.maximum(corrected3, 0.0), centers, geometry.pulse_window_s)
        leakage_fraction = float(np.sum(positive3) / max(np.sum(positive1 + positive2) + np.sum(positive3), EPS))

    clip_low = detectors.get("clip_low")
    clip_high = detectors.get("clip_high")
    clipping = max(
        clipping_fraction(trace.ch1, None if clip_low is None else float(clip_low), None if clip_high is None else float(clip_high)),
        clipping_fraction(trace.ch2, None if clip_low is None else float(clip_low), None if clip_high is None else float(clip_high)),
    )

    late_start = max(1, int(0.8 * min(target_layers, max_layer)))
    late_values = layer_snr[late_start:min(target_layers, max_layer) + 1]
    late_snr = float(np.quantile(late_values, 0.25)) if len(late_values) else 0.0
    median_snr = float(np.median(layer_snr[1:min(target_layers, max_layer) + 1])) if max_layer >= 1 else 0.0

    target_gain = float(targets.get("target_log_gain_per_layer", 0.0))
    gain_tolerance = max(float(targets.get("log_gain_tolerance", 0.012)), EPS)
    split_tolerance = max(float(targets.get("split_tolerance", 0.12)), EPS)
    max_leakage = max(float(targets.get("max_leakage_fraction", 0.08)), EPS)
    max_clipping = max(float(targets.get("max_clipping_fraction", 0.001)), EPS)
    target_late_snr = max(float(targets.get("late_snr", 6.0)), EPS)
    target_concentration = max(float(targets.get("lattice_concentration", 0.55)), EPS)

    components = {
        "depth": min(usable_layers / max(target_layers, 1), 1.0),
        "late_snr": sigmoid_ratio(late_snr, target_late_snr),
        "lattice": min(concentration / target_concentration, 1.0),
        "gain": math.exp(-0.5 * ((slope - target_gain) / gain_tolerance) ** 2) if fit_reliable else 0.03,
        "balance": math.exp(-0.5 * (balance_error / split_tolerance) ** 2) if balance_known else 1.0,
        "leakage": math.exp(-leakage_fraction / max_leakage),
        "linear": math.exp(-clipping / max_clipping),
        "curvature": math.exp(-abs(curvature) / 0.0015) if fit_reliable else 0.1,
    }
    weights = {
        "depth": 3.0,
        "late_snr": 2.0,
        "lattice": 2.0,
        "gain": 1.6,
        "balance": 1.0 if balance_known else 0.0,
        "leakage": 1.0 if trace.ch3 is not None else 0.0,
        "linear": 1.5,
        "curvature": 0.8,
    }
    weighted_log = sum(weights[name] * math.log(max(value, 1e-8)) for name, value in components.items())
    total_weight = sum(weights.values())
    score = 100.0 * math.exp(weighted_log / max(total_weight, EPS))

    ready = (
        usable_layers >= target_layers
        and late_snr >= target_late_snr
        and concentration >= target_concentration
        and abs(slope - target_gain) <= gain_tolerance
        and (not balance_known or balance_error <= split_tolerance)
        and leakage_fraction <= max_leakage
        and clipping <= max_clipping
    )
    return Metrics(
        score=float(score), ready=bool(ready), usable_layers=int(usable_layers), target_layers=target_layers,
        lattice_concentration=float(concentration), late_snr=late_snr, median_snr=median_snr,
        log_gain_per_layer=float(slope), gain_curvature=float(curvature), early_split_long=float(early_split),
        balance_known=bool(balance_known), balance_error=float(balance_error), leakage_fraction=float(leakage_fraction),
        clipping_fraction=float(clipping), pulse_energy_long=float(np.sum(layer_long)),
        pulse_energy_short=float(np.sum(layer_short)), background_sigma_long=float(sigma1 * gain1),
        background_sigma_short=float(sigma2 * gain2), t0_s=geometry.t0_s, roundtrip_s=geometry.roundtrip_s,
        site_spacing_s=geometry.site_spacing_s, components={name: float(value) for name, value in components.items()},
        layer_energy_long=layer_long.tolist(), layer_energy_short=layer_short.tolist(), layer_snr=layer_snr.tolist(),
    )


def response_value(metrics: Metrics, name: str) -> float:
    if name == "early_split_long":
        return metrics.early_split_long
    if name == "early_long_energy":
        cutoff = min(8, len(metrics.layer_energy_long) - 1)
        return float(sum(max(item, 0.0) for item in metrics.layer_energy_long[1:cutoff + 1]))
    if name == "early_short_energy":
        cutoff = min(8, len(metrics.layer_energy_short) - 1)
        return float(sum(max(item, 0.0) for item in metrics.layer_energy_short[1:cutoff + 1]))
    if name == "log_gain_per_layer":
        return metrics.log_gain_per_layer
    if name == "score":
        return metrics.score
    if name == "late_snr":
        return metrics.late_snr
    if name == "lattice_concentration":
        return metrics.lattice_concentration
    raise ValueError(f"unknown response {name}")
