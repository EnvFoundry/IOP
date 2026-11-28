from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends import Backend
from .core import EPS, Control, Geometry, Metrics, Trace, emit, read_json, utc_now, write_json
from .traces import analyze_trace, fit_geometry, response_value


class Calibrator:
    def __init__(self, config: Mapping[str, Any], backend: Backend, geometry: Geometry,
                 controls: Sequence[Control], output_dir: Path):
        self.config = config
        self.backend = backend
        self.nominal_geometry = geometry
        self.geometry: Geometry | None = None
        self.controls = [item for item in controls if item.enabled]
        self.control_by_name = {item.name: item for item in self.controls}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        acquisition = config.get("acquisition", {}) if isinstance(config.get("acquisition", {}), Mapping) else {}
        self.repeats = int(acquisition.get("repeats", 1))
        self.final_repeats = int(acquisition.get("final_repeats", 3))
        search = config.get("search", {}) if isinstance(config.get("search", {}), Mapping) else {}
        self.minimum_improvement = float(search.get("minimum_improvement", 0.15))
        self.rng = random.Random(int(search.get("seed", 9)))
        self.history: list[dict[str, Any]] = []
        self.best_metrics: Metrics | None = None
        self.best_values: dict[str, float] | None = None
        self.curves: dict[str, list[dict[str, float]]] = {}

    def _resolve_geometry(self, trace: Trace) -> Geometry:
        if self.geometry is not None:
            return self.geometry
        lattice = self.config.get("lattice", {}) if isinstance(self.config.get("lattice", {}), Mapping) else {}
        self.geometry = fit_geometry(trace, self.nominal_geometry, lattice)
        emit("geometry", **dataclasses.asdict(self.geometry))
        return self.geometry

    def measure(self, label: str, repeats: int | None = None) -> Metrics:
        repeats = self.repeats if repeats is None else repeats
        metrics: list[Metrics] = []
        sources: list[str] = []
        for repeat in range(max(1, repeats)):
            trace = self.backend.capture(f"{label}_{repeat + 1}")
            geometry = self._resolve_geometry(trace)
            metrics.append(analyze_trace(trace, geometry, self.config))
            sources.append(trace.source)
        chosen = sorted(metrics, key=lambda item: item.score)[len(metrics) // 2]
        record = {
            "time": utc_now(),
            "label": label,
            "controls": self.backend.values(),
            "metrics": chosen.to_dict(include_layers=False),
            "trace_sources": sources,
        }
        self.history.append(record)
        if self.best_metrics is None or chosen.score > self.best_metrics.score:
            self.best_metrics = chosen
            self.best_values = self.backend.values()
        with (self.output_dir / "measurements.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        emit("measurement", label=label, score=round(chosen.score, 4), layers=chosen.usable_layers,
             late_snr=round(chosen.late_snr, 3), gain=round(chosen.log_gain_per_layer, 6),
             split=round(chosen.early_split_long, 4), leakage=round(chosen.leakage_fraction, 5),
             clipping=round(chosen.clipping_fraction, 6), ready=chosen.ready)
        return chosen

    def set_values(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            if name in self.control_by_name:
                self.backend.set(self.control_by_name[name], float(value))

    def response_target(self, control: Control) -> float | None:
        if control.target is not None:
            return control.target
        if control.response == "early_split_long":
            targets = self.config.get("targets", {}) if isinstance(self.config.get("targets", {}), Mapping) else {}
            value = targets.get("target_split_long")
            return None if value is None else float(value)
        return None

    def scan_response(self, control: Control) -> Metrics:
        target_value = self.response_target(control)
        if control.response is None:
            return self.measure(f"{control.name}_unchanged")
        current = self.backend.values()[control.name]
        values = np.linspace(control.lower, control.upper, max(control.scan_points, 3))
        rows: list[tuple[float, float, Metrics]] = []
        emit("scan_start", control=control.name, response=control.response, target=target_value,
             points=len(values))
        for value in values:
            self.backend.set(control, float(value))
            metrics = self.measure(f"scan_{control.name}_{value:.6g}")
            rows.append((float(value), response_value(metrics, control.response), metrics))

        self.curves[control.name] = [
            {"value": value, "response": response, "score": metrics.score}
            for value, response, metrics in rows
        ]
        write_json(self.output_dir / "response_curves.json", self.curves)
        responses = np.asarray([item[1] for item in rows], dtype=float)
        if target_value is None:
            chosen_value = max(rows, key=lambda item: item[2].score)[0]
        else:
            if control.response in {"early_long_energy", "early_short_energy"}:
                low, high = float(np.min(responses)), float(np.max(responses))
                normalized = (responses - low) / max(high - low, EPS)
                target = target_value
            else:
                normalized = responses
                target = target_value

            candidates: list[tuple[float, float]] = []
            for index in range(len(values) - 1):
                y0, y1 = float(normalized[index]), float(normalized[index + 1])
                if (y0 - target) * (y1 - target) <= 0 and abs(y1 - y0) > EPS:
                    fraction = (target - y0) / (y1 - y0)
                    x = float(values[index] + fraction * (values[index + 1] - values[index]))
                    candidates.append((abs(x - current), x))
            if candidates:
                chosen_value = min(candidates)[1]
            else:
                chosen_value = float(values[int(np.argmin(np.abs(normalized - target)))])
        self.backend.set(control, chosen_value)
        final = self.measure(f"scan_{control.name}_chosen")
        emit("scan_done", control=control.name, value=chosen_value, score=final.score)
        return final

    def one_dimensional_search(self, control: Control, objective: Callable[[Metrics], float],
                               local: bool = False) -> Metrics:
        current = self.backend.values()[control.name]
        step = min(control.step, (control.upper - control.lower) * (0.15 if local else 1.0))
        best_metrics = self.measure(f"search_{control.name}_start")
        best_value = current
        best_objective = objective(best_metrics)
        evaluations = 1
        while step >= control.resolution and evaluations < 40:
            improved = False
            for direction in (+1.0, -1.0):
                trial = float(np.clip(best_value + direction * step, control.lower, control.upper))
                if abs(trial - best_value) < 0.5 * control.resolution:
                    continue
                self.backend.set(control, trial)
                metrics = self.measure(f"search_{control.name}_{trial:.6g}")
                evaluations += 1
                value = objective(metrics)
                if value > best_objective + self.minimum_improvement:
                    best_value, best_metrics, best_objective = trial, metrics, value
                    improved = True
                    break
                self.backend.set(control, best_value)
            if not improved:
                step *= 0.5
        self.backend.set(control, best_value)
        emit("search_done", control=control.name, value=best_value, objective=best_objective,
             evaluations=evaluations)
        return best_metrics

    def spsa(self, controls: Sequence[Control], iterations: int, label: str, local: bool = False) -> Metrics:
        controls = [item for item in controls if item.upper > item.lower]
        if not controls:
            return self.measure(label + "_empty")
        search = self.config.get("search", {}) if isinstance(self.config.get("search", {}), Mapping) else {}
        a0 = float(search.get("spsa_a", 0.12)) * (0.4 if local else 1.0)
        c0 = float(search.get("spsa_c", 0.08)) * (0.5 if local else 1.0)
        spans = np.asarray([item.upper - item.lower for item in controls], dtype=float)
        lower = np.asarray([item.lower for item in controls], dtype=float)
        x = np.asarray([(self.backend.values()[item.name] - item.lower) / (item.upper - item.lower) for item in controls])
        initial = self.measure(label + "_start")
        best_metrics = initial
        best_x = x.copy()
        emit("spsa_start", label=label, controls=[item.name for item in controls], score=initial.score)
        for iteration in range(iterations):
            ak = a0 / ((iteration + 1 + 4.0) ** 0.602)
            ck = c0 / ((iteration + 1) ** 0.101)
            delta = np.asarray([1.0 if self.rng.random() > 0.5 else -1.0 for _ in controls])
            plus = np.clip(x + ck * delta, 0.0, 1.0)
            minus = np.clip(x - ck * delta, 0.0, 1.0)
            self._set_normalized(controls, lower, spans, plus)
            plus_metrics = self.measure(f"{label}_{iteration + 1}_plus")
            self._set_normalized(controls, lower, spans, minus)
            minus_metrics = self.measure(f"{label}_{iteration + 1}_minus")
            difference = plus - minus
            gradient = np.zeros_like(difference)
            movable = np.abs(difference) > 1e-6
            gradient[movable] = (plus_metrics.score - minus_metrics.score) / difference[movable]
            norm = float(np.linalg.norm(gradient))
            if norm > 0:
                gradient /= norm
            x = np.clip(x + ak * gradient, 0.0, 1.0)
            self._set_normalized(controls, lower, spans, x)
            current = self.measure(f"{label}_{iteration + 1}_center")
            for candidate_x, candidate in ((plus, plus_metrics), (minus, minus_metrics), (x.copy(), current)):
                if candidate.score > best_metrics.score:
                    best_x = np.asarray(candidate_x).copy()
                    best_metrics = candidate
        self._set_normalized(controls, lower, spans, best_x)
        emit("spsa_done", label=label, score=best_metrics.score,
             controls={item.name: self.backend.values()[item.name] for item in controls})
        return best_metrics

    def _set_normalized(self, controls: Sequence[Control], lower: np.ndarray, spans: np.ndarray, values: np.ndarray) -> None:
        physical = lower + spans * values
        for control, value in zip(controls, physical):
            self.backend.set(control, float(value))

    def gain_objective(self, metrics: Metrics) -> float:
        targets = self.config.get("targets", {}) if isinstance(self.config.get("targets", {}), Mapping) else {}
        target = float(targets.get("target_log_gain_per_layer", 0.0))
        tolerance = max(float(targets.get("log_gain_tolerance", 0.012)), EPS)
        return metrics.score - 25.0 * abs(metrics.log_gain_per_layer - target) / tolerance

    def calibrate(self, local: bool = False) -> Metrics:
        baseline = self.measure("baseline", repeats=max(1, self.repeats))
        baseline_values = self.backend.values()
        emit("calibration_start", score=baseline.score, controls=baseline_values)

        for control in [item for item in self.controls if item.stage == "timing"]:
            self.one_dimensional_search(control, lambda item: 100.0 * item.lattice_concentration + item.score, local=local)

        polarization = [item for item in self.controls if item.stage == "polarization"]
        if polarization:
            iterations = int(self.config.get("search", {}).get("spsa_iterations", 12))
            self.spsa(polarization, max(3, iterations // 2) if local else iterations, "polarization", local=local)

        for control in [item for item in self.controls if item.stage == "mzm"]:
            if local:
                self.one_dimensional_search(control, lambda item: item.score, local=True)
            else:
                self.scan_response(control)

        for control in [item for item in self.controls if item.stage == "vbs"]:
            if self.response_target(control) is None:
                continue
            if local:
                self.one_dimensional_search(control, lambda item: item.score - 80.0 * item.balance_error, local=True)
            else:
                self.scan_response(control)

        for control in [item for item in self.controls if item.stage == "gain"]:
            self.one_dimensional_search(control, self.gain_objective, local=local)

        joint = [
            item for item in self.controls
            if item.stage in {"timing", "polarization", "gain", "joint"}
            or (item.stage == "vbs" and self.response_target(item) is not None)
        ]
        if joint:
            iterations = int(self.config.get("search", {}).get("joint_iterations", 8))
            self.spsa(joint, max(2, iterations // 2) if local else iterations, "joint", local=True)

        current_values = self.backend.values()
        candidates: list[tuple[str, dict[str, float]]] = [("current", current_values), ("baseline", baseline_values)]
        if self.best_values is not None:
            candidates.append(("best_seen", dict(self.best_values)))
        unique: dict[tuple[tuple[str, float], ...], tuple[str, dict[str, float]]] = {}
        for name, values in candidates:
            unique[tuple(sorted(values.items()))] = (name, values)
        ranked: list[tuple[Metrics, str, dict[str, float]]] = []
        for name, values in unique.values():
            self.set_values(values)
            measured = self.measure("final_" + name, repeats=self.final_repeats)
            ranked.append((measured, name, values))
        final, selected_name, selected_values = max(ranked, key=lambda item: (item[0].ready, item[0].score))
        self.set_values(selected_values)
        emit("final_state", source=selected_name, score=final.score, controls=selected_values)
        state = {
            "time": utc_now(),
            "geometry": dataclasses.asdict(self.geometry or self.nominal_geometry),
            "controls": self.backend.values(),
            "metrics": final.to_dict(),
            "history_file": "measurements.jsonl",
        }
        write_json(self.output_dir / "state.json", state)
        write_json(self.output_dir / "result.json", {
            "ready": final.ready,
            "baseline": baseline.to_dict(include_layers=False),
            "final": final.to_dict(),
            "controls": self.backend.values(),
        })
        emit("calibration_done", ready=final.ready, baseline_score=baseline.score,
             final_score=final.score, layers=final.usable_layers, state=str(self.output_dir / "state.json"))
        return final

    def restore_state(self, state: Mapping[str, Any]) -> None:
        geometry = state.get("geometry")
        if isinstance(geometry, Mapping):
            self.geometry = Geometry(**{field.name: geometry[field.name] for field in dataclasses.fields(Geometry)})
        controls = state.get("controls")
        if isinstance(controls, Mapping):
            self.backend.adopt({str(name): float(value) for name, value in controls.items()})

    def monitor(self, state_path: Path, once: bool = False) -> None:
        state = read_json(state_path)
        self.restore_state(state)
        reference = state.get("metrics", {})
        reference_score = float(reference.get("score", 0.0))
        reference_layers = int(reference.get("usable_layers", 0))
        acquisition = self.config.get("acquisition", {}) if isinstance(self.config.get("acquisition", {}), Mapping) else {}
        interval = float(acquisition.get("monitor_interval_s", 300))
        score_fraction = float(acquisition.get("retune_score_fraction", 0.90))
        layer_loss = int(acquisition.get("retune_layer_loss", 5))
        emit("monitor_start", reference_score=reference_score, reference_layers=reference_layers, interval_s=interval)
        while True:
            current = self.measure("monitor")
            needs_retune = current.score < reference_score * score_fraction or current.usable_layers < reference_layers - layer_loss
            if needs_retune:
                emit("retune_start", score=current.score, layers=current.usable_layers)
                current = self.calibrate(local=True)
                if current.score > reference_score:
                    reference_score = current.score
                    reference_layers = current.usable_layers
            if once:
                return
            time.sleep(max(interval, 1.0))
