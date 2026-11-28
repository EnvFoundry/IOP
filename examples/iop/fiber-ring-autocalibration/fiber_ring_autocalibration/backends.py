from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import Control, Geometry, Trace
from .traces import load_trace


class Backend:
    def values(self) -> dict[str, float]:
        raise NotImplementedError

    def adopt(self, values: Mapping[str, float]) -> None:
        """Adopt known actuator coordinates without moving the hardware."""
        raise NotImplementedError

    def set(self, control: Control, value: float) -> None:
        raise NotImplementedError

    def capture(self, label: str) -> Trace:
        raise NotImplementedError


class GatewayBackend(Backend):
    def __init__(self, config: Mapping[str, Any], controls: Sequence[Control], output_dir: Path):
        gateway = config.get("gateway", {})
        if not isinstance(gateway, Mapping):
            raise ValueError("gateway must be an object")
        self.path = Path(str(gateway.get("path", "D:/Fiber_Loop_MZM/bin/lab_gateway.cmd")))
        self.timeout_s = float(gateway.get("timeout_s", 300))
        self.capture_command = [str(item) for item in gateway.get("capture_command", [])]
        if not self.capture_command:
            raise ValueError("gateway.capture_command is empty")
        self.trace_hints = gateway.get("trace", {}) if isinstance(gateway.get("trace", {}), Mapping) else {}
        self.output_dir = output_dir
        active_without_command = [control.name for control in controls if control.enabled and not control.set_command]
        if active_without_command:
            raise ValueError("enabled gateway controls need set_command: " + ", ".join(active_without_command))
        self._controls = {control.name: control for control in controls}
        self._values = {control.name: control.initial for control in controls}
        self.sequence = 0

    def values(self) -> dict[str, float]:
        return dict(self._values)

    def adopt(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            control = self._controls.get(name)
            if control is not None:
                self._values[name] = float(np.clip(value, control.lower, control.upper))

    def _run(self, tokens: Sequence[str], substitutions: Mapping[str, Any]) -> str:
        rendered = [item.format(**substitutions) for item in tokens]
        if os.name == "nt" or self.path.suffix.lower() in {".cmd", ".bat"}:
            command = ["cmd.exe", "/d", "/s", "/c", str(self.path), *rendered]
        else:
            command = [str(self.path), *rendered]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_s, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"gateway command failed ({completed.returncode}): {' '.join(shlex.quote(item) for item in rendered)}\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    def set(self, control: Control, value: float) -> None:
        value = float(np.clip(value, control.lower, control.upper))
        previous = self._values[control.name]
        delta = value - previous
        if control.set_mode == "relative" and abs(delta) < 0.5 * control.resolution:
            return
        proposed = dict(self._values)
        proposed[control.name] = value
        substitutions = {name: f"{item:.12g}" for name, item in proposed.items()}
        scaled_value = value * control.command_scale
        scaled_delta = delta * control.command_scale
        substitutions.update({
            "value": f"{value:.12g}",
            "delta": f"{delta:.12g}",
            "previous": f"{previous:.12g}",
            "scaled_value": f"{scaled_value:.12g}",
            "scaled_delta": f"{scaled_delta:.12g}",
            "scaled_value_int": str(int(round(scaled_value))),
            "scaled_delta_int": str(int(round(scaled_delta))),
        })
        if control.set_command:
            self._run(control.set_command, substitutions)
        self._values[control.name] = value
        if control.settle_s > 0:
            time.sleep(control.settle_s)

    def capture(self, label: str) -> Trace:
        self.sequence += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "capture"
        destination = self.output_dir / "captures" / f"{self.sequence:05d}_{safe_label}"
        destination.mkdir(parents=True, exist_ok=True)
        self._run(self.capture_command, {"output_dir": str(destination)})
        return load_trace(destination, self.trace_hints)


class ReplayBackend(Backend):
    def __init__(self, traces: Sequence[str | Path], controls: Sequence[Control]):
        if not traces:
            raise ValueError("replay backend needs at least one trace")
        self.traces = [Path(item) for item in traces]
        self.index = 0
        self._controls = {control.name: control for control in controls}
        self._values = {control.name: control.initial for control in controls}

    def values(self) -> dict[str, float]:
        return dict(self._values)

    def adopt(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            control = self._controls.get(name)
            if control is not None:
                self._values[name] = float(np.clip(value, control.lower, control.upper))

    def set(self, control: Control, value: float) -> None:
        self._values[control.name] = float(np.clip(value, control.lower, control.upper))

    def capture(self, label: str) -> Trace:
        source = self.traces[min(self.index, len(self.traces) - 1)]
        self.index += 1
        return load_trace(source)


class SimulatedBackend(Backend):
    """Fast physical surrogate used for development and agent rehearsal."""

    def __init__(self, config: Mapping[str, Any], controls: Sequence[Control], seed: int = 4):
        self.config = config
        self.controls = {control.name: control for control in controls}
        self._values = {control.name: control.initial for control in controls}
        self.rng = np.random.default_rng(seed)
        lattice = config.get("lattice", {})
        self.geometry = Geometry(
            t0_s=5e-6 if lattice.get("t0_s") is None else float(lattice["t0_s"]),
            roundtrip_s=float(lattice.get("roundtrip_s", 24.6e-6)),
            site_spacing_s=float(lattice.get("site_spacing_s", 212.29e-9)),
            pulse_window_s=float(lattice.get("pulse_window_s", 70e-9)),
            layers=int(lattice.get("layers", 100)),
        )
        self.sequence = 0

    def values(self) -> dict[str, float]:
        return dict(self._values)

    def adopt(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            control = self.controls.get(name)
            if control is not None:
                self._values[name] = float(np.clip(value, control.lower, control.upper))

    def set(self, control: Control, value: float) -> None:
        self._values[control.name] = float(np.clip(value, control.lower, control.upper))
        if control.settle_s:
            time.sleep(min(control.settle_s, 0.001))

    def _value(self, name: str, default: float) -> float:
        return float(self._values.get(name, default))

    def capture(self, label: str) -> Trace:
        self.sequence += 1
        g = self.geometry
        dt = min(g.pulse_window_s / 8.0, 12e-9)
        end = g.t0_s + g.layers * g.roundtrip_s + 0.55 * g.layers * g.site_spacing_s + 3e-6
        time_s = np.arange(0.0, end, dt)
        ch1 = np.zeros_like(time_s)
        ch2 = np.zeros_like(time_s)
        ch3 = np.zeros_like(time_s)

        vbs = self._value("vbs_v", 2.414)
        theta = np.clip(vbs / 4.8, 0.0, 1.0) * math.pi / 2.0
        mzm_long = (
            0.5 - 0.5 * math.sin(math.pi * (self._value("mzm_long_v", 3.913) - 3.913) / 4.8)
            if "mzm_long_v" in self._values else 1.0
        )
        mzm_short = (
            0.5 - 0.5 * math.sin(math.pi * (self._value("mzm_short_v", 1.653) - 1.653) / 5.2)
            if "mzm_short_v" in self._values else 1.0
        )
        gain_long = math.exp(self._value("gain_long", -0.006))
        gain_short = math.exp(self._value("gain_short", -0.004))
        trigger = self._value("trigger_delay_s", 0.0)

        polarization_names = [name for name in self._values if name.startswith("mpc_") or name.startswith("filter_")]
        distance = 0.0
        for index, name in enumerate(polarization_names):
            optimum = 2.5 * math.sin(index + 0.4)
            scale = 9.0 if name.startswith("mpc_") else 4.0
            distance += ((self._values[name] - optimum) / scale) ** 2
        polarization = math.exp(-0.35 * distance)
        leakage = 0.015 + 0.18 * (1.0 - polarization)

        long: dict[int, complex] = {0: 1.0 + 0j}
        short: dict[int, complex] = {0: 0j}
        sigma = g.pulse_window_s / 5.0
        radius = max(2, int(math.ceil(4 * sigma / dt)))
        for m in range(g.layers + 1):
            for n in range(-m, m + 1, 2):
                a = long.get(n, 0j)
                b = short.get(n, 0j)
                center = g.t0_s + trigger + m * g.roundtrip_s + 0.5 * n * g.site_spacing_s
                index = int(round(center / dt))
                left = max(0, index - radius)
                right = min(len(time_s), index + radius + 1)
                if right <= left:
                    continue
                kernel = np.exp(-0.5 * ((time_s[left:right] - center) / sigma) ** 2)
                scale = 0.8 * polarization
                ch1[left:right] += scale * abs(a) ** 2 * kernel
                ch2[left:right] += scale * abs(b) ** 2 * kernel
                ch3[left:right] += scale * leakage * (abs(a) ** 2 + abs(b) ** 2) * kernel
            if m == g.layers:
                break
            next_long: dict[int, complex] = {}
            next_short: dict[int, complex] = {}
            positions = set(long) | set(short)
            for n in positions:
                a = long.get(n, 0j)
                b = short.get(n, 0j)
                out_long = (math.cos(theta) * a + 1j * math.sin(theta) * b) * math.sqrt(max(mzm_long, 0.01)) * gain_long
                out_short = (1j * math.sin(theta) * a + math.cos(theta) * b) * math.sqrt(max(mzm_short, 0.01)) * gain_short
                next_long[n + 1] = next_long.get(n + 1, 0j) + out_long
                next_short[n - 1] = next_short.get(n - 1, 0j) + out_short
            long, short = next_long, next_short

        noise = 0.0035 + 0.0015 * max(0.0, (gain_long + gain_short) / 2.0 - 1.0)
        ch1 += self.rng.normal(0.0, noise, len(ch1))
        ch2 += self.rng.normal(0.0, noise, len(ch2))
        ch3 += self.rng.normal(0.0, noise * 0.8, len(ch3))
        ch1 = np.clip(ch1, -0.05, 1.0)
        ch2 = np.clip(ch2, -0.05, 1.0)
        ch3 = np.clip(ch3, -0.05, 1.0)
        return Trace(time_s, ch1, ch2, ch3, f"simulation:{self.sequence}:{label}")
