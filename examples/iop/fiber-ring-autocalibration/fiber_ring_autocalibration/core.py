from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

EPS = 1e-15

@dataclass(frozen=True)
class Trace:
    time_s: np.ndarray
    ch1: np.ndarray
    ch2: np.ndarray
    ch3: np.ndarray | None = None
    source: str = ""

    def __post_init__(self) -> None:
        arrays = [self.time_s, self.ch1, self.ch2]
        if self.ch3 is not None:
            arrays.append(self.ch3)
        lengths = {len(np.asarray(value)) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 16:
            raise ValueError("time and channel arrays must have the same nontrivial length")
        if not np.all(np.isfinite(self.time_s)):
            raise ValueError("time array contains non-finite values")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("time array must be strictly increasing")
        for name, value in (("ch1", self.ch1), ("ch2", self.ch2), ("ch3", self.ch3)):
            if value is not None and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class Geometry:
    t0_s: float
    roundtrip_s: float
    site_spacing_s: float
    pulse_window_s: float
    layers: int

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pulse-center times, layer indices, and synthetic positions."""
        times: list[float] = []
        layers: list[int] = []
        sites: list[int] = []
        for m in range(self.layers + 1):
            for n in range(-m, m + 1, 2):
                times.append(self.t0_s + m * self.roundtrip_s + 0.5 * n * self.site_spacing_s)
                layers.append(m)
                sites.append(n)
        return np.asarray(times), np.asarray(layers), np.asarray(sites)


@dataclass
class Metrics:
    score: float
    ready: bool
    usable_layers: int
    target_layers: int
    lattice_concentration: float
    late_snr: float
    median_snr: float
    log_gain_per_layer: float
    gain_curvature: float
    early_split_long: float
    balance_known: bool
    balance_error: float
    leakage_fraction: float
    clipping_fraction: float
    pulse_energy_long: float
    pulse_energy_short: float
    background_sigma_long: float
    background_sigma_short: float
    t0_s: float
    roundtrip_s: float
    site_spacing_s: float
    components: dict[str, float] = field(default_factory=dict)
    layer_energy_long: list[float] = field(default_factory=list)
    layer_energy_short: list[float] = field(default_factory=list)
    layer_snr: list[float] = field(default_factory=list)

    def to_dict(self, include_layers: bool = True) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        if not include_layers:
            result.pop("layer_energy_long", None)
            result.pop("layer_energy_short", None)
            result.pop("layer_snr", None)
        return result


@dataclass
class Control:
    name: str
    stage: str
    lower: float
    upper: float
    step: float
    resolution: float
    initial: float
    settle_s: float = 0.0
    set_command: list[str] = field(default_factory=list)
    set_mode: str = "absolute"  # absolute or relative
    command_scale: float = 1.0
    response: str | None = None
    target: float | None = None
    scan_points: int = 13
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Control":
        required = ("name", "stage", "lower", "upper", "step", "resolution", "initial")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError("control is missing " + ", ".join(missing))
        control = cls(
            name=str(value["name"]),
            stage=str(value["stage"]),
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            step=float(value["step"]),
            resolution=float(value["resolution"]),
            initial=float(value["initial"]),
            settle_s=float(value.get("settle_s", 0.0)),
            set_command=[str(item) for item in value.get("set_command", [])],
            set_mode=str(value.get("set_mode", "absolute")),
            command_scale=float(value.get("command_scale", 1.0)),
            response=None if value.get("response") is None else str(value["response"]),
            target=None if value.get("target") is None else float(value["target"]),
            scan_points=int(value.get("scan_points", 13)),
            enabled=bool(value.get("enabled", True)),
        )
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", control.name) is None:
            raise ValueError("control names must be Python-style identifiers for command templates")
        if not control.lower <= control.initial <= control.upper:
            raise ValueError(f"{control.name}.initial lies outside its range")
        if control.lower >= control.upper or control.step <= 0 or control.resolution <= 0:
            raise ValueError(f"{control.name} has an invalid range or step")
        if control.set_mode not in {"absolute", "relative"}:
            raise ValueError(f"{control.name}.set_mode must be absolute or relative")
        if control.command_scale == 0:
            raise ValueError(f"{control.name}.command_scale must be nonzero")
        return control


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(destination)


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"time": utc_now(), "event": event, **fields}, sort_keys=True, allow_nan=False), flush=True)


def _mpc_placeholders() -> list[dict[str, Any]]:
    """Return the four present MPC units as disabled local mappings.

    The hardware records identify four controllers and twelve paddles, while the
    exact gateway role strings remain workstation-local. A disabled mapping is
    preferable to silently moving the wrong controller.
    """
    units = [
        ("pre_long", "REPLACE_PRE_LONG_ROLE", "38540474"),
        ("pre_short", "REPLACE_PRE_SHORT_ROLE", "38540674"),
        ("post_long", "REPLACE_POST_LONG_ROLE", "38541124"),
        ("post_short", "REPLACE_POST_SHORT_ROLE", "38541134"),
    ]
    controls: list[dict[str, Any]] = []
    for name, role, serial in units:
        for paddle in (1, 2, 3):
            controls.append({
                "name": f"mpc_{name}_p{paddle}_deg",
                "stage": "polarization",
                "lower": -30.0,
                "upper": 30.0,
                "step": 1.0,
                "resolution": 0.1,
                "initial": 0.0,
                "settle_s": 0.3,
                "enabled": False,
                "set_mode": "relative",
                "set_command": [
                    "mpc320-jog",
                    f"role={role}",
                    f"paddle={paddle}",
                    "delta_deg={delta}",
                    "max_step_deg=2",
                    "tolerance_deg=0.05",
                ],
                "hardware_serial": serial,
            })
    return controls


def example_configuration() -> dict[str, Any]:
    """Configuration for the present double-PM, short-loop-MZM platform."""
    controls: list[dict[str, Any]] = [
        {
            "name": "vbs_v", "stage": "vbs", "lower": 0.0, "upper": 4.8,
            "step": 0.4, "resolution": 0.01, "initial": 2.43, "settle_s": 0.2,
            "set_command": ["vbs-set-dc", "voltage={value}"],
            "response": "early_split_long", "scan_points": 17,
        },
        {
            "name": "mzm_short_v", "stage": "mzm", "lower": -2.0, "upper": 5.0,
            "step": 0.35, "resolution": 0.005, "initial": 1.653, "settle_s": 0.2,
            "set_command": ["awg-set-dc", "role=mzm-short", "voltage={value}", "output=on"],
            "response": "early_short_energy", "scan_points": 25,
        },
        {
            "name": "filter_long_deg", "stage": "polarization", "lower": -8.0, "upper": 8.0,
            "step": 1.0, "resolution": 0.1, "initial": 0.0, "settle_s": 0.0,
            "set_mode": "relative", "command_scale": 1000.0,
            "set_command": [
                "filter-jog", "role=long", "steps={scaled_delta_int}",
                "speed_hz=400", "settle_s=3.5",
            ],
        },
        {
            "name": "filter_short_deg", "stage": "polarization", "lower": -8.0, "upper": 8.0,
            "step": 1.0, "resolution": 0.1, "initial": 0.0, "settle_s": 0.0,
            "set_mode": "relative", "command_scale": 1000.0,
            "set_command": [
                "filter-jog", "role=short", "steps={scaled_delta_int}",
                "speed_hz=400", "settle_s=3.5",
            ],
        },
    ]
    controls.extend(_mpc_placeholders())
    controls.extend([
        {
            "name": "gain_long", "stage": "gain", "lower": -20.0, "upper": 20.0,
            "step": 1.0, "resolution": 0.1, "initial": 0.0, "settle_s": 1.0,
            "enabled": False, "set_mode": "relative", "set_command": [],
        },
        {
            "name": "gain_short", "stage": "gain", "lower": -20.0, "upper": 20.0,
            "step": 1.0, "resolution": 0.1, "initial": 0.0, "settle_s": 1.0,
            "enabled": False, "set_mode": "relative", "set_command": [],
        },
        {
            "name": "trigger_delay_s", "stage": "timing", "lower": -1.0e-6, "upper": 1.0e-6,
            "step": 100.0e-9, "resolution": 1.0e-9, "initial": 0.0, "settle_s": 0.2,
            "enabled": False, "set_command": [],
        },
    ])
    return {
        "backend": "gateway",
        "gateway": {
            "path": "D:/Fiber_Loop_MZM/bin/lab_gateway.cmd",
            "timeout_s": 300,
            "capture_command": [
                "--daemon-timeout-s", "300", "scope-capture-same-frame",
                "output_dir={output_dir}", "channels=1,2,3", "formal=false", "build_lattice=false",
            ],
            "trace": {
                "time_keys": ["time_s", "time", "t", "x"],
                "channel_keys": {
                    "ch1": ["ch1", "CH1", "channel1", "long"],
                    "ch2": ["ch2", "CH2", "channel2", "short"],
                    "ch3": ["ch3", "CH3", "channel3", "leakage"],
                },
            },
        },
        "lattice": {
            "layers": 100,
            "roundtrip_s": 24.6e-6,
            "site_spacing_s": 212.29e-9,
            "pulse_window_s": 70e-9,
            "t0_s": None,
            "fit_early_layers": 18,
            "fit_roundtrip_fraction": 0.0005,
            "fit_spacing_fraction": 0.01,
        },
        "detectors": {
            "gain_ch1": 1.0,
            "gain_ch2": 1.0,
            "gain_ch3": 1.0,
            "clip_low": None,
            "clip_high": None,
        },
        "targets": {
            "layers": 100,
            "min_layer_snr": 4.0,
            "late_snr": 6.0,
            "lattice_concentration": 0.55,
            "target_log_gain_per_layer": -0.002,
            "log_gain_tolerance": 0.012,
            "target_split_long": None,
            "split_tolerance": 0.12,
            "max_leakage_fraction": 0.08,
            "max_clipping_fraction": 0.001,
        },
        "acquisition": {
            "repeats": 1,
            "final_repeats": 3,
            "monitor_interval_s": 300,
            "retune_score_fraction": 0.90,
            "retune_layer_loss": 5,
        },
        "search": {
            "seed": 9,
            "spsa_iterations": 12,
            "spsa_a": 0.12,
            "spsa_c": 0.08,
            "joint_iterations": 8,
            "minimum_improvement": 0.15,
        },
        "controls": controls,
        "notes": [
            "Replace the four MPC role placeholders with names returned by the local gateway, then enable them.",
            "Fill the two live EDFA commands and the hardware timing command before enabling those controls.",
            "Measure CH1/CH2 detector response factors, then set target_split_long to enable VBS balancing.",
            "Set targets.layers to the depth required by the next experiment; 100 requests the full historical reach.",
        ],
    }


def parse_configuration(value: Mapping[str, Any]) -> tuple[Geometry, list[Control]]:
    lattice = value.get("lattice", {})
    if not isinstance(lattice, Mapping):
        raise ValueError("lattice must be an object")
    t0 = lattice.get("t0_s")
    geometry = Geometry(
        t0_s=0.0 if t0 is None else float(t0),
        roundtrip_s=float(lattice.get("roundtrip_s", 24.6e-6)),
        site_spacing_s=float(lattice.get("site_spacing_s", 212.29e-9)),
        pulse_window_s=float(lattice.get("pulse_window_s", 70e-9)),
        layers=int(lattice.get("layers", 100)),
    )
    if geometry.layers < 2 or min(geometry.roundtrip_s, geometry.site_spacing_s, geometry.pulse_window_s) <= 0:
        raise ValueError("lattice times and layers must be positive")
    raw_controls = value.get("controls", [])
    if not isinstance(raw_controls, list):
        raise ValueError("controls must be a list")
    controls = [Control.from_dict(item) for item in raw_controls]
    names = [item.name for item in controls]
    if len(names) != len(set(names)):
        raise ValueError("control names must be unique")
    return geometry, controls
