#!/usr/bin/env python3
"""Restore and maintain a coupled-fiber-ring operating point.

Typical use on the experiment workstation::

    python fiber_ring_autocalibrate.py write-config fiber_ring.json
    python fiber_ring_autocalibrate.py calibrate fiber_ring.json --output runs/calibration
    python fiber_ring_autocalibrate.py monitor fiber_ring.json --state runs/calibration/state.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fiber_ring_autocalibration import (  # noqa: E402
    Backend,
    Calibrator,
    Control,
    GatewayBackend,
    Geometry,
    Metrics,
    ReplayBackend,
    SimulatedBackend,
    Trace,
    analyze_trace,
    emit,
    example_configuration,
    fit_geometry,
    load_trace,
    parse_configuration,
    read_json,
    write_json,
)


def make_backend(config: Mapping[str, Any], controls: Sequence[Control], output_dir: Path,
                 replay: Sequence[str] | None = None) -> Backend:
    kind = str(config.get("backend", "gateway"))
    if replay:
        return ReplayBackend(replay, controls)
    if kind == "simulated":
        seed = int(config.get("search", {}).get("seed", 9)) if isinstance(config.get("search", {}), Mapping) else 9
        return SimulatedBackend(config, controls, seed)
    if kind == "gateway":
        return GatewayBackend(config, controls, output_dir)
    raise ValueError(f"unknown backend {kind}")


def command_score(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    nominal, _controls = parse_configuration(config)
    trace = load_trace(args.trace, config.get("gateway", {}).get("trace", {}))
    geometry = fit_geometry(trace, nominal, config.get("lattice", {}))
    metrics = analyze_trace(trace, geometry, config)
    print(json.dumps({"geometry": dataclasses.asdict(geometry), "metrics": metrics.to_dict()}, indent=2, allow_nan=False))
    return 0 if metrics.ready else 2


def command_calibrate(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    geometry, controls = parse_configuration(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    backend = make_backend(config, controls, output, args.replay)
    calibrator = Calibrator(config, backend, geometry, controls, output)
    final = calibrator.calibrate(local=args.local)
    return 0 if final.ready else 2


def command_monitor(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    geometry, controls = parse_configuration(config)
    output = Path(args.output or Path(args.state).parent)
    output.mkdir(parents=True, exist_ok=True)
    backend = make_backend(config, controls, output, args.replay)
    calibrator = Calibrator(config, backend, geometry, controls, output)
    calibrator.monitor(Path(args.state), once=args.once)
    return 0


def command_write_config(args: argparse.Namespace) -> int:
    destination = Path(args.path)
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it")
    write_json(destination, example_configuration())
    print(destination)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic operating-point recovery for a coupled-fiber-ring experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write-config", help="write a configuration using the present platform command names")
    write.add_argument("path")
    write.add_argument("--force", action="store_true")
    write.set_defaults(function=command_write_config)

    score = subparsers.add_parser("score", help="reconstruct and score an existing same-frame trace")
    score.add_argument("config")
    score.add_argument("trace")
    score.set_defaults(function=command_score)

    calibrate = subparsers.add_parser("calibrate", help="restore the operating point")
    calibrate.add_argument("config")
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--replay", action="append", help="use saved trace files/directories in sequence")
    calibrate.add_argument("--local", action="store_true", help="make only local moves around the current state")
    calibrate.set_defaults(function=command_calibrate)

    monitor = subparsers.add_parser("monitor", help="measure periodically and retune when the state has moved")
    monitor.add_argument("config")
    monitor.add_argument("--state", required=True)
    monitor.add_argument("--output")
    monitor.add_argument("--replay", action="append")
    monitor.add_argument("--once", action="store_true")
    monitor.set_defaults(function=command_monitor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except KeyboardInterrupt:
        emit("stopped")
        return 130
    except Exception as exc:
        emit("error", type=type(exc).__name__, message=str(exc))
        if os.environ.get("FIBER_RING_TRACEBACK") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
