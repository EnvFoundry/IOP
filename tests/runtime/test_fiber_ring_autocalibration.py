from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "examples" / "iop" / "fiber-ring-autocalibration" / "fiber_ring_autocalibrate.py"
SPEC = importlib.util.spec_from_file_location("fiber_ring_autocalibrate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FiberRingAutocalibrationTests(unittest.TestCase):
    def test_example_configuration_matches_present_topology(self) -> None:
        config = MODULE.example_configuration()
        controls = config["controls"]
        names = {item["name"] for item in controls}
        self.assertIn("mzm_short_v", names)
        self.assertNotIn("mzm_long_v", names)
        self.assertIsNone(config["targets"]["target_split_long"])
        mzm = next(item for item in controls if item["name"] == "mzm_short_v")
        self.assertNotIn("target", mzm)
        mpc = [item for item in controls if item["name"].startswith("mpc_")]
        self.assertEqual(len(mpc), 12)
        self.assertTrue(all(not item["enabled"] for item in mpc))
        filters = [item for item in controls if item["name"].startswith("filter_")]
        self.assertTrue(all(item["set_mode"] == "relative" for item in filters))
        self.assertTrue(all(item["set_command"][0] == "filter-jog" for item in filters))
        self.assertIn("channels=1,2,3", config["gateway"]["capture_command"])

    def test_npz_trace_round_trip_and_lattice_analysis(self) -> None:
        config = MODULE.example_configuration()
        config["backend"] = "simulated"
        config["lattice"]["layers"] = 18
        config["targets"]["layers"] = 18
        geometry, controls = MODULE.parse_configuration(config)
        backend = MODULE.SimulatedBackend(config, controls, seed=17)
        trace = backend.capture("unit")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            np.savez(path, time_s=trace.time_s, ch1=trace.ch1, ch2=trace.ch2, ch3=trace.ch3)
            loaded = MODULE.load_trace(path)
        fitted = MODULE.fit_geometry(loaded, geometry, config["lattice"])
        metrics = MODULE.analyze_trace(loaded, fitted, config)
        self.assertLess(abs(fitted.roundtrip_s - geometry.roundtrip_s), 5e-9)
        self.assertLess(abs(fitted.site_spacing_s - geometry.site_spacing_s), 5e-9)
        self.assertGreater(metrics.lattice_concentration, 0.25)
        self.assertGreater(metrics.usable_layers, 5)
        self.assertFalse(metrics.balance_known)

    def test_gateway_templates_can_use_other_control_values(self) -> None:
        config = MODULE.example_configuration()
        controls = [
            MODULE.Control("a", "joint", 0.0, 2.0, 0.1, 0.01, 0.4, set_command=["set", "a={value}", "b={b}"]),
            MODULE.Control("b", "joint", 0.0, 2.0, 0.1, 0.01, 1.3, set_command=["set", "b={value}"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            backend = MODULE.GatewayBackend(config, controls, Path(directory))
            calls: list[tuple[list[str], dict[str, str]]] = []
            backend._run = lambda tokens, substitutions: calls.append((list(tokens), dict(substitutions))) or ""  # type: ignore[method-assign]
            backend.adopt({"a": 0.6, "b": 1.3})
            self.assertEqual(calls, [])
            backend.set(controls[0], 0.7)
        self.assertEqual(calls[-1][1]["value"], "0.7")
        self.assertEqual(calls[-1][1]["delta"], "0.1")
        self.assertEqual(calls[-1][1]["scaled_delta_int"], "0")
        self.assertEqual(calls[-1][1]["b"], "1.3")

    def test_vbs_stays_fixed_until_corrected_split_is_defined(self) -> None:
        config = MODULE.example_configuration()
        config["backend"] = "simulated"
        config["lattice"]["layers"] = 8
        config["targets"].update({
            "layers": 8,
            "late_snr": 2.0,
            "min_layer_snr": 2.0,
            "lattice_concentration": 0.25,
            "max_clipping_fraction": 0.05,
        })
        config["acquisition"].update({"repeats": 1, "final_repeats": 1})
        config["search"].update({"spsa_iterations": 1, "joint_iterations": 1})
        for item in config["controls"]:
            item["enabled"] = item["name"] == "vbs_v"
        geometry, controls = MODULE.parse_configuration(config)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            backend = MODULE.SimulatedBackend(config, controls, seed=3)
            initial = backend.values()["vbs_v"]
            calibrator = MODULE.Calibrator(config, backend, geometry, controls, Path(directory))
            calibrator.calibrate(local=True)
            self.assertEqual(backend.values()["vbs_v"], initial)

    def test_simulated_recovery_improves_a_displaced_state(self) -> None:
        config = MODULE.example_configuration()
        config["backend"] = "simulated"
        config["lattice"]["layers"] = 24
        config["targets"].update({
            "layers": 24,
            "late_snr": 3.0,
            "min_layer_snr": 3.0,
            "lattice_concentration": 0.40,
            "log_gain_tolerance": 0.018,
            "max_clipping_fraction": 0.01,
        })
        config["acquisition"].update({"repeats": 1, "final_repeats": 1})
        config["search"].update({
            "spsa_iterations": 5,
            "joint_iterations": 3,
            "minimum_improvement": 0.05,
        })
        for item in config["controls"]:
            name = item["name"]
            if name == "vbs_v":
                item.update({"initial": 0.35, "scan_points": 9})
            elif name == "mzm_short_v":
                item.update({"initial": 4.6, "scan_points": 11})
            elif name == "filter_long_deg":
                item["initial"] = -7.0
            elif name == "filter_short_deg":
                item["initial"] = 7.0
            elif name == "gain_long":
                item.update({
                    "lower": -0.04, "upper": 0.02, "step": 0.008,
                    "resolution": 0.0005, "initial": -0.028, "enabled": True,
                })
            elif name == "gain_short":
                item.update({
                    "lower": -0.04, "upper": 0.02, "step": 0.008,
                    "resolution": 0.0005, "initial": -0.025, "enabled": True,
                })
            else:
                item["enabled"] = False

        geometry, controls = MODULE.parse_configuration(config)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            backend = MODULE.SimulatedBackend(config, controls, seed=9)
            calibrator = MODULE.Calibrator(config, backend, geometry, controls, output)
            calibrator.calibrate()
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            baseline = result["baseline"]
            final = result["final"]
            self.assertGreater(final["score"], 2.0 * baseline["score"])
            self.assertGreaterEqual(final["usable_layers"], baseline["usable_layers"])
            self.assertTrue((output / "state.json").exists())
            self.assertTrue((output / "measurements.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
