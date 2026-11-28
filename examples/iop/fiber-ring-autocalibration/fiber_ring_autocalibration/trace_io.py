from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import EPS, Trace, read_json


def _pick_key(keys: Iterable[str], candidates: Sequence[str]) -> str | None:
    lookup = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _trace_from_mapping(data: Mapping[str, Any], source: str, hints: Mapping[str, Any] | None = None) -> Trace:
    hints = hints or {}
    time_candidates = list(hints.get("time_keys", [])) + ["time_s", "time", "t", "x"]
    channel_hints = hints.get("channel_keys", {}) if isinstance(hints.get("channel_keys", {}), Mapping) else {}
    time_key = _pick_key(data.keys(), time_candidates)
    if time_key is None:
        raise ValueError(f"{source}: no time array")

    def channel(name: str, defaults: list[str], required: bool) -> np.ndarray | None:
        candidates = list(channel_hints.get(name, [])) + defaults
        key = _pick_key(data.keys(), candidates)
        if key is None:
            if required:
                raise ValueError(f"{source}: no {name} array")
            return None
        return np.asarray(data[key], dtype=float).reshape(-1)

    return Trace(
        time_s=np.asarray(data[time_key], dtype=float).reshape(-1),
        ch1=channel("ch1", ["ch1", "channel1", "long", "y1"], True),  # type: ignore[arg-type]
        ch2=channel("ch2", ["ch2", "channel2", "short", "y2"], True),  # type: ignore[arg-type]
        ch3=channel("ch3", ["ch3", "channel3", "leakage", "y3"], False),
        source=source,
    )


def _load_npz(path: Path, hints: Mapping[str, Any] | None) -> Trace:
    with np.load(path, allow_pickle=False) as archive:
        mapping = {key: archive[key] for key in archive.files}
    return _trace_from_mapping(mapping, str(path), hints)


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(stream, dialect)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    header_is_text = any(not _looks_numeric(item) for item in rows[0])
    header = [item.strip() for item in rows[0]] if header_is_text else [f"col{i}" for i in range(len(rows[0]))]
    body = rows[1:] if header_is_text else rows
    numeric: list[list[float]] = []
    for row in body:
        if len(row) < len(header):
            continue
        try:
            numeric.append([float(item.strip()) for item in row[:len(header)]])
        except ValueError:
            continue
    if not numeric:
        raise ValueError(f"{path}: no numeric rows")
    return header, np.asarray(numeric, dtype=float)


def _looks_numeric(value: str) -> bool:
    try:
        float(value.strip())
        return True
    except ValueError:
        return False


def _load_csv(path: Path, hints: Mapping[str, Any] | None) -> Trace:
    header, values = _read_csv(path)
    mapping = {name: values[:, index] for index, name in enumerate(header)}
    if len(header) >= 3 and _pick_key(mapping.keys(), ["time_s", "time", "t", "x"]) is not None:
        return _trace_from_mapping(mapping, str(path), hints)
    raise ValueError(f"{path}: expected a multi-channel CSV")


def _load_csv_directory(path: Path, hints: Mapping[str, Any] | None) -> Trace:
    files = sorted(path.rglob("*.csv"))
    by_channel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for file in files:
        lowered = file.name.lower()
        name = next((candidate for candidate in ("ch1", "ch2", "ch3") if candidate in lowered), None)
        if name is None:
            continue
        header, values = _read_csv(file)
        if values.shape[1] < 2:
            continue
        time_index = 0
        value_index = 1
        for index, column in enumerate(header):
            low = column.lower()
            if "time" in low or low in {"t", "x"}:
                time_index = index
            if any(token in low for token in ("volt", "amplitude", "value", "ch")) and index != time_index:
                value_index = index
        by_channel[name] = (values[:, time_index], values[:, value_index])
    if not {"ch1", "ch2"} <= by_channel.keys():
        raise ValueError(f"{path}: no paired CH1/CH2 CSV files")
    time1, ch1 = by_channel["ch1"]
    time2, ch2 = by_channel["ch2"]
    if len(time1) != len(time2) or not np.allclose(time1, time2, rtol=0, atol=max(np.median(np.diff(time1)), EPS) * 0.1):
        ch2 = np.interp(time1, time2, ch2)
    ch3 = None
    if "ch3" in by_channel:
        time3, values3 = by_channel["ch3"]
        ch3 = values3 if len(time3) == len(time1) and np.allclose(time3, time1) else np.interp(time1, time3, values3)
    return Trace(time1, ch1, ch2, ch3, str(path))


def _load_json_trace(path: Path, hints: Mapping[str, Any] | None) -> Trace:
    value = read_json(path)
    if "trace" in value and isinstance(value["trace"], Mapping):
        value = value["trace"]
    return _trace_from_mapping(value, str(path), hints)


def load_trace(path: str | Path, hints: Mapping[str, Any] | None = None) -> Trace:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() == ".npz":
            return _load_npz(source, hints)
        if source.suffix.lower() == ".csv":
            return _load_csv(source, hints)
        if source.suffix.lower() == ".json":
            return _load_json_trace(source, hints)
        raise ValueError(f"unsupported trace file {source}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    candidates = sorted(source.rglob("*.npz"), key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            return _load_npz(candidate, hints)
        except (ValueError, KeyError):
            continue
    candidates = sorted(source.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            return _load_json_trace(candidate, hints)
        except (ValueError, KeyError, TypeError):
            continue
    return _load_csv_directory(source, hints)
