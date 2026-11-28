# Coupled-fiber-ring automatic calibration

`fiber_ring_autocalibrate.py` restores the operating point of the coupled-fiber-ring synthetic-time lattice before an experiment and returns to it when the apparatus drifts.

The program does not maximize total brightness. A bright first return can coexist with rapid decay, poor long/short-loop balance, polarization leakage, wrong time windows, or oscilloscope clipping. It reconstructs the lattice from the same frozen CH1/CH2/CH3 frame and optimizes the quantities that determine whether a deep propagation experiment is usable:

- pulse-window alignment in
  \[
  t(m,n)=t_0+m\bar T+\frac{n\Delta T}{2};
  \]
- number of layers whose integrated signal remains above the same-frame background;
- late-layer signal-to-noise ratio;
- round-to-round logarithmic gain and gain curvature;
- long/short-loop balance after detector-response correction;
- off-grid background, CH3 leakage, and digitizer clipping.

The scalar objective is a weighted geometric mean,

\[
Q=100\exp\!\left(\frac{\sum_i w_i\log q_i}{\sum_i w_i}\right),
\]

so a large early pulse cannot hide a collapsed late lattice or a clipped channel.

## What it moves

The supplied configuration follows the present double-PM, short-loop-MZM layout. Filter positions are represented as offsets from the physical position at program start; the program never assumes that the absolute angle is zero.

Enabled commands:

- VBS voltage: held fixed until `target_split_long` is supplied; then a response scan interpolates the requested split;
- short-loop MZM voltage: full response scan after polarization has moved; by default the program selects the voltage with the strongest deep-propagation score, while an explicit normalized transmission target can be supplied for a particular experiment;
- long/short filter angle: relative `filter-jog` moves around the present physical position, optimized jointly from same-frame optical feedback.

The configuration also contains disabled entries for four MPC320 units (twelve paddles), the two live EDFA adjustments, and the trigger delay. Their physical roles are included, but their exact gateway role strings or argument names are local to the experiment workstation. Fill the command templates and set `enabled` to `true`; the optimization code does not change.

Detector factors `gain_ch1` and `gain_ch2` convert the two readout chains to one intensity scale. `target_split_long` is `null` by default, so the program leaves VBS unchanged. After measuring the CH1/CH2 response ratio, enter the two factors and the desired corrected split, such as `0.5`; only then does automatic VBS balancing run.

## Run

Python 3.10 or newer and NumPy are required.

Create the local configuration:

```powershell
python fiber_ring_autocalibrate.py write-config fiber_ring.json
```

Score an existing capture without moving hardware:

```powershell
python fiber_ring_autocalibrate.py score fiber_ring.json D:\Fiber_Loop_MZM\Physics\capture_directory
```

Restore the full operating point:

```powershell
python fiber_ring_autocalibrate.py calibrate fiber_ring.json `
  --output D:\Fiber_Loop_MZM\Physics\autocal\run_001
```

Continue measuring every five minutes and make a local correction when the score or usable depth falls:

```powershell
python fiber_ring_autocalibrate.py monitor fiber_ring.json `
  --state D:\Fiber_Loop_MZM\Physics\autocal\run_001\state.json
```

A single monitoring pass is useful from another agent program:

```powershell
python fiber_ring_autocalibrate.py monitor fiber_ring.json `
  --state D:\Fiber_Loop_MZM\Physics\autocal\run_001\state.json `
  --once
```

The gateway capture command defaults to:

```text
scope-capture-same-frame output_dir=... channels=1,2,3 formal=false build_lattice=false
```

The trace reader accepts NPZ, JSON, a multi-channel CSV, or a directory containing paired channel CSV files.

## Search sequence

A full run uses the following order:

1. Fit `t0`, mean round-trip time, and long/short delay from early pulse centers.
2. Optimize all enabled polarization and filter coordinates jointly with simultaneous perturbations. The number of optical measurements per iteration is independent of the number of paddles.
3. Re-scan the short-loop MZM because its transmission curve changes with polarization. Without an explicit transmission target, retain the voltage with the strongest deep-propagation score rather than forcing the historical half-transmission point.
4. When a corrected split target is supplied, re-scan the VBS and interpolate the voltage that gives that split.
5. Move enabled gain controls toward the requested per-layer logarithmic slope.
6. Perform a short joint refinement, then repeat the best, current, and initial states and leave the apparatus at the strongest repeated state.

`--local` reduces perturbation size and iteration count. `monitor` uses this mode rather than repeating every full scan.

## Depth target

`targets.layers` states the propagation depth needed by the next experiment. Set it to 40 for a 40-layer study, 60 for a 60-layer study, or 100 when the complete hundred-layer lattice is required. `lattice.layers` must cover at least the same depth.

The target logarithmic gain should normally remain slightly negative. A positive per-round gain may produce a long trace by amplification while increasing ASE, clipping, and instability. The score therefore rewards depth and late-layer SNR together with a bounded gain slope.

## Results

Each run writes:

- `state.json`: fitted lattice coordinates, final controls, and measured operating point;
- `result.json`: initial and final values;
- `measurements.jsonl`: every optical measurement and the controls used for it;
- `response_curves.json`: measured VBS/MZM response curves;
- `captures/`: the same-frame source data returned by the gateway.

The main program writes the workstation-specific command mapping and numerical targets with `write-config`, so the configuration has a single source in the executable code.

## Rehearsal without instruments

Set `"backend": "simulated"` and use the same `calibrate` command. The included surrogate propagates coherent long/short amplitudes, applies VBS mixing, MZM transmission, gain, polarization loss, leakage, noise, and clipping. It exists to exercise the complete program path; it is not a replacement for measurements from the apparatus.
