# Space-Based Tracker Analysis (Question 1)

`space_tracker_orekit.py` models a space-based tracker and the TLE object NORAD 63223 over a one-day window
(2025-09-01 00:00:00 to 2025-09-02 00:00:00 UTC) using Orekit 13.1. It reports every **crossing** (object inside the
tracker's 30° field of view) and every **visible event** (inside the FOV, sunlit, and closer than 1000 km).

## Prerequisites and Installation

- Python 3 (developed on 3.13)
- `orekit` 13.1 from conda-forge (the JCC-based wrapper, not `orekit-jpype`), plus NumPy, SciPy and Matplotlib
- `orekit-data.zip` in the same folder as the script. The script exits with a message if it is missing.

```bash
conda env create -f environment.yml
conda activate space-tracker-orekit
```

## Running the Script

```bash
python space_tracker_orekit.py --plot
```

Arguments:

- `--outdir DIR` – output folder for the CSVs and plot (default `results_orekit/`)
- `--plot` – also write the 3-panel PNG (range, off-boresight angle, sunlit flag)
- `--compare` – print a side-by-side with reference results. It reads the CSVs in `--ref` (default `results/`) and falls back to the validation values from the assignment if they are not there.
- `--fixed-maxcheck S` – use a constant detector step of `S` seconds instead of the adaptive one (see the last section)

A full run takes about 7 s.

## Outputs

Written to `results_orekit/`:

- `crossings.csv` – start, end, duration, minimum range and off-boresight angle, sunlit time, clip flags
- `visible_events.csv` – the same columns plus `crossing_id`
- `close_approaches.csv` – every pass inside 1000 km, with minimum range and whether the object was in the FOV
- `space_tracker_orekit.png` – the plot (only with `--plot`)

The console also prints the crossing and visible-event tables and a sanity line (altitude ranges, minimum range of the
day, Earth-blocked samples).

## Methodology and Assumptions

The tracker is propagated numerically (`NumericalPropagator`, Dormand-Prince 8(5,3)) from the osculating elements at
the epoch. The object is propagated with SGP4 (`TLEPropagator`). Both are evaluated in TEME, which Orekit accepts as a
pseudo-inertial frame, so it is treated as inertial over the day. The sensor boresight is the inertial velocity
direction (`LofOffset`, TNW), and the FOV is a circular cone with a 15° half-angle. The window is exactly 24 h from the
tracker epoch, so the object is propagated backward for the first 14 h until its TLE epoch.

- **Keplerian singularity.** Eccentricity is 0, so the argument of perigee is undefined. The script combines argument
  of perigee and mean anomaly into a single argument of latitude (331.7425°) and builds the orbit from that.
- **Tracker orbit.** Gravity is two-body plus J2 only (degree 2, order 0), a compromise between accuracy and speed over
  one day. No drag, solar radiation pressure or third bodies. Degree and order are constants at the top of the script.
- **Tracker mass.** 100 kg, because Orekit's `SpacecraftState` requires one. It has no effect, since only gravity acts.
- **Earth constants.** The tracker uses the gravity field's μ and radius (eigen-6s), which differ slightly from WGS-84.
  Altitudes and occlusion use the WGS-84 ellipsoid.
- **Illumination.** "Sunlit" means outside Earth's umbra. Penumbra counts as sunlit. Only the object has to be lit.
- **Earth occlusion.** Checked numerically on 400 samples per crossing (straight tracker-to-object segment against the
  WGS-84 ellipsoid): 0 blocked. It cannot constrain visibility here. From about 500 km the Earth's limb lies roughly
  22° below the boresight, outside the 15° cone.
- **Visible event.** In FOV, sunlit and within 1000 km all at the same time. Crossings also list minimum range, so the
  looser reading ("closer than 1000 km at some point during the crossing") can be checked. Both give the same result here.
- **Not modelled.** Sun, Moon and bright-Earth exclusion angles, light-time delay, atmosphere along the line of sight,
  pointing error and manoeuvres.

## Note on Event Detection

At closest approach (2.55 km at about 02:04:28.8 UTC) the object crosses the edge of the cone in 0.18 s. A fixed
1 s detector step skips it, and the script then reports one crossing instead of two. The script therefore uses an
`AdaptiveMaxCheck` interval of `0.1 × range / relative speed`, clipped between 5 ms and 5 s, with a 1 ms threshold. It
checks every few seconds when the objects are far apart and every few milliseconds during a close pass. Holding a fixed
step of 0.02 s finds the event too, but the run is about 35 times slower than the adaptive one.

Use `--fixed-maxcheck 1.0` to reproduce the miss.
