#!/usr/bin/env python3
"""Space-based tracker vs. TLE object, done with Orekit (JCC wrapper, Orekit 13.1).

Tracker : NumericalPropagator (two-body + J2), attitude LofOffset(TEME, TNW) -> body +X = velocity.
Object  : TLEPropagator (SGP4).
Events  : FieldOfViewDetector (crossings), EclipseDetector (sunlit), range < 1000 km from
          ephemeris sampling + Brent root finding, then interval intersection.

Internal units are SI (m, s, rad); output is km, deg, UTC.
Times are carried around as float seconds since the window start T0 (no leap second in the window).
"""
import argparse
import csv
import datetime as dt
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import orekit
from orekit import JArray_double

orekit.initVM()

from java.io import File  # noqa: E402
from org.hipparchus.geometry.euclidean.threed import Vector3D  # noqa: E402
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # noqa: E402
from org.orekit.attitudes import LofOffset  # noqa: E402
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid  # noqa: E402
from org.orekit.data import DataContext, DirectoryCrawler, ZipJarCrawler  # noqa: E402
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel  # noqa: E402
from org.orekit.forces.gravity.potential import GravityFieldFactory  # noqa: E402
from org.orekit.frames import FramesFactory, LOFType  # noqa: E402
from org.orekit.geometry.fov import CircularFieldOfView  # noqa: E402
from org.orekit.orbits import CircularOrbit, OrbitType, PositionAngleType  # noqa: E402
from org.orekit.propagation import Propagator, SpacecraftState  # noqa: E402
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator  # noqa: E402
from org.orekit.propagation.events import EclipseDetector, EventsLogger, FieldOfViewDetector  # noqa: E402
from org.orekit.propagation.events.handlers import ContinueOnEvent  # noqa: E402
from org.orekit.propagation.events.intervals import PythonAdaptableInterval  # noqa: E402
from org.orekit.propagation.numerical import NumericalPropagator  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants, IERSConventions  # noqa: E402

# ----------------------------------------------------------------------------- inputs
_HERE = os.path.dirname(os.path.abspath(__file__))
OREKIT_DATA = os.path.join(_HERE, "orekit-data.zip")  # shipped next to this script

TLE_LINE1 = "1 63223U 25052P   25244.59601767  .00010814  00000-0  51235-3 0  9991"
TLE_LINE2 = "2 63223  97.4217 137.0451 0006365  74.2830 285.9107 15.19475170 25990"

EPOCH_UTC = (2025, 9, 1, 0, 0, 0.0)
WINDOW_S = 86400.0

SMA_M = 6878.0e3
ECC = 0.0
INC_DEG = 97.4
RAAN_DEG = 72.628
ARG_LAT_DEG = 331.7425  # omega + M (e = 0 -> perigee undefined)

FOV_FULL_DEG = 30.0
RANGE_LIMIT_M = 1000.0e3

GRAVITY_DEGREE, GRAVITY_ORDER = 2, 0  # J2 only; raise to add harmonics
POS_TOL_M = 1.0e-3  # integrator position tolerance
TRACKER_MASS_KG = 100.0

# detector settings
THRESHOLD_S = 1.0e-3
MAXCHECK_FAR_S = 5.0  # cap of the adaptive maxCheck
MAXCHECK_FLOOR_S = 0.005
MAXCHECK_FACTOR = 0.1  # maxCheck = FACTOR * range / relative speed
ECLIPSE_MAXCHECK_S = 30.0

SAMPLE_S = 5.0  # range sampling grid for minima / 1000 km crossings
SUN_RADIUS_M = Constants.SUN_RADIUS


# ----------------------------------------------------------------------------- helpers
def load_orekit_data(path):
    if not os.path.exists(path):
        sys.exit(f"orekit-data not found: {path}\nPlace orekit-data.zip in the same folder as this script.")
    mgr = DataContext.getDefault().getDataProvidersManager()
    f = File(path)
    if os.path.isdir(path):
        mgr.addProvider(DirectoryCrawler(f))
    else:
        mgr.addProvider(ZipJarCrawler(f))


class Scenario:
    """Holds Orekit objects shared by all analysis steps."""

    def __init__(self):
        self.utc = TimeScalesFactory.getUTC()
        self.t0 = AbsoluteDate(*EPOCH_UTC, self.utc)
        self.t1 = self.t0.shiftedBy(WINDOW_S)
        self.teme = FramesFactory.getTEME()
        self.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, self.itrf
        )
        self.sun = CelestialBodyFactory.getSun()
        self.tle = TLE(TLE_LINE1, TLE_LINE2)
        self.gravity = GravityFieldFactory.getNormalizedProvider(GRAVITY_DEGREE, GRAVITY_ORDER)
        self.mu = self.gravity.getMu()
        self.half_fov = math.radians(FOV_FULL_DEG / 2.0)

    # -- time conversions
    def date(self, t):
        return self.t0.shiftedBy(float(t))

    def fmt(self, t):
        """Seconds since T0 -> 'YYYY-MM-DD HH:MM:SS.mmm' UTC."""
        base = dt.datetime(*EPOCH_UTC[:5], int(EPOCH_UTC[5]))
        ms = int(round(float(t) * 1000.0))
        return (base + dt.timedelta(milliseconds=ms)).strftime("%Y-%m-%d %H:%M:%S.") + f"{ms % 1000:03d}"

    # -- propagators
    def object_propagator(self):
        """A fresh SGP4 propagator (TEME output)."""
        return TLEPropagator.selectExtrapolator(self.tle)

    def tracker_initial_orbit(self):
        return CircularOrbit(
            SMA_M, 0.0, 0.0,  # a, ex, ey (e = 0)
            math.radians(INC_DEG), math.radians(RAAN_DEG), math.radians(ARG_LAT_DEG),
            PositionAngleType.MEAN, self.teme, self.t0, self.mu,
        )

    def tracker_propagator(self):
        orbit = self.tracker_initial_orbit()
        tol = NumericalPropagator.tolerances(POS_TOL_M, orbit, OrbitType.CARTESIAN)
        integ = DormandPrince853Integrator(1.0e-3, 300.0, JArray_double.cast_(tol[0]), JArray_double.cast_(tol[1]))
        prop = NumericalPropagator(integ)
        prop.setOrbitType(OrbitType.CARTESIAN)
        prop.addForceModel(HolmesFeatherstoneAttractionModel(self.itrf, self.gravity))
        prop.setAttitudeProvider(LofOffset(self.teme, LOFType.TNW))  # body +X = velocity
        prop.setInitialState(SpacecraftState(orbit, TRACKER_MASS_KG))
        return prop


def vec(v):
    return np.array([v.getX(), v.getY(), v.getZ()])


def pos_vel(prop, sc, t):
    pv = Propagator.cast_(prop).getPVCoordinates(sc.date(t), sc.teme)
    return vec(pv.getPosition()), vec(pv.getVelocity())


class AdaptiveMaxCheck(PythonAdaptableInterval):
    """maxCheck = FACTOR * range / relative speed, clipped to [FLOOR, FAR].

    Called by the detector for every step with the *tracker* state; the object is evaluated
    with its own SGP4 propagator. Keeps the check interval short enough that a ~0.2 s FOV-edge
    graze during a 2.5 km pass cannot be skipped over, while staying coarse when far apart.
    """

    def __init__(self, sc, object_prop):
        super().__init__()
        self.sc, self.obj = sc, Propagator.cast_(object_prop)
        self.calls = 0

    def currentInterval(self, state, is_forward):
        self.calls += 1
        pv_o = self.obj.getPVCoordinates(state.getDate(), self.sc.teme)
        pv_t = state.getPVCoordinates(self.sc.teme)
        rng = pv_o.getPosition().distance(pv_t.getPosition())
        vrel = pv_o.getVelocity().distance(pv_t.getVelocity())
        return float(min(MAXCHECK_FAR_S, max(MAXCHECK_FLOOR_S, MAXCHECK_FACTOR * rng / max(vrel, 1.0))))

    def finalize(self):
        self.pythonDecRef()


def run_tracker(sc, object_prop, fixed_maxcheck=None):
    """Propagate the tracker over the window with the FOV detector; return (ephemeris, fov_events, fov_detector, interval)."""
    trk = sc.tracker_propagator()
    gen = trk.getEphemerisGenerator()
    logger = EventsLogger()
    interval = AdaptiveMaxCheck(sc, object_prop)
    max_check = interval if fixed_maxcheck is None else float(fixed_maxcheck)
    fov = FieldOfViewDetector(object_prop, CircularFieldOfView(Vector3D.PLUS_I, sc.half_fov, 0.0))
    fov = FieldOfViewDetector.cast_(
        fov.withMaxCheck(max_check).withThreshold(THRESHOLD_S).withHandler(ContinueOnEvent())
    )
    trk.addEventDetector(logger.monitorDetector(fov))
    trk.propagate(sc.t0, sc.t1)
    events = [(e.getDate().durationFrom(sc.t0), bool(e.isIncreasing())) for e in logger.getLoggedEvents()]
    return gen.getGeneratedEphemeris(), events, fov, interval


def off_boresight(sc, trk_eph, obj_prop, t):
    """Angle (rad) between tracker body +X (from the Orekit attitude) and the line of sight to the object."""
    d = sc.date(t)
    st = trk_eph.propagate(d)
    los = Propagator.cast_(obj_prop).getPVCoordinates(d, sc.teme).getPosition().subtract(
        st.getPVCoordinates(sc.teme).getPosition())
    los_body = st.getAttitude().getRotation().applyTo(los)  # TEME -> body frame
    return Vector3D.angle(los_body, Vector3D.PLUS_I)


# ----------------------------------------------------------------------------- interval algebra
def intervals_from_events(events, inside_sign, inside0, t_end):
    """Turn (time, increasing) detector events into closed intervals where the condition holds.

    inside_sign : sign of g while the condition holds (+1 / -1).
    inside0     : condition status at t = 0.  Returns list of (a, b, clipped_start, clipped_end).
    """
    out, start, clip_s = [], (0.0 if inside0 else None), inside0
    for t, inc in events:
        entering = inc == (inside_sign > 0)
        if entering:
            assert start is None, "two consecutive entries - detector missed an exit"
            start, clip_s = t, False
        else:
            assert start is not None, "exit without entry - detector missed an entry"
            out.append((start, t, clip_s, False))
            start = None
    if start is not None:
        out.append((start, t_end, clip_s, True))
    return out


def intersect(*lists):
    """Intersection of several interval lists [(a, b, ...)]; returns [(a, b)]."""
    cur = [(a, b) for a, b, *_ in lists[0]]
    for other in lists[1:]:
        nxt = []
        for a, b in cur:
            for c, d, *_ in other:
                lo, hi = max(a, c), min(b, d)
                if hi > lo:
                    nxt.append((lo, hi))
        cur = nxt
    return sorted(cur)


def overlap_s(a, b, ivs):
    return sum(max(0.0, min(b, d) - max(a, c)) for c, d, *_ in ivs)


def minimize_on(f, a, b, n=200):
    """Global minimum of f on [a, b]: grid + bounded refinement around the best sample."""
    ts = np.linspace(a, b, n + 1)
    vals = np.array([f(t) for t in ts])
    k = int(np.argmin(vals))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, n)]
    best_t, best_v = ts[k], vals[k]
    if hi > lo:
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
        if res.fun < best_v:
            best_t, best_v = res.x, res.fun
    return float(best_t), float(best_v)


# ----------------------------------------------------------------------------- analysis
class Analysis:
    def __init__(self, sc, fixed_maxcheck=None):
        self.sc = sc
        self.W = WINDOW_S
        self.obj = sc.object_propagator()
        t_run = time.time()
        self.fixed_maxcheck = fixed_maxcheck
        self.trk_eph, self.fov_events, self.fov, self.interval = run_tracker(sc, self.obj, fixed_maxcheck)
        self.run_time = time.time() - t_run
        self.trk_pv = Propagator.cast_(self.trk_eph)
        self.obj_pv = Propagator.cast_(self.obj)

    # -- geometry primitives (t in seconds since T0)
    def rng(self, t):
        d = self.sc.date(t)
        return self.obj_pv.getPVCoordinates(d, self.sc.teme).getPosition().distance(
            self.trk_pv.getPVCoordinates(d, self.sc.teme).getPosition())

    def offb(self, t):
        return off_boresight(self.sc, self.trk_eph, self.obj, t)

    def lat_deg(self, t):
        d = self.sc.date(t)
        p = self.trk_pv.getPVCoordinates(d, self.sc.teme).getPosition()
        return math.degrees(self.sc.earth.transform(p, self.sc.teme, d).getLatitude())

    def earth_blocked(self, t):
        """True if the straight tracker->object segment passes through the Earth ellipsoid."""
        sc, d = self.sc, self.sc.date(t)
        tr = sc.teme.getStaticTransformTo(sc.itrf, d)
        a = vec(tr.transformPosition(self.trk_pv.getPVCoordinates(d, sc.teme).getPosition()))
        b = vec(tr.transformPosition(self.obj_pv.getPVCoordinates(d, sc.teme).getPosition()))
        k = np.array([1.0, 1.0, Constants.WGS84_EARTH_EQUATORIAL_RADIUS / (
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS * (1.0 - Constants.WGS84_EARTH_FLATTENING))])
        a, b = a * k, b * k  # ellipsoid -> sphere
        ab = b - a
        s = np.clip(-np.dot(a, ab) / np.dot(ab, ab), 0.0, 1.0)
        return float(np.linalg.norm(a + s * ab)) < Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    # -- FOV crossings
    def crossings(self):
        g0 = self.fov.g(self.trk_eph.propagate(self.sc.t0))
        in0 = self.offb(0.0) < self.sc.half_fov
        inside_sign = (1 if g0 > 0 else -1) * (1 if in0 else -1)
        self.fov_sign_note = f"g {'>' if inside_sign < 0 else '<'} 0 outside the FOV; decreasing = entry"
        return intervals_from_events(self.fov_events, inside_sign, in0, self.W)

    # -- sunlight (EclipseDetector on the object, umbra only)
    def sunlit(self):
        sc = self.sc
        obj = sc.object_propagator()
        det = EclipseDetector(sc.sun, SUN_RADIUS_M, sc.earth).withUmbra()
        det = EclipseDetector.cast_(det.withMaxCheck(ECLIPSE_MAXCHECK_S).withThreshold(THRESHOLD_S).withHandler(ContinueOnEvent()))
        logger = EventsLogger()
        obj.addEventDetector(logger.monitorDetector(det))
        obj.propagate(sc.t0, sc.t1)
        events = [(e.getDate().durationFrom(sc.t0), bool(e.isIncreasing())) for e in logger.getLoggedEvents()]
        g0 = det.g(sc.object_propagator().propagate(sc.t0))
        sunlit0 = self.cyl_sunlit(0.0)  # independent geometric status at t0 fixes the sign convention
        sign = (1 if g0 > 0 else -1) * (1 if sunlit0 else -1)
        self.sun_sign_note = f"g {'>' if sign > 0 else '<'} 0 when sunlit"
        ivs = intervals_from_events(events, sign, sunlit0, self.W)
        # cross-check vs. a cylindrical shadow on a 60 s grid
        ts = np.arange(0.0, self.W, 60.0)
        agree = sum(self.cyl_sunlit(t) == any(a <= t <= b for a, b, *_ in ivs) for t in ts)
        self.sun_agree = (agree, len(ts))
        return ivs

    def cyl_sunlit(self, t):
        d = self.sc.date(t)
        r = vec(self.obj_pv.getPVCoordinates(d, self.sc.teme).getPosition())
        s = vec(self.sc.sun.getPosition(d, self.sc.teme))
        s /= np.linalg.norm(s)
        along = np.dot(r, s)
        return not (along < 0 and np.linalg.norm(r - along * s) < Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

    # -- range: grid minima + Brent on the 1000 km level
    def approaches(self):
        W, L = self.W, RANGE_LIMIT_M
        ts = np.arange(0.0, W + SAMPLE_S / 2, SAMPLE_S)
        ts[-1] = min(ts[-1], W)
        r = np.array([self.rng(t) for t in ts])
        self.grid_t, self.grid_r = ts, r
        out = []
        for i in range(len(ts)):
            left = r[i - 1] if i > 0 else np.inf
            right = r[i + 1] if i < len(ts) - 1 else np.inf
            if not (r[i] <= left and r[i] <= right) or r[i] >= L * 1.2:
                continue
            lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
            res = minimize_scalar(self.rng, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
            tm, rm = (res.x, res.fun) if res.fun < r[i] else (ts[i], r[i])
            if rm >= L:
                continue
            # walk outward on the grid until range > L, then Brent
            j = i
            while j > 0 and r[j] < L:
                j -= 1
            if r[j] < L:  # reached window start
                t_in, clip_s = 0.0, True
            else:
                t_in, clip_s = brentq(lambda t: self.rng(t) - L, ts[j], ts[j + 1], xtol=1e-7), False
            j = i
            while j < len(ts) - 1 and r[j] < L:
                j += 1
            if r[j] < L:
                t_out, clip_e = W, True
            else:
                t_out, clip_e = brentq(lambda t: self.rng(t) - L, ts[j - 1], ts[j], xtol=1e-7), False
            out.append(dict(a=t_in, b=t_out, clip_s=clip_s, clip_e=clip_e, t_min=float(tm), r_min=float(rm)))
        # drop duplicates (same minimum found twice at flat grid ties)
        uniq = []
        for ap in out:
            if not uniq or ap["t_min"] - uniq[-1]["t_min"] > 1.0:
                uniq.append(ap)
        return uniq


def build_results(an):
    sc, W = an.sc, an.W
    cross = an.crossings()
    sun = an.sunlit()
    apps = an.approaches()
    rng_ivs = [(a["a"], a["b"], a["clip_s"], a["clip_e"]) for a in apps]

    crossings = []
    for k, (a, b, cs, ce) in enumerate(cross, 1):
        t_r, r_min = minimize_on(an.rng, a, b)
        t_o, o_min = minimize_on(an.offb, a, b)
        crossings.append(dict(
            id=k, a=a, b=b, dur=b - a, r_min=r_min, t_rmin=t_r, o_min=math.degrees(o_min), t_omin=t_o,
            lat=an.lat_deg(t_r), sunlit_s=overlap_s(a, b, sun), within_s=overlap_s(a, b, rng_ivs),
            clip_s=cs, clip_e=ce))

    visible = []
    for a, b in intersect(cross, rng_ivs, sun):
        k = next(c["id"] for c in crossings if c["a"] - 1e-6 <= a and b <= c["b"] + 1e-6)
        t_r, r_min = minimize_on(an.rng, a, b)
        t_o, o_min = minimize_on(an.offb, a, b)
        visible.append(dict(
            id=len(visible) + 1, crossing_id=k, a=a, b=b, dur=b - a, r_min=r_min, t_rmin=t_r,
            o_min=math.degrees(o_min), t_omin=t_o, lat=an.lat_deg(t_r),
            sunlit_s=overlap_s(a, b, sun), within_s=overlap_s(a, b, rng_ivs),
            clip_s=a <= 1e-9, clip_e=b >= W - 1e-9))

    approaches = []
    for k, ap in enumerate(apps, 1):
        a, b = ap["a"], ap["b"]
        # dense sampling around the closest point so that a short graze of the cone is not skipped
        tm = ap["t_min"]
        cand = sorted(set(list(np.linspace(a, b, 300)) + list(np.clip(tm + np.linspace(-3, 3, 241), a, b))))
        vals = [an.offb(t) for t in cand]
        n = len(cand)
        locmin = [i for i in range(n) if vals[i] <= (vals[i - 1] if i else 9) and vals[i] <= (vals[i + 1] if i < n - 1 else 9)]
        best = (min(vals), cand[int(np.argmin(vals))])
        for i in sorted(locmin, key=lambda i: vals[i])[:3]:
            res = minimize_scalar(an.offb, bounds=(cand[max(i - 1, 0)], cand[min(i + 1, n - 1)]), method="bounded", options={"xatol": 1e-7})
            if res.fun < best[0]:
                best = (res.fun, res.x)
        approaches.append(dict(
            id=k, a=a, b=b, r_min=ap["r_min"], t_min=tm, o_min=math.degrees(best[0]),
            in_fov=math.degrees(best[0]) < FOV_FULL_DEG / 2.0,
            sunlit_at_min=any(c <= tm <= d for c, d, *_ in sun)))

    # Earth occlusion check over all in-FOV time (dense sampling of every crossing)
    blocked = total = 0
    for c in crossings:
        for t in np.linspace(c["a"], c["b"], 400):
            total += 1
            blocked += an.earth_blocked(t)
    sunlit_frac = sum(min(b, W) - max(a, 0.0) for a, b, *_ in sun) / W
    return dict(crossings=crossings, visible=visible, approaches=approaches, sun=sun,
                blocked=blocked, blocked_total=total, sunlit_frac=sunlit_frac)


# ----------------------------------------------------------------------------- output
CROSS_COLS = ["event_id", "start_utc", "end_utc", "duration_s", "min_range_km", "min_range_utc",
              "min_off_boresight_deg", "min_off_boresight_utc", "tracker_lat_at_min_range_deg",
              "sunlit_s", "within_range_s", "clipped_at_start", "clipped_at_end"]
APPR_COLS = ["approach_id", "enter_1000km_utc", "exit_1000km_utc", "min_range_km", "min_range_utc",
             "min_off_boresight_deg", "in_fov", "object_sunlit_at_min_range"]


def event_row(sc, e, with_crossing=False):
    row = [e["id"], sc.fmt(e["a"]), sc.fmt(e["b"]), f'{e["dur"]:.3f}', f'{e["r_min"]/1e3:.3f}',
           sc.fmt(e["t_rmin"]), f'{e["o_min"]:.3f}', sc.fmt(e["t_omin"]), f'{e["lat"]:.3f}',
           f'{e["sunlit_s"]:.3f}', f'{e["within_s"]:.3f}', bool(e["clip_s"]), bool(e["clip_e"])]
    return row + ([e["crossing_id"]] if with_crossing else [])


def appr_row(sc, a):
    return [a["id"], sc.fmt(a["a"]), sc.fmt(a["b"]), f'{a["r_min"]/1e3:.3f}', sc.fmt(a["t_min"]),
            f'{a["o_min"]:.3f}', bool(a["in_fov"]), bool(a["sunlit_at_min"])]


def write_csvs(sc, res, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, cols, rows in (
        ("crossings.csv", CROSS_COLS, [event_row(sc, e) for e in res["crossings"]]),
        ("visible_events.csv", CROSS_COLS + ["crossing_id"], [event_row(sc, e, True) for e in res["visible"]]),
        ("close_approaches.csv", APPR_COLS, [appr_row(sc, a) for a in res["approaches"]]),
    ):
        with open(os.path.join(outdir, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)


def flag(e):
    return ("S" if e["clip_s"] else "") + ("E" if e["clip_e"] else "") or "-"


def print_report(sc, an, res):
    W = an.W
    print("=" * 112)
    print("SPACE-BASED TRACKER - Orekit analysis")
    print(f"Window : {sc.fmt(0)} -> {sc.fmt(W)} UTC ({W/3600:.1f} h)")
    print(f"Sensor : circular cone, {FOV_FULL_DEG:.0f} deg full angle ({FOV_FULL_DEG/2:.0f} deg half), boresight = inertial velocity (body +X)")
    print(f"Visible: object sunlit (umbra only) AND range < {RANGE_LIMIT_M/1e3:.0f} km AND inside FOV")
    mc = (f"adaptive maxCheck {MAXCHECK_FLOOR_S}..{MAXCHECK_FAR_S} s (factor {MAXCHECK_FACTOR}), {an.interval.calls} interval callbacks"
          if an.fixed_maxcheck is None else f"FIXED maxCheck {an.fixed_maxcheck} s")
    print(f"Detectors: FOV {mc}, threshold {THRESHOLD_S*1e3:.0f} ms; tracker run + FOV detection {an.run_time:.1f} s")
    print(f"Conventions: FOV {an.fov_sign_note}; Eclipse {an.sun_sign_note}")
    print("=" * 112)
    rmin = min(res["approaches"], key=lambda a: a["r_min"])
    print(f"SANITY  tracker alt (geocentric) {an.alt['trk'][0]:.1f}-{an.alt['trk'][1]:.1f} km | object alt {an.alt['obj'][0]:.1f}-{an.alt['obj'][1]:.1f} km | "
          f"min range of day {rmin['r_min']/1e3:.2f} km at {sc.fmt(rmin['t_min'])} | "
          f"Earth-blocked in-FOV samples {res['blocked']}/{res['blocked_total']} | "
          f"sunlit-vs-cylinder agreement {an.sun_agree[0]}/{an.sun_agree[1]}")

    def table(title, evs):
        print(f"\n{title}: {len(evs)}")
        if not evs:
            return
        print(f"{'#':>2} {'start UTC':23} {'end UTC':23} {'dur s':>7} {'min rng km':>10} {'at UTC':23} {'lat deg':>8} {'sunlit s':>8} {'<1000km s':>9} {'clip':>4}")
        for e in evs:
            print(f"{e['id']:>2} {sc.fmt(e['a']):23} {sc.fmt(e['b']):23} {e['dur']:7.3f} {e['r_min']/1e3:10.2f} {sc.fmt(e['t_rmin']):23} "
                  f"{e['lat']:8.2f} {e['sunlit_s']:8.3f} {e['within_s']:9.3f} {flag(e):>4}")

    table("CROSSINGS (object geometrically inside the FOV)", res["crossings"])
    print(f"\nVISIBLE EVENTS (in FOV AND sunlit AND range < {RANGE_LIMIT_M/1e3:.0f} km): {len(res['visible'])}")
    if res["visible"]:
        print(f"{'#':>2} {'start UTC':23} {'end UTC':23} {'dur s':>7} {'min rng km':>10} {'at UTC':23} {'crossing':>8}")
        for e in res["visible"]:
            print(f"{e['id']:>2} {sc.fmt(e['a']):23} {sc.fmt(e['b']):23} {e['dur']:7.3f} {e['r_min']/1e3:10.2f} {sc.fmt(e['t_rmin']):23} {e['crossing_id']:>8}")
    print("\nSUMMARY")
    print(f"  object sunlit {100*res['sunlit_frac']:.1f}% of the window")
    print(f"  approaches < {RANGE_LIMIT_M/1e3:.0f} km: {len(res['approaches'])}")
    print(f"  crossings: {len(res['crossings'])}   visible events: {len(res['visible'])}")
    print("  closest 3 approaches: " + "; ".join(f"{sc.fmt(a['t_min'])[11:]} {a['r_min']/1e3:.2f} km (min off-bore {a['o_min']:.1f} deg)"
                                               for a in sorted(res['approaches'], key=lambda a: a['r_min'])[:3]))


def altitude_ranges(an):
    sc, out = an.sc, {}
    ts = np.arange(0.0, an.W + 1, 30.0)
    for key, pv in (("trk", an.trk_pv), ("obj", an.obj_pv)):
        r = [pv.getPVCoordinates(sc.date(t), sc.teme).getPosition().getNorm() for t in ts]
        out[key] = (min(r) / 1e3 - Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1e3, max(r) / 1e3 - Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1e3)
    return out


# ----------------------------------------------------------------------------- plot
def make_plot(sc, an, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = set(np.arange(0.0, an.W + 0.1, 5.0))
    for a in res["approaches"]:
        ts.update(np.clip(a["t_min"] + np.linspace(-2, 2, 401), 0, an.W))
    ts = np.array(sorted(ts))
    r = np.array([an.rng(t) for t in ts]) / 1e3
    o = np.array([math.degrees(an.offb(t)) for t in ts])
    hrs = ts / 3600.0
    lit = np.array([any(a <= t <= b for a, b, *_ in res["sun"]) for t in ts], dtype=float)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].semilogy(hrs, r, lw=0.8, color="#1f4e79")
    ax[0].axhline(RANGE_LIMIT_M / 1e3, color="#888", ls="--", lw=0.8)
    ax[0].set_ylabel("range [km]")
    ax[1].plot(hrs, o, lw=0.8, color="#1f4e79")
    ax[1].axhline(FOV_FULL_DEG / 2, color="#888", ls="--", lw=0.8)
    ax[1].set_ylabel("off-boresight [deg]")
    ax[1].set_ylim(0, 180)
    ax[2].fill_between(hrs, lit, step="post", color="#e0a500", alpha=0.6)
    ax[2].set_ylabel("object sunlit")
    ax[2].set_yticks([0, 1])
    ax[2].set_xlabel(f"hours since {sc.fmt(0)} UTC")
    for k, e in enumerate(res["crossings"]):
        for a in ax:
            a.axvspan(e["a"] / 3600, max(e["b"], e["a"] + 180) / 3600, color="#c0392b", alpha=0.25, label="crossing (shading widened to >= 3 min)" if k == 0 and a is ax[0] else None)
    for k, e in enumerate(res["visible"]):
        for a in ax:
            a.axvspan(e["a"] / 3600, max(e["b"], e["a"] + 90) / 3600, color="#2e8b57", alpha=0.5, label="visible event (shading widened to >= 1.5 min)" if k == 0 and a is ax[0] else None)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_xlim(0, an.W / 3600)
    ax[0].set_title("Tracker -> object: range, off-boresight angle, illumination")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- compare
REF_FALLBACK = [  # from the assignment validation table
    ("2025-09-01 00:29:34.299", "2025-09-01 00:29:41.986", 7.69, 84.3, 0.08),
    ("2025-09-01 02:04:28.512", "2025-09-01 02:04:28.691", 0.18, 2.7, 10.1),
]


def parse_utc(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def compare(sc, res, ref_dir):
    print("\nCOMPARISON vs reference")
    path = os.path.join(ref_dir, "crossings.csv")
    if os.path.exists(path):
        with open(path) as f:
            ref = [(r["start_utc"], r["end_utc"], float(r["duration_s"]), float(r["min_range_km"]), float(r["min_off_boresight_deg"])) for r in csv.DictReader(f)]
        print(f"  (reference CSVs read from {ref_dir}/)")
    else:
        ref = REF_FALLBACK
        print(f"  {path} not found -> using the validation table from the assignment")
    print(f"  {'#':>2} {'':3} {'start UTC':23} {'end UTC':23} {'dur s':>7} {'min rng km':>10} {'min offb deg':>12}   d_start s  d_end s")
    for i in range(max(len(ref), len(res["crossings"]))):
        r = ref[i] if i < len(ref) else None
        o = res["crossings"][i] if i < len(res["crossings"]) else None
        if r:
            print(f"  {i+1:>2} ref {r[0]:23} {r[1]:23} {r[2]:7.3f} {r[3]:10.2f} {r[4]:12.2f}")
        if o:
            line = f"  {i+1:>2} ore {sc.fmt(o['a']):23} {sc.fmt(o['b']):23} {o['dur']:7.3f} {o['r_min']/1e3:10.2f} {o['o_min']:12.2f}"
            if r:
                line += f"   {(parse_utc(sc.fmt(o['a'])) - parse_utc(r[0])).total_seconds():+8.3f} {(parse_utc(sc.fmt(o['b'])) - parse_utc(r[1])).total_seconds():+8.3f}"
            print(line)
    for name in ("visible_events.csv", "close_approaches.csv"):
        p = os.path.join(ref_dir, name)
        if not os.path.exists(p):
            print(f"  {p} not found - skipped")
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if name == "visible_events.csv":
            print(f"  visible events: ref {len(rows)} vs orekit {len(res['visible'])}")
            for r, o in zip(rows, res["visible"]):
                print(f"    start {r['start_utc']} vs {sc.fmt(o['a'])}   end {r['end_utc']} vs {sc.fmt(o['b'])}")
        else:
            print(f"  approaches < 1000 km: ref {len(rows)} vs orekit {len(res['approaches'])}")
            diffs = []
            for r, o in zip(rows, res["approaches"]):
                diffs.append(abs(float(r["min_range_km"]) - o["r_min"] / 1e3))
                dtm = (parse_utc(sc.fmt(o["t_min"])) - parse_utc(r["min_range_utc"])).total_seconds()
                diffs[-1] = (diffs[-1], dtm)
            if diffs:
                print(f"    max |d min_range| = {max(d[0] for d in diffs):.3f} km, max |d t_min| = {max(abs(d[1]) for d in diffs):.3f} s")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="results_orekit/", help="directory for the CSVs")
    ap.add_argument("--plot", action="store_true", help="write a 3-panel PNG")
    ap.add_argument("--compare", action="store_true", help="compare with the reference CSVs")
    ap.add_argument("--ref", default="results/", help="directory with the reference CSVs (for --compare)")
    ap.add_argument("--fixed-maxcheck", type=float, default=None, metavar="S",
                    help="use a constant FOV maxCheck (s) instead of the adaptive one (demonstrates missed events)")
    args = ap.parse_args()

    load_orekit_data(OREKIT_DATA)
    sc = Scenario()
    an = Analysis(sc, args.fixed_maxcheck)
    an.alt = altitude_ranges(an)
    res = build_results(an)
    print_report(sc, an, res)
    write_csvs(sc, res, args.outdir)
    print(f"\nCSVs written to {args.outdir}")
    if args.plot:
        path = os.path.join(args.outdir, "space_tracker_orekit.png")
        make_plot(sc, an, res, path)
        print(f"plot written to {path}")
    if args.compare:
        compare(sc, res, args.ref)


if __name__ == "__main__":
    main()
