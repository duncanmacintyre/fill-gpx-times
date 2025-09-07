#!/usr/bin/env python3
"""
fill-gpx-times.py — Fill in missing GPX timestamps by interpolation.

This script reads a GPX file and fills in missing timestamps for track points by
interpolating between the nearest known timestamps. Interpolation is done based on
the distances along the track (distance-weighted interpolation).

If the script encounters non-monotonic times (a point with a time not strictly later
than the previous one), it discards the bad time and instead uses the next strictly
later good time as the future anchor for interpolation.

The script ensures that all assigned times are strictly increasing. If needed, it will
add microsecond adjustments to break ties.

Dependencies:
    - Python 3.8+
    - gpxpy

Installation:
    pip install gpxpy

    or with conda:
    conda install -c conda-forge gpxpy
"""

import argparse
from datetime import datetime, timedelta
import math
import sys
from typing import List, Optional, Tuple

import gpxpy
import gpxpy.gpx


# ------------------------------ Utilities ------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in meters."""
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def elevation_gain_loss_m(points: List[gpxpy.gpx.GPXTrackPoint]) -> Tuple[float, float]:
    gain = 0.0
    loss = 0.0
    prev_elev = None
    for p in points:
        if p.elevation is None:
            prev_elev = p.elevation
            continue
        if prev_elev is not None:
            de = p.elevation - prev_elev
            if de > 0:
                gain += de
            elif de < 0:
                loss += -de
        prev_elev = p.elevation
    return gain, loss


def point_time(p: gpxpy.gpx.GPXTrackPoint) -> Optional[datetime]:
    return p.time if isinstance(p.time, datetime) else None


# ------------------------------ Core logic ------------------------------

def find_good_anchors(points: List[gpxpy.gpx.GPXTrackPoint]) -> List[int]:
    """
    Return indices of "good" anchor points with strictly increasing times.
    Any point whose time is None or not strictly greater than the last good time is excluded.
    """
    good_idxs: List[int] = []
    last_good_time: Optional[datetime] = None
    for i, p in enumerate(points):
        t = point_time(p)
        if t is None:
            continue
        if last_good_time is None or t > last_good_time:
            good_idxs.append(i)
            last_good_time = t
        else:
            # Non-monotonic -> treat as bad (we will overwrite later)
            pass
    return good_idxs


def cumulative_distances(points: List[gpxpy.gpx.GPXTrackPoint], start_idx: int, end_idx: int) -> List[float]:
    """
    Compute cumulative distances (meters) from start_idx across points up to end_idx.
    Returns a list of length (end_idx - start_idx + 1) where entry 0 is 0.0 at start_idx.
    """
    cum = [0.0]
    for i in range(start_idx + 1, end_idx + 1):
        a = points[i - 1]
        b = points[i]
        da = haversine_m(a.latitude, a.longitude, b.latitude, b.longitude)
        cum.append(cum[-1] + da)
    return cum


def interpolate_between(points: List[gpxpy.gpx.GPXTrackPoint],
                        left_idx: int,
                        right_idx: int,
                        counters: dict):
    """
    Distance-weighted interpolation for points strictly between left_idx and right_idx.
    - left and right must both be "good" anchors with strictly increasing times.
    - Any interior point (including ones that had "bad" times) gets reassigned.
    """
    n = right_idx - left_idx
    if n <= 1:
        return  # nothing to do

    t0 = point_time(points[left_idx])
    t1 = point_time(points[right_idx])
    assert t0 is not None and t1 is not None and t1 > t0

    # Distances along the path
    cum = cumulative_distances(points, left_idx, right_idx)
    total = cum[-1]

    if total <= 0.0:
        # Fallback to even spacing
        dt = (t1 - t0).total_seconds()
        for k in range(1, n):
            frac = k / n
            new_t = t0 + timedelta(seconds=dt * frac)
            existing = point_time(points[left_idx + k])
            # We are assigning a new time, so count it
            counters["filled"] += 1
            if existing is not None:
                counters["reassigned_existing"] += 1
                # classify as "bad" if it falls outside (t0, t1)
                if existing <= t0 or existing >= t1:
                    counters["overwritten_bad"] += 1
            points[left_idx + k].time = new_t
        return

    # Distance-weighted interpolation
    dt = (t1 - t0).total_seconds()
    for k in range(1, n):
        frac = cum[k] / total
        new_t = t0 + timedelta(seconds=dt * frac)
        existing = point_time(points[left_idx + k])
        if existing is None:
            counters['filled'] += 1
        elif existing <= t0 or existing >= t1:
            counters['overwritten_bad'] += 1
            counters['filled'] += 1
        else:
            counters['filled'] += 1  # reassigned monotonic case
        points[left_idx + k].time = new_t


def fix_segment(points: List[gpxpy.gpx.GPXTrackPoint], verbose: bool, counters: dict):
    """
    For a single GPX segment:
    - Identify good anchors with strictly increasing times.
    - For each consecutive pair of good anchors, interpolate (distance-weighted) interior points.
    - Do not extrapolate before first good or after last good.
    """
    if not points:
        return

    # Count initial states
    for p in points:
        if point_time(p) is not None:
            counters["initial_with_time"] += 1
        else:
            counters["initial_missing"] += 1

    good = find_good_anchors(points)

    if len(good) == 0:
        if verbose:
            print("  [warn] Segment has no timestamps; leaving unchanged.")
        return

    # If there is only one good anchor, we cannot interpolate on either side.
    # But we can still sanitize any later times that are non-increasing by leaving them alone
    # (since we lack a strictly later anchor). We will count them as preserved if we don't touch.
    if len(good) == 1 and verbose:
        print("  [warn] Segment has only one strictly-later anchor; no interpolation possible.")

    # Interpolate within each pair of consecutive good anchors
    for a, b in zip(good, good[1:]):
        t0 = point_time(points[a])
        t1 = point_time(points[b])
        if t0 is None or t1 is None or not (t1 > t0):
            # Shouldn't happen because we constructed "good" that way; skip defensively
            if verbose:
                print(f"  [warn] Skipping pair ({a},{b}) due to non-positive duration.")
            continue
        interpolate_between(points, a, b, counters)

    # Final pass: ensure strict monotonicity (microsecond bumps if needed)
    for i in range(1, len(points)):
        if point_time(points[i]) is not None and point_time(points[i-1]) is not None:
            if points[i].time <= points[i-1].time:
                points[i].time = points[i-1].time + timedelta(microseconds=1)
                counters["micro_bumped"] += 1


def process_gpx(gpx: gpxpy.gpx.GPX, verbose: bool) -> dict:
    counters = {
        "initial_with_time": 0,
        "initial_missing": 0,
        "filled": 0,              # assigned new timestamps to previously missing/bad
        "overwritten_bad": 0,     # had a timestamp but it was non-monotonic; we overwrote it
        "reassigned_existing": 0,  # reassigned a point that originally had a time
        "micro_bumped": 0,        # tiny adjustments to guarantee strict monotonicity
        "preserved": 0,
        "missing": 0,
        "distance_m": 0.0,
        "elev_gain_m": 0.0,
        "elev_loss_m": 0.0,
        "elapsed": None,
        "avg_pace": None,
    }

    # Work through all segments
    for trk in gpx.tracks:
        for seg in trk.segments:
            fix_segment(seg.points, verbose, counters)

            # Distance and elevation stats
            # Compute distance across the segment (after potential fixes)
            seg_dist = 0.0
            for i in range(1, len(seg.points)):
                a, b = seg.points[i-1], seg.points[i]
                seg_dist += haversine_m(a.latitude, a.longitude, b.latitude, b.longitude)
            counters["distance_m"] += seg_dist

            gain, loss = elevation_gain_loss_m(seg.points)
            counters["elev_gain_m"] += gain
            counters["elev_loss_m"] += loss

    # Summary counts (after changes)
    total_points = 0
    with_time_after = 0
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                total_points += 1
                if point_time(p) is not None:
                    with_time_after += 1

    counters["missing"] = total_points - with_time_after

    # Preserved = points that had time initially and we did not overwrite or micro-bump past it.
    # Approximate preserved as initial_with_time - overwritten_bad - (micro bumps that affected those).
    # (We can't perfectly recover which ones were micro-bumped originally-with-time; use conservative estimate.)
    counters["preserved"] = max(
        0, counters["initial_with_time"] - counters["reassigned_existing"]
    )

    # Elapsed/pace if possible (use earliest and latest timestamps found)
    all_times = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if point_time(p) is not None:
                    all_times.append(p.time)
    if all_times:
        start_t = min(all_times)
        end_t = max(all_times)
        if end_t > start_t:
            elapsed = end_t - start_t
            counters["elapsed"] = elapsed
            # pace in min/km if distance available
            if counters["distance_m"] > 0:
                secs = elapsed.total_seconds()
                pace = secs / (counters["distance_m"] / 1000.0)  # sec per km
                minutes = int(pace // 60)
                seconds = int(round(pace % 60))
                counters["avg_pace"] = f"{minutes}:{seconds:02d} min/km"

    return counters


# ------------------------------ CLI ------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fill in missing GPX timestamps by interpolation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("input", help="Input GPX file")
    p.add_argument("-o", "--output", help="Output GPX file (default: <input>.fixed.gpx)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_path = args.output or args.input.replace(".gpx", ".fixed.gpx")

    with open(args.input, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    counters = process_gpx(gpx, verbose=args.verbose)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

    # Summary
    print("=== Summary ===")
    print(f"Output file: {out_path}")
    print(f"Points with initial time preserved: {counters['preserved']}")
    print(f"Points assigned new times: {counters['filled']} "
          f"(including {counters['overwritten_bad']} overwritten non-monotonic times)")
    print(f"Points still missing times: {counters['missing']}")
    if counters["elapsed"]:
        print(f"Total elapsed time: {counters['elapsed']}")
    print(f"Total distance: {counters['distance_m'] / 1000.0:.2f} km")
    print(f"Elevation gain: {counters['elev_gain_m']:.1f} m")
    print(f"Elevation loss: {counters['elev_loss_m']:.1f} m")
    if counters["avg_pace"]:
        print(f"Average pace: {counters['avg_pace']}")
    if counters["micro_bumped"]:
        print(f"Microsecond bumps to enforce monotonicity: {counters['micro_bumped']}")


if __name__ == "__main__":
    sys.exit(main())
