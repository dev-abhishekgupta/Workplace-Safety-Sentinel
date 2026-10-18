#!/usr/bin/env python3
"""
zone_monitor.py - Restricted-zone entry detection + JSON incident logging (demo).

This is a standalone demonstration of the safety logic that the PipeGen pipeline
is designed to feed. It does NOT run inference; it consumes detection records
(bounding boxes) and emits deduplicated incident records as JSON Lines.

Usage:
  python3 zone_monitor.py                       # runs built-in demo detections
  python3 zone_monitor.py --input dets.json     # reads detections from a JSON file
  python3 zone_monitor.py --zone 100,100,400,300 --cooldown 5

Input JSON format (list of frames):
[
  {"frame": 1, "objects": [{"id": 1, "bbox": [x1,y1,x2,y2], "conf": 0.91}]},
  ...
]

Output: incidents.jsonl (one JSON object per line) + console summary.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

# ---------- Default restricted zone (x1, y1, x2, y2) ----------
DEFAULT_ZONE = (200, 150, 500, 400)  # rectangle in pixels
COOLDOWN_SEC = 3.0  # min seconds between alerts for the same track id


def bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def in_zone(point, zone):
    x, y = point
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= x <= zx2 and zy1 <= y <= zy2


def make_incident(track_id, conf, zone_id, camera_id, counter):
    return {
        "incident_id": f"INC-{counter:04d}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_type": "restricted_zone_entry",
        "camera_id": camera_id,
        "track_id": track_id,
        "zone_id": zone_id,
        "confidence": round(float(conf), 3),
        "reference_point": "bottom_center",
    }


def process(frames, zone, zone_id, camera_id, cooldown, out_path):
    inside = {}          # track_id -> last_alert_time
    incidents = []
    counter = 0
    for f in frames:
        frame_no = f.get("frame", 0)
        for obj in f.get("objects", []):
            tid = obj.get("id")
            bbox = obj["bbox"]
            conf = obj.get("conf", 0.0)
            pt = bottom_center(bbox)
            now_in = in_zone(pt, zone)
            last = inside.get(tid, 0.0)
            if now_in:
                if tid not in inside or (time.time() - last) >= cooldown:
                    counter += 1
                    inc = make_incident(tid, conf, zone_id, camera_id, counter)
                    inc["frame"] = frame_no
                    incidents.append(inc)
                    inside[tid] = time.time()
                    print(f"[ALERT] frame={frame_no} track={tid} entered zone "
                          f"(conf={conf:.2f}) -> {inc['incident_id']}")
            else:
                if tid in inside:
                    print(f"[EXIT]  frame={frame_no} track={tid} left zone")
                    del inside[tid]
    # write JSON Lines
    with open(out_path, "w", encoding="utf-8") as fh:
        for inc in incidents:
            fh.write(json.dumps(inc) + "\n")
    print(f"\nTotal incidents: {len(incidents)}  ->  {out_path}")
    return incidents


def demo_frames():
    """Built-in demo: track 1 enters zone, stays, exits, re-enters."""
    return [
        {"frame": 1,  "objects": [{"id": 1, "bbox": [50, 50, 120, 180], "conf": 0.92}]},
        {"frame": 2,  "objects": [{"id": 1, "bbox": [180, 120, 260, 260], "conf": 0.90}]},  # enters
        {"frame": 3,  "objects": [{"id": 1, "bbox": [220, 140, 300, 280], "conf": 0.91}]},  # stays (no new alert)
        {"frame": 4,  "objects": [{"id": 1, "bbox": [230, 150, 310, 290], "conf": 0.89}]},  # stays (no new alert)
        {"frame": 5,  "objects": [{"id": 1, "bbox": [50, 50, 120, 180], "conf": 0.88}]},   # exits
        {"frame": 6,  "objects": [{"id": 1, "bbox": [210, 130, 290, 270], "conf": 0.93}]},  # re-enters (new alert)
        {"frame": 7,  "objects": [{"id": 2, "bbox": [250, 160, 330, 300], "conf": 0.87}]},  # second person enters
    ]


def main():
    ap = argparse.ArgumentParser(description="Restricted-zone entry monitor (demo)")
    ap.add_argument("--input", help="JSON file with detection frames")
    ap.add_argument("--zone", default=None,
                    help="zone as x1,y1,x2,y2 (default: 200,150,500,400)")
    ap.add_argument("--zone-id", default="restricted_zone_01")
    ap.add_argument("--camera-id", default="camera_01")
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_SEC)
    ap.add_argument("--output", default="incidents.jsonl")
    args = ap.parse_args()

    zone = DEFAULT_ZONE
    if args.zone:
        zone = tuple(int(v) for v in args.zone.split(","))
    assert len(zone) == 4

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            frames = json.load(fh)
    else:
        frames = demo_frames()

    print(f"Zone: {zone}  cooldown: {args.cooldown}s  output: {args.output}")
    process(frames, zone, args.zone_id, args.camera_id, args.cooldown, args.output)


if __name__ == "__main__":
    main()
