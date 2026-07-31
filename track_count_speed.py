"""Project 2 — Traffic Lens: detect + track + count vehicles (+ speed if calibrated).

Pipeline: YOLO detections -> ByteTrack IDs -> line-crossing counts -> annotated video.
Speed needs real-world calibration (--calib); see README and LEARNINGS.md.

The core logic lives in `process_video()` so this CLI and `app.py` share one
implementation. `run_all_colab.ipynb` exercises the same pipeline on a GPU.

Demo on the bundled highway video:
    python track_count_speed.py --max-frames 200 --calib calib.json

Your own footage:
    python track_count_speed.py --source my_dashcam.mp4 --calib calib.json --imgsz 1280
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

OUT = Path(__file__).parent / "outputs"

# Resolve vehicle classes BY NAME, never by hardcoded index. Different checkpoints use
# different label spaces -- COCO index 2 is `car`, but in the VisDrone fine-tune index 2
# is `bicycle`. Hardcoding indices silently detects the wrong categories when the model
# changes; that exact class-index assumption is what invalidated run 1's baseline
# (see LEARNINGS.md #1).
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "motor", "van", "bicycle"}


def vehicle_class_ids(model) -> list[int]:
    """Indices in *this* model's label space that count as vehicles."""
    ids = [i for i, name in model.names.items() if str(name).lower() in VEHICLE_NAMES]
    if not ids:
        raise ValueError(f"no vehicle classes found in model labels: {model.names}")
    return ids


def load_model(name: str) -> YOLO:
    """Prefer YOLO26 (NMS-free, Jan 2026); fall back if this ultralytics lacks it."""
    try:
        return YOLO(name)
    except Exception as e:
        fallback = "yolo11n.pt"
        print(f"could not load {name} ({e}); falling back to {fallback}")
        return YOLO(fallback)


class ViewTransformer:
    """Homography: image pixels -> road-plane meters. This is the 'speed' trick:
    pixel distances are distorted by perspective, metric distances are not."""

    def __init__(self, source_px, target_m):
        self.m, _ = cv2.findHomography(np.array(source_px, dtype=np.float32),
                                       np.array(target_m, dtype=np.float32))

    def transform(self, points):
        if len(points) == 0:
            return points
        pts = np.asarray(points).reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


def process_video(
    source: str,
    output_path: str,
    model: YOLO = None,
    model_name: str = "yolo26n.pt",
    conf: float = 0.3,
    max_frames: int = 0,
    calib: dict | None = None,
    line_y_frac: float = 0.5,
    imgsz: int = 640,
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    progress_every: int = 50,
    verbose: bool = True,
) -> dict:
    """Run detect -> track -> count (+speed) on one video.

    `calib` is {"source_px": [[x,y] x4], "target_m": [[x,y] x4]} mapping a road
    quadrilateral to real-world meters. It must be *measured*, not eyeballed -- distance
    error propagates linearly into speed (LEARNINGS.md #2).

    `imgsz` matters on high-res footage: 4K downscaled to the 640 default turns distant
    vehicles into a few pixels and they go undetected (LEARNINGS.md #3).

    Returns stats incl. detections/frame and mean track length, which together
    distinguish tracker fragmentation from genuinely-more-detections (LEARNINGS.md #4).
    """
    if model is None:
        model = load_model(model_name)
    vehicle_ids = vehicle_class_ids(model)

    info = sv.VideoInfo.from_video_path(source)
    fps = info.fps

    tracker = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
    )
    line_y = int(info.height * line_y_frac)
    line = sv.LineZone(start=sv.Point(0, line_y), end=sv.Point(info.width, line_y))

    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=1.0)
    trace_ann = sv.TraceAnnotator(trace_length=int(fps))
    line_ann = sv.LineZoneAnnotator(text_scale=1.5)

    view = road_w = road_h = None
    if calib:
        view = ViewTransformer(calib["source_px"], calib["target_m"])
        tm = np.array(calib["target_m"], dtype=float)
        road_w, road_h = tm[:, 0].max(), tm[:, 1].max()

    history = defaultdict(lambda: deque(maxlen=int(fps)))
    id_frames = defaultdict(int)
    total_dets = n_frames = 0
    speeds_seen = []

    with sv.VideoSink(output_path, video_info=info) as sink:
        for idx, frame in enumerate(sv.get_video_frames_generator(source)):
            if max_frames and idx >= max_frames:
                break
            n_frames += 1

            result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            det = sv.Detections.from_ultralytics(result)
            det = det[np.isin(det.class_id, vehicle_ids)]
            det = tracker.update_with_detections(det)
            line.trigger(det)

            total_dets += len(det)
            for tid in det.tracker_id:
                id_frames[tid] += 1

            labels = []
            anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            for tid, anchor in zip(det.tracker_id, anchors):
                speed_txt = ""
                if view is not None:
                    x_m, y_m = view.transform(anchor.reshape(1, 2))[0]
                    # ignore points outside the calibrated road region -- the homography
                    # extrapolates nonsense beyond the quad it was fitted on
                    if -5 <= x_m <= road_w + 5 and 0 <= y_m <= road_h:
                        history[tid].append((y_m, idx))
                        if len(history[tid]) >= fps // 2:
                            (y0, f0), (y1, f1) = history[tid][0], history[tid][-1]
                            kmh = abs(y1 - y0) / ((f1 - f0) / fps) * 3.6
                            speed_txt = f" {kmh:.0f} km/h"
                            speeds_seen.append(kmh)
                labels.append(f"#{tid}{speed_txt}")

            frame = trace_ann.annotate(frame, det)
            frame = box_ann.annotate(frame, det)
            frame = label_ann.annotate(frame, det, labels=labels)
            frame = line_ann.annotate(frame, line_counter=line)
            sink.write_frame(frame)

            if verbose and idx % progress_every == 0:
                print(f"frame {idx}: {len(det)} vehicles tracked  "
                      f"(in={line.in_count} out={line.out_count})")

    track_lengths = list(id_frames.values())
    return {
        "in_count": line.in_count,
        "out_count": line.out_count,
        "total_unique_ids": len(id_frames),
        "detections_per_frame": round(total_dets / max(n_frames, 1), 2),
        "mean_track_length_frames": round(float(np.mean(track_lengths)), 1) if track_lengths else 0,
        "median_speed_kmh": round(float(np.median(speeds_seen)), 1) if speeds_seen else None,
        "output_path": output_path,
        "has_speed": view is not None,
        "imgsz": imgsz,
        "vehicle_classes": [model.names[i] for i in vehicle_ids],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video path (default: bundled demo video)")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; use 1280+ for 4K footage with distant vehicles")
    ap.add_argument("--calib", default=None,
                    help="JSON with source_px (4 image points) + target_m (same 4 in meters) "
                         "-> enables speed estimation")
    ap.add_argument("--line-y-frac", type=float, default=0.5,
                    help="counting line position as a fraction of frame height")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    source = args.source
    if source is None:
        from supervision.assets import VideoAssets, download_assets
        source = download_assets(VideoAssets.VEHICLES)
        print(f"using demo video: {source}")

    calib = json.loads(Path(args.calib).read_text()) if args.calib else None
    stats = process_video(
        source=source,
        output_path=str(OUT / "annotated.mp4"),
        model_name=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        calib=calib,
        line_y_frac=args.line_y_frac,
    )

    print(f"\ncounts: in={stats['in_count']} out={stats['out_count']}")
    print(f"detections/frame: {stats['detections_per_frame']}  "
          f"mean track length: {stats['mean_track_length_frames']} frames")
    if stats["median_speed_kmh"] is not None:
        print(f"median speed: {stats['median_speed_kmh']} km/h  "
              "(sanity-check against the road type!)")
    print(f"annotated video -> {stats['output_path']}")
    if not stats["has_speed"]:
        print("note: no --calib given, so no speeds. See README 'Calibrating for speed'.")


if __name__ == "__main__":
    main()
