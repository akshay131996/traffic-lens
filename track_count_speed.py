"""Project 2 — Traffic Lens: detect + track + count vehicles (+ speed if calibrated).

Pipeline: YOLO detections -> ByteTrack IDs -> line-crossing counts -> annotated video.
Speed needs real-world calibration (--calib), explained in the README/notebook.

The core logic lives in `process_video()` so both this CLI and `app.py` share one
implementation — see `run_all_colab.ipynb` for the calibrated run, the ByteTrack
parameter sweep, and everything else exercising this function on real data.

Demo on a bundled highway video (auto-downloads, ~100 frames on CPU ≈ 2 min):
    python track_count_speed.py --max-frames 100

Your own footage:
    python track_count_speed.py --source my_dashcam.mp4 --calib calib.json
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
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck


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
    pixel distances are meaningless (perspective), metric distances are not."""

    def __init__(self, source_px: np.ndarray, target_m: np.ndarray):
        self.m, _ = cv2.findHomography(source_px.astype(np.float32),
                                       target_m.astype(np.float32))

    def transform(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
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
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    progress_every: int = 25,
    verbose: bool = True,
) -> dict:
    """Run the detect->track->count(+speed) pipeline on one video.

    `calib`, if given, is {"source_px": [[x,y],...], "target_m": [[x,y],...]} (4+ points)
    -- the same format as calib_demo.json. `line_y_frac` places the counting line at that
    fraction of the frame height (0.5 = middle); pass whatever matches your camera angle.
    `track_activation_threshold`/`lost_track_buffer` are ByteTrack's tunables, exposed so
    a caller can sweep them (see the notebook's ID-switch experiment).

    Returns a stats dict: in_count, out_count, total_unique_ids, output_path.
    """
    if model is None:
        model = load_model(model_name)

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
    label_ann = sv.LabelAnnotator(text_scale=0.5)
    trace_ann = sv.TraceAnnotator(trace_length=int(fps))
    line_ann = sv.LineZoneAnnotator(text_scale=0.6)

    view = None
    if calib:
        view = ViewTransformer(np.array(calib["source_px"]), np.array(calib["target_m"]))

    history = defaultdict(lambda: deque(maxlen=int(fps)))  # tracker_id -> recent (y_m, frame)
    seen_ids = set()

    frames = sv.get_video_frames_generator(source)
    with sv.VideoSink(output_path, video_info=info) as sink:
        for idx, frame in enumerate(frames):
            if max_frames and idx >= max_frames:
                break

            result = model(frame, conf=conf, verbose=False)[0]
            det = sv.Detections.from_ultralytics(result)
            det = det[np.isin(det.class_id, VEHICLE_CLASSES)]
            det = tracker.update_with_detections(det)
            line.trigger(det)
            seen_ids.update(det.tracker_id.tolist())

            labels = []
            anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            for tid, anchor in zip(det.tracker_id, anchors):
                speed_txt = ""
                if view is not None:
                    y_m = view.transform(anchor.reshape(1, 2))[0][1]
                    history[tid].append((y_m, idx))
                    if len(history[tid]) >= fps // 2:
                        (y0, f0), (y1, f1) = history[tid][0], history[tid][-1]
                        mps = abs(y1 - y0) / ((f1 - f0) / fps)
                        speed_txt = f" {mps * 3.6:.0f} km/h"
                labels.append(f"#{tid}{speed_txt}")

            frame = trace_ann.annotate(frame, det)
            frame = box_ann.annotate(frame, det)
            frame = label_ann.annotate(frame, det, labels=labels)
            frame = line_ann.annotate(frame, line_counter=line)
            sink.write_frame(frame)

            if verbose and idx % progress_every == 0:
                print(f"frame {idx}: {len(det)} vehicles tracked  "
                      f"(in={line.in_count} out={line.out_count})")

    return {
        "in_count": line.in_count,
        "out_count": line.out_count,
        "total_unique_ids": len(seen_ids),
        "output_path": output_path,
        "has_speed": view is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video path (default: bundled demo video)")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--conf", type=float, default=0.3)
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
        max_frames=args.max_frames,
        calib=calib,
        line_y_frac=args.line_y_frac,
    )

    print(f"\ncounts: in={stats['in_count']} out={stats['out_count']}")
    print(f"annotated video -> {stats['output_path']}")
    if not stats["has_speed"]:
        print("note: no --calib given, so no speeds. See README 'Calibrating for speed'.")


if __name__ == "__main__":
    main()
