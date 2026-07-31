"""Gradio app — upload traffic footage, get back an annotated video with vehicle
tracks, line-crossing counts, and (with a calibration) speed estimates.

Deployable as-is to Hugging Face Spaces: this file + requirements.txt + the weights.

    python app.py    ->  http://127.0.0.1:7860
    python app.py --share    -> public link (useful from a remote GPU box)
"""
import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr

from track_count_speed import load_model, process_video, vehicle_class_ids

HERE = Path(__file__).parent
FINETUNED = HERE / "yolo26n_visdrone_finetuned.pt"

# Calibration measured for supervision's vehicles.mp4 demo clip (25 m x 250 m of road).
# Only valid for THAT camera -- any other footage needs its own measured quad.
DEMO_CALIB = {
    "source_px": [[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]],
    "target_m": [[0, 0], [24, 0], [24, 249], [0, 249]],
}

MODEL_CHOICES = {
    "YOLO26n — COCO pretrained (general traffic)": "yolo26n.pt",
    "YOLO11n — COCO pretrained": "yolo11n.pt",
}
if FINETUNED.exists():
    MODEL_CHOICES["YOLO26n — fine-tuned on VisDrone (drone/overhead views)"] = str(FINETUNED)

_model_cache: dict = {}


def _get_model(path: str):
    if path not in _model_cache:
        _model_cache[path] = load_model(path)
    return _model_cache[path]


def run(video_path, model_label, conf, imgsz, line_y_frac, use_demo_calib, calib_file):
    if video_path is None:
        return None, "Upload a video first."

    calib = None
    calib_note = "no calibration → no speeds"
    if calib_file is not None:
        calib = json.loads(Path(calib_file).read_text())
        calib_note = "speeds from your uploaded calibration"
    elif use_demo_calib:
        calib = DEMO_CALIB
        calib_note = "speeds from the built-in demo calibration (only valid for the sample clip)"

    model = _get_model(MODEL_CHOICES[model_label])
    out_path = str(Path(tempfile.gettempdir()) / "traffic_lens_annotated.mp4")

    stats = process_video(
        source=video_path,
        output_path=out_path,
        model=model,
        conf=conf,
        imgsz=int(imgsz),
        calib=calib,
        line_y_frac=line_y_frac,
        verbose=False,
    )

    speed_line = ""
    if stats["median_speed_kmh"] is not None:
        speed_line = f"\nmedian speed: {stats['median_speed_kmh']} km/h  — sanity-check this against the road type"

    summary = (
        f"crossings — in: {stats['in_count']}, out: {stats['out_count']}\n"
        f"unique vehicles tracked: {stats['total_unique_ids']}\n"
        f"detections/frame: {stats['detections_per_frame']}   "
        f"mean track length: {stats['mean_track_length_frames']} frames\n"
        f"inference size: {stats['imgsz']} px\n"
        f"vehicle classes in this model: {', '.join(stats['vehicle_classes'])}\n"
        f"{calib_note}{speed_line}"
    )
    return stats["output_path"], summary


with gr.Blocks(title="Traffic Lens") as iface:
    gr.Markdown(
        "# 🚗 Traffic Lens\n"
        "Detect, track and count vehicles in traffic footage — with optional speed "
        "estimation via homography.\n\n"
        "**Notes worth reading before trusting the output:**\n"
        "- The VisDrone fine-tune is trained on *drone/overhead* imagery. On road-level "
        "footage the COCO-pretrained model is usually the better choice — the selector "
        "lets you compare rather than assume.\n"
        "- Speeds are only as good as the calibration. The built-in one is measured for "
        "the sample clip **only**; other footage needs its own measured road quad, or the "
        "numbers will be confidently wrong.\n"
        "- On 4K footage, raise inference size to 1280+ or distant vehicles are missed."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Traffic video")
            model_dd = gr.Dropdown(list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0],
                                   label="Model")
            imgsz_dd = gr.Dropdown([640, 960, 1280, 1600], value=1280,
                                   label="Inference size (px) — higher finds smaller/distant vehicles")
            conf_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence threshold")
            line_slider = gr.Slider(0.1, 0.9, value=0.6, step=0.05,
                                    label="Counting line position (fraction of frame height)")
            demo_calib_cb = gr.Checkbox(value=False,
                                        label="Use built-in demo calibration (sample clip only)")
            calib_in = gr.File(label="…or upload your own calibration JSON",
                               file_types=[".json"])
            run_btn = gr.Button("Process", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Annotated result (downloadable)")
            summary_out = gr.Textbox(label="Summary", lines=8)

    run_btn.click(run,
                  inputs=[video_in, model_dd, conf_slider, imgsz_dd, line_slider,
                          demo_calib_cb, calib_in],
                  outputs=[video_out, summary_out])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    iface.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")
