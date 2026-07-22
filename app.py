"""Gradio app — upload a traffic video, get back an annotated one (detect+track+count,
+speed if you supply a calibration). This exact file (plus requirements.txt) is what
you upload to HF Spaces later; for now it's exercised inside run_all_colab.ipynb.

    python app.py    ->  http://127.0.0.1:7860
"""
import json
import tempfile
from pathlib import Path

import gradio as gr

from track_count_speed import load_model, process_video

_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]


def run(video_path, model_name, conf, line_y_frac, calib_file):
    if video_path is None:
        return None, "Upload a video first."

    calib = None
    if calib_file is not None:
        calib = json.loads(Path(calib_file).read_text())

    out_path = str(Path(tempfile.gettempdir()) / "traffic_lens_annotated.mp4")
    stats = process_video(
        source=video_path,
        output_path=out_path,
        model=_get_model(model_name),
        conf=conf,
        calib=calib,
        line_y_frac=line_y_frac,
        verbose=False,
    )

    summary = (
        f"in: {stats['in_count']}  |  out: {stats['out_count']}  |  "
        f"unique vehicles tracked: {stats['total_unique_ids']}"
        + ("  |  speeds estimated" if stats["has_speed"] else "  |  no calibration -> no speeds")
    )
    return stats["output_path"], summary


with gr.Blocks(title="Traffic Lens") as iface:
    gr.Markdown(
        "# 🚗 Traffic Lens\n"
        "Upload traffic footage to detect, track, and count vehicles (cars/motorcycles/"
        "buses/trucks). Optionally supply a calibration JSON for speed estimates."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Upload traffic video")
            model_dd = gr.Dropdown(["yolo26n.pt", "yolo11n.pt"], value="yolo26n.pt", label="Model")
            conf_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence threshold")
            line_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                    label="Counting line position (fraction of frame height)")
            calib_in = gr.File(label="Calibration JSON (optional -- enables speed)", file_types=[".json"])
            run_btn = gr.Button("Process", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Annotated result (downloadable)")
            summary_out = gr.Textbox(label="Summary")

    run_btn.click(run, inputs=[video_in, model_dd, conf_slider, line_slider, calib_in],
                  outputs=[video_out, summary_out])

if __name__ == "__main__":
    iface.launch()
