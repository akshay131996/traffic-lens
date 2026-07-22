# Project 2 — Traffic Lens 🚗

**Roadmap week 2.** Vehicle detection → tracking → counting → (calibrated) speed, plus
the 2026 auto-labeling workflow. This is where your YOLOv5 knowledge gets upgraded.

## Lessons

1. **What changed in YOLO since v5** (interview staple): anchor boxes died (v8 went anchor-free), then NMS died (v10/YOLO26 are trained end-to-end with one-to-one label assignment, so no duplicate-box cleanup step). Result: simpler pipelines, faster CPU/edge inference. The API you remember (`model(image)`) survived.
2. **YOLO vs DETR-family:** transformer detectors (RT-DETR, RF-DETR) win on accuracy at similar model sizes; YOLO wins on edge speed and tooling maturity. "It depends, here's my benchmark" is the senior-engineer answer — run `benchmark.py --include-rtdetr` and have your own numbers.
3. **Tracking = detection + data association.** ByteTrack is dumb-simple (IoU matching + Kalman prediction, and it *keeps low-confidence boxes* — that's the paper's whole trick) and still near-SOTA. Know why ID switches happen: occlusion, missed detections, crossing paths.
4. **Auto-labeling changed data economics.** Grounding DINO/SAM 3 propose labels from text prompts; humans correct instead of draw. `autolabel.py` is that workflow. The catch — and your blog angle — is *silent bias*: the foundation model misses exactly the hard cases you most need labeled.
5. **Speed needs geometry, not ML.** Pixel distances are distorted by perspective; a homography maps the road plane to meters (see `ViewTransformer`). Calibration: pick 4 points on the road forming a rectangle you can measure in the real world (lane width ≈ 3.5 m, dashed-line period ≈ 12 m in many countries) — that's your `calib.json`.

## Run everything on Colab (this is how the results below were generated)

This laptop has no GPU, so all real execution — the calibrated pipeline run, the
ByteTrack parameter sweep, the auto-labeling demo, the VisDrone fine-tune, and a
functional test of the app below — happens in one notebook:
**[`run_all_colab.ipynb`](run_all_colab.ipynb)**. Open it in Colab, set
**Runtime > Change runtime type > T4 GPU**, Run all, and it downloads every artifact
(annotated video, sweep results, auto-label preview bundle, fine-tuned weights, mAP
comparison) at the end.

The VisDrone fine-tune section is the long one (dataset download + ~20 epochs on a T4 is
roughly 30-60+ minutes) — see the notebook's note about switching to a remote VM if a
Colab session limit gets hit before it finishes.

## Run it locally

The scripts still work standalone if you have your own GPU (or just want a quick CPU
check on a short clip) — this is what `run_all_colab.ipynb` calls under the hood:

```powershell
# demo: bundled highway video, pretrained model, 100 frames
python track_count_speed.py --max-frames 100
# -> outputs/annotated.mp4  (open it! traces, IDs, line counts)

# benchmark on your machine
python benchmark.py

# auto-label your own frames (model downloads ~700MB once)
python autolabel.py --source my_frames/ --classes "car,truck,bus"

# upload-a-video app (Gradio) -- same pipeline, browser UI
python app.py
```

Extract frames from any video for labeling:
`ffmpeg -i dashcam.mp4 -vf fps=2 my_frames/f_%04d.jpg` (or use the notebook's cv2 cell).

## Getting real data (the part that makes it portfolio-worthy)

- **Best:** your own phone/dashcam footage of a busy road (tripod, 10 min, 1080p). Custom data = the story interviewers ask about. Point `run_all_colab.ipynb` or `track_count_speed.py --source` at it once you have it.
- **What this repo actually fine-tunes on today:** Ultralytics' built-in `VisDrone.yaml` (drone traffic, properly labeled, auto-downloads) — chosen specifically so the fine-tune/mAP comparison doesn't block on manual label correction.
- **The auto-labeling workflow** (`autolabel.py` / the notebook's section 6) is demonstrated separately on frames from the bundled demo video, producing YOLO-format labels ready for CVAT import. Correcting those by hand and timing it against labeling from scratch is a real exercise worth doing on your own footage — that's inherently manual, not something to script away.

## What's here

| File | Purpose |
|---|---|
| `run_all_colab.ipynb` | **Run this first.** Does everything below on a Colab GPU, downloads every artifact. |
| `track_count_speed.py` | Detect+track+count(+speed) pipeline — `process_video()` is the shared core logic |
| `app.py` | Gradio app: upload a video, get an annotated one back, built on `process_video()` |
| `autolabel.py` | Grounding-DINO auto-labeling — text prompt in, YOLO-format labels out |
| `benchmark.py` | YOLO26 vs YOLO11 (vs RT-DETR) latency benchmark |
| `calib_demo.json` | Estimated speed-calibration (from the notebook, downloaded back — see Results) |

## 🔨 Your turn

1. Point the pipeline at your own footage (phone/dashcam) instead of the bundled demo video — everything else stays the same.
2. Time yourself: label 20 frames from scratch in CVAT vs correcting the notebook's auto-labeled output. That ratio is blog post #2's headline number.
3. Fine-tune on your own corrected dataset instead of VisDrone, once you have one, and compare against both the VisDrone-fine-tuned and pretrained-COCO checkpoints.
4. Verify the calibrated speed estimates against a real, known reference (a speed-limit sign, a GPS speedometer) rather than trusting the estimate as-is.
5. Deploy `app.py` to a Hugging Face Space for a live demo link.

## Results

_Run `run_all_colab.ipynb` and paste the real numbers here: ByteTrack sweep table,
VisDrone pretrained-vs-fine-tuned mAP, in/out counts from the calibrated run._

## Definition of done

- [x] Pipeline code + Gradio app + auto-labeling + fine-tune notebook all written and pushed
- [ ] `run_all_colab.ipynb` executed end-to-end, artifacts downloaded and committed
- [ ] Real VisDrone pretrained-vs-fine-tuned mAP numbers in the Results section above
- [ ] Benchmark table (YOLO26 vs YOLO11 vs RT-DETR) in this README
- [ ] Own footage (not just the bundled demo video) run through the pipeline at least once
- [ ] Blog post #2: the auto-labeling workflow with honest numbers
