# Project 2 — Traffic Lens 🚗

**Roadmap week 2.** Vehicle detection → tracking → counting → (calibrated) speed, plus
the 2026 auto-labeling workflow. This is where your YOLOv5 knowledge gets upgraded.

## Lessons

1. **What changed in YOLO since v5** (interview staple): anchor boxes died (v8 went anchor-free), then NMS died (v10/YOLO26 are trained end-to-end with one-to-one label assignment, so no duplicate-box cleanup step). Result: simpler pipelines, faster CPU/edge inference. The API you remember (`model(image)`) survived.
2. **YOLO vs DETR-family:** transformer detectors (RT-DETR, RF-DETR) win on accuracy at similar model sizes; YOLO wins on edge speed and tooling maturity. "It depends, here's my benchmark" is the senior-engineer answer — run `benchmark.py --include-rtdetr` and have your own numbers.
3. **Tracking = detection + data association.** ByteTrack is dumb-simple (IoU matching + Kalman prediction, and it *keeps low-confidence boxes* — that's the paper's whole trick) and still near-SOTA. Know why ID switches happen: occlusion, missed detections, crossing paths.
4. **Auto-labeling changed data economics.** Grounding DINO/SAM 3 propose labels from text prompts; humans correct instead of draw. `autolabel.py` is that workflow. The catch — and your blog angle — is *silent bias*: the foundation model misses exactly the hard cases you most need labeled.
5. **Speed needs geometry, not ML.** Pixel distances are distorted by perspective; a homography maps the road plane to meters (see `ViewTransformer`). Calibration: pick 4 points on the road forming a rectangle you can measure in the real world (lane width ≈ 3.5 m, dashed-line period ≈ 12 m in many countries) — that's your `calib.json`.

## Run it

```powershell
# demo: bundled highway video, pretrained model, 100 frames (~2 min CPU)
python track_count_speed.py --max-frames 100
# -> outputs/annotated.mp4  (open it! traces, IDs, line counts)

# benchmark on your machine
python benchmark.py

# auto-label your own frames (model downloads ~700MB once)
python autolabel.py --source my_frames/ --classes "car,truck,bus"
```

Extract frames from any video for labeling:
`ffmpeg -i dashcam.mp4 -vf fps=2 my_frames/f_%04d.jpg` (or use the notebook's cv2 cell).

## Getting real data (the part that makes it portfolio-worthy)

- **Best:** your own phone/dashcam footage of a busy road (tripod, 10 min, 1080p). Custom data = the story interviewers ask about.
- Public fallbacks: VisDrone (drone traffic), UA-DETRAC (surveillance), any YouTube CCTV compilation for *personal-project* use.
- Label loop: `autolabel.py` → import to [CVAT](https://cvat.ai) → correct ~300 frames → `yolo train model=yolo26n.pt data=autolabel_dataset/data.yaml epochs=50` (⚡ Kaggle GPU).

## 🔨 Your turn

1. Move the counting line in `track_count_speed.py` to match a real camera angle; add a second line for a turn lane.
2. Time yourself: label 20 frames from scratch in CVAT vs correcting auto-labels. That ratio is blog post #2's headline number.
3. Fine-tune YOLO26n on your corrected dataset; compare val mAP of pretrained-COCO vs fine-tuned on 30 held-out frames. Feel the domain-gap lesson.
4. Calibrate speed on real footage using lane-width geometry; sanity-check against the speed limit.
5. Swap ByteTrack params (`track_activation_threshold`, `lost_track_buffer`) and count ID switches in 60 s of video — write down what each knob trades.

## Definition of done

- [ ] Annotated video from YOUR footage with counts (+ speeds if calibrated)
- [ ] ≥300 self-corrected labels + fine-tuned model beating pretrained on your val set
- [ ] Benchmark table (YOLO26 vs YOLO11 vs RT-DETR) in this README
- [ ] Blog post #2: the auto-labeling workflow with honest numbers
