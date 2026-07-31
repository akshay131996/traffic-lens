# Project 2 — Traffic Lens 🚗

Vehicle detection → tracking → line-crossing counts → calibrated speed estimates, plus a
foundation-model auto-labeling workflow and a YOLO26 fine-tune on VisDrone.

![Traffic Lens demo](outputs/media/traffic_demo.gif)

*YOLO26n + ByteTrack on 4K motorway footage: per-vehicle IDs, motion traces, a
line-crossing counter, and homography-derived speed labels.*

> 📓 **[LEARNINGS.md](LEARNINGS.md) — five silent bugs in run 1, and how each was caught.**
> The first pass of this project ran end-to-end without a single error and still produced
> wrong results: a baseline that measured label-space mismatch instead of model quality,
> speeds off by ~8×, and a tracking metric that couldn't answer its own question. None of
> them threw an exception. That write-up is the most useful thing in this repo.

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

> ⏳ **These are run-1 numbers.** A corrected run (v2 notebook: valid remapped baseline,
> measured calibration, 1280 px inference, 50 epochs) is in progress — see
> [LEARNINGS.md](LEARNINGS.md) for what changed and why. The invalid-baseline section
> below is kept deliberately, because the bug is more instructive than the number.

All numbers below come from one end-to-end run of
[`run_all_colab.ipynb`](run_all_colab.ipynb) (committed **with its outputs**, so you can
read the actual run without re-executing it) on an **NVIDIA RTX 4000 Ada (20 GB)**,
Ultralytics 8.4.104 / torch 2.4.1+cu124.

### VisDrone fine-tune

YOLO26n fine-tuned on [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)
(6,471 train / 548 val images, 10 classes), 20 epochs, imgsz 640, batch 16, ~23 min:

| metric | value |
|---|---|
| mAP50 | **0.246** |
| mAP50-95 | **0.134** |
| precision | 0.364 |
| recall | 0.292 |

Weights: [`yolo26n_visdrone_finetuned.pt`](yolo26n_visdrone_finetuned.pt) (5.4 MB) ·
per-epoch metrics: [`outputs/visdrone/results.csv`](outputs/visdrone/results.csv) ·
full training config: [`outputs/visdrone/args.yaml`](outputs/visdrone/args.yaml)

![Training curves](outputs/visdrone/results.png)

**The run wasn't finished learning.** Both val losses are still declining and mAP is
still climbing at epoch 20 — training stopped because the 20-epoch cosine schedule
decayed the LR to ~4e-5, not because the model plateaued. Train and val losses track each
other closely throughout (no overfitting), so a longer schedule would very likely score
higher. 20 epochs was a deliberate budget choice, not a converged result.

For reference, absolute numbers on VisDrone are *supposed* to look low — it's drone
imagery full of tiny, densely-packed objects, and this is the smallest model in the
family. Here's what it actually predicts:

![VisDrone predictions](outputs/visdrone/val_batch0_pred.jpg)

### ⚠️ The baseline in `visdrone_finetune_summary.json` is invalid — here's why

The notebook also evaluated the **COCO-pretrained** YOLO26n on VisDrone val and recorded
0.0174 mAP50, which would imply fine-tuning bought a ~14× improvement. **That comparison
does not hold up, and the number should not be quoted.**

The giveaway is in the per-class breakdown of that eval, which lists:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light
```

Those are **COCO's** class names. VisDrone's are `pedestrian, people, bicycle, car, van,
truck, tricycle, awning-tricycle, bus, motor`. The evaluation scored COCO *class indices*
against VisDrone *ground-truth indices* — so "airplane" predictions (COCO idx 4) were
graded against VisDrone's **van** boxes, "train" (idx 6) against **tricycle**, "boat"
(idx 8) against **bus**. Nine of the ten rows are comparing unrelated categories, and
0.0174 mostly measures that mismatch rather than any domain gap.

The one pair that accidentally lines up semantically is index 0 — COCO *person* vs
VisDrone *pedestrian* — and it scored **0.124 mAP50**, roughly 7× the bogus aggregate.
That's the only interpretable signal in the baseline, and it suggests genuine zero-shot
transfer is far better than 0.0174, while still well short of the fine-tuned model.

A defensible baseline would require remapping VisDrone ground truth into COCO's label
space (`pedestrian`+`people`→person, `car`+`van`→car, `motor`→motorcycle, `bus`→bus,
`truck`→truck, `bicycle`→bicycle) and dropping the classes with no COCO equivalent
(`tricycle`, `awning-tricycle`). That's left as future work — deliberately not papered
over with a number I can't defend.

**Takeaway:** a suspiciously catastrophic baseline is usually a bug in your evaluation,
not a discovery about your model. Always check that your predictions and your ground
truth live in the same label space before believing a comparison.

### ByteTrack parameter sweep

Same 200-frame clip, three tracker configurations
([`outputs/bytetrack_sweep.json`](outputs/bytetrack_sweep.json)):

| `track_activation_threshold` | `lost_track_buffer` | unique IDs | in / out |
|---|---|---|---|
| 0.50 | 10 | 7 | 2 / 1 |
| 0.25 | 30 | 13 | 2 / 1 |
| 0.10 | 60 | 14 | 2 / 1 |

**What this does and doesn't show.** Unique-ID count is *not* an ID-switch metric:
lowering the activation threshold admits more marginal detections, so some of the extra
IDs are genuinely additional (distant, low-confidence) vehicles rather than a stable track
being fragmented. Separating the two needs ground-truth track annotations and a proper
MOTA/IDF1 evaluation, which this demo clip doesn't have. The stable in/out counts across
all three settings are reassuring — the vehicles that actually cross the line are large
and confidently detected, so the counter is insensitive to these knobs.

### Speed calibration — estimated, and probably wrong

[`outputs/calib_demo.json`](outputs/calib_demo.json) maps an assumed 7 m × 30 m road
quadrilateral to image coordinates. It was placed by eye on a frame, **not measured**,
and the output shows it:

![Annotated frame](outputs/media/annotated_frame.jpg)

The labelled speeds (~53–64 km/h) are implausibly low for free-flowing motorway traffic,
which points to the assumed 30 m depth being a significant underestimate of the road
stretch actually spanned by that quad — under-estimating real-world distance
under-estimates speed proportionally. Treat the speed figures as a demonstration that the
homography pipeline works end-to-end, **not** as measurements. Fixing this needs a known
real-world reference in frame (measured lane markings, a GPS-tracked vehicle).

That frame also shows a second honest limitation: several vehicles near the gantry aren't
detected at all. 4K footage is downscaled to 640 px for inference, so distant vehicles
shrink to a handful of pixels — the classic small-object problem, and the same reason
VisDrone is hard. Higher inference resolution or tiled (SAHI) inference is the fix.

### Auto-labeling with Grounding DINO

Zero-shot boxes from the text prompt `"car. truck. bus. motorcycle."` — no training, no
manual annotation:

![Auto-labeling preview](outputs/media/autolabel_preview.jpg)

Output is written in YOLO format, ready for import into CVAT for human correction. The
correction pass itself — and the "how much faster is correcting than drawing from
scratch?" timing — is genuinely manual work and hasn't been done yet; it's listed below.

## Definition of done

- [x] Pipeline code + Gradio app + auto-labeling + fine-tune notebook all written and pushed
- [x] `run_all_colab.ipynb` executed end-to-end on GPU, artifacts downloaded and committed
- [x] Real VisDrone fine-tune numbers + trained weights published
- [x] Baseline comparison investigated — and documented as invalid rather than quoted
- [ ] A *valid* pretrained baseline via label-space remapping (see Results)
- [ ] Longer training schedule, since 20 epochs ended before the model stopped improving
- [ ] Benchmark table (YOLO26 vs YOLO11 vs RT-DETR) in this README
- [ ] Speed calibration against a measured real-world reference
- [ ] Own footage (not just the bundled demo video) run through the pipeline
- [ ] CVAT correction pass + auto-label-vs-scratch timing
- [ ] Blog post: the auto-labeling workflow and the label-space bug
