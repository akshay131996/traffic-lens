# What went wrong in run 1 — and how each was found

Run 1 of [`run_all_colab.ipynb`](run_all_colab.ipynb) completed without a single error
and produced results that were, in several places, wrong. Every bug below was silent:
the code ran, numbers came out, plots rendered. They were caught by sanity-checking the
*outputs* against physical and domain expectations, not by anything crashing.

That's the actual lesson of this project, so it's documented here rather than quietly
fixed.

---

## 1. The baseline that measured nothing

**Claimed:** COCO-pretrained YOLO26n scores 0.0174 mAP50 on VisDrone; fine-tuning
improves it ~14×.

**The code:**
```python
YOLO("yolo26n.pt").val(data="VisDrone.yaml")   # looks completely reasonable
```

**What actually happened.** The pretrained model predicts COCO's 80 classes. The ground
truth uses VisDrone's 10. `val()` matches predictions to ground truth **by class index**,
not by class name — so it compared:

| index | model predicted (COCO) | graded against (VisDrone) |
|---|---|---|
| 0 | person | pedestrian ← *only near-match* |
| 2 | car | bicycle |
| 4 | **airplane** | **van** |
| 6 | **train** | **tricycle** |
| 8 | **boat** | **bus** |

Nine of ten rows compared unrelated categories. 0.0174 measured label-space mismatch, not
model quality.

**How it was caught.** The per-class table printed `airplane`, `train`, and `boat` — for a
dataset of drone footage over Chinese streets. A dataset with no aircraft and no boats
returning per-class scores for aircraft and boats is a contradiction; that's the thread to
pull. Corroborating evidence: index 0 (`person` vs `pedestrian`, the one pair that lines
up semantically) scored **0.124 mAP50**, ~7× the aggregate — exactly the pattern you'd
expect if one row is real and nine are noise.

**Fix (§8 of the v2 notebook):** translate VisDrone ground truth into COCO's label space
(`pedestrian`+`people`→person, `car`+`van`→car, `motor`→motorcycle, `bus`→bus,
`truck`→truck, `bicycle`→bicycle), drop `tricycle`/`awning-tricycle` which have no COCO
equivalent, and evaluate both models on the shared categories.

**Transferable rule:** before comparing a model to a dataset it wasn't trained on, verify
predictions and ground truth live in the same label space. A suspiciously catastrophic
baseline is nearly always a broken evaluation, not a discovery.

---

## 2. Speeds that were physically implausible

**Claimed:** vehicles travelling 53–64 km/h.

**The scene:** a free-flowing four-lane motorway with no congestion.

**What actually happened.** The homography needs a real-world rectangle to map onto. Run 1
placed one by eye at arbitrary frame fractions and asserted it covered **7 m × 30 m** of
road. The road region actually visible in that quad is closer to **25 m × 250 m**.

Speed is distance over time, so distance error propagates linearly: under-measuring the
road by ~8× under-reports every speed by ~8×. 55 km/h × 8 ≈ 110 km/h — a completely
ordinary motorway speed.

**How it was caught.** Not by any metric — by asking "is this number physically
sensible?" Nothing in the pipeline knows motorways aren't 55 km/h zones. A human looking
at the annotated frame does.

**Fix (§3):** use the measured `SOURCE`/`TARGET` published for this exact clip in
[`supervision`'s speed-estimation example](https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation),
plus a guard rejecting points outside the calibrated region.

**Transferable rule:** any pipeline converting pixels to physical units needs a
ground-truth anchor. "It ran and produced numbers" says nothing about whether the numbers
mean anything.

---

## 3. Half the vehicles were never detected

**Symptom:** in the annotated frame, vehicles near the overhead gantry have no boxes,
despite being clearly visible.

**Cause:** the source is 4K (3840×2160); inference ran at the 640 px default. A car
~60 px wide in the original is ~10 px after downscaling — below what the detector can
reliably fire on. This is the same small-object problem that makes VisDrone itself hard.

**How it was caught.** Looking at the output image instead of only the summary counts.
The counter said "in: 1, out: 1" and was technically correct — about the two cars it
could see.

**Fix (§4):** infer at 1280 px, with the run-2 notebook reporting a 640-vs-1280
detections-per-frame comparison so the gain is measured rather than assumed.

---

## 4. A tracking metric that couldn't answer its own question

**Claimed:** ByteTrack settings compared via `total_unique_ids` (7 / 13 / 14 across three
configurations).

**The problem.** The intent was to measure track *fragmentation* — a tracker losing a car
and re-issuing it as a new ID. But lowering `track_activation_threshold` also admits more
genuine detections of distant vehicles. Both effects raise the unique-ID count, and the
metric can't separate them, so the numbers can't support any conclusion.

**Fix (§5):** log `detections_per_frame` and `mean_track_length_frames` alongside. IDs up
*with* detections up and track length steady ⇒ more real objects. IDs up *with* track
length down ⇒ fragmentation. Still a proxy — a true ID-switch count needs ground-truth
tracks and MOTA/IDF1 — and it's labelled as one.

**Transferable rule:** state what a metric would look like under each hypothesis *before*
collecting it. If two opposite explanations predict the same movement, it isn't a metric
yet.

---

## 5. Training that stopped for the wrong reason

**Claimed (implicitly):** 20 epochs, final mAP50 0.246 — presented as the model's result.

**What the curves showed.** Both validation losses still declining and mAP still rising at
epoch 20. Training ended because the 20-epoch cosine schedule had decayed the learning
rate to ~4e-5, not because the model had stopped learning.

**How it was caught.** Reading `results.png` rather than only the final row of
`results.csv`. Train and val losses tracked each other throughout — no overfitting, no
plateau, just a schedule that ran out.

**Fix (§7):** 50 epochs, giving the curve room to actually flatten.

**Transferable rule:** "the run finished" and "the model converged" are different claims.
Only the loss curve distinguishes them.

---

## Environment issues (cost time, no bad results)

- **`transformers` 5.x + `torch` 2.4.1 → `ImportError: cannot import name 'DTensor'`.**
  v5's checkpoint-loading internals require `torch.distributed.tensor.DTensor`, absent
  from the torch build on the GPU image. Breaks `from transformers import AutoProcessor`
  entirely — not specific to Grounding DINO. Pinned `transformers<5`.
- **`numpy` ≥ 2.4 breaks `supervision`'s `LineZone`**, which calls `np.cross` on 2-D
  vectors (removed in 2.4). Pinned `numpy<2.4`.
- **Ephemeral container storage.** Work was first cloned to `/root/projects`, which a pod
  restart wiped along with every pip install. Only `/workspace` persists. Code now lives
  there, and the notebook reinstalls its dependencies in cell 0 by design.

---

## The through-line

None of these produced a stack trace. A pipeline that runs end-to-end and emits
plausible-looking JSON can still be wrong in five places at once, and the failures were
only visible by checking outputs against things known independently of the code: motorways
aren't 55 km/h zones, drone datasets don't contain boats, loss curves that are still
falling mean training isn't done.
