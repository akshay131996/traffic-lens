"""Auto-label images with Grounding DINO (open-vocabulary detection) -> YOLO format.

The 2026 labeling workflow: a foundation model proposes boxes from a text prompt, you
*correct* them in CVAT/Label Studio instead of drawing from scratch (~5x faster).
Never ship auto-labels unreviewed — that's the honest part of the blog post.

    python autolabel.py --source my_frames/ --classes "car,truck,bus,motorcycle"
"""
import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of .jpg/.png frames")
    ap.add_argument("--classes", default="car,truck,bus,motorcycle")
    ap.add_argument("--out", default="autolabel_dataset")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--preview", type=int, default=5, help="save N preview images with boxes")
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    classes = [c.strip() for c in args.classes.split(",")]
    # Grounding DINO expects lowercase queries separated by ". "
    text = ". ".join(classes) + "."

    out = Path(__file__).parent / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "preview").mkdir(exist_ok=True)

    images = sorted(p for p in Path(args.source).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"labeling {len(images)} images with prompt: '{text}'")

    total_boxes = 0
    for n, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.conf,
            text_threshold=args.conf, target_sizes=[image.size[::-1]])[0]

        w, h = image.size
        lines, draw_img = [], image.copy()
        draw = ImageDraw.Draw(draw_img)
        for box, label in zip(result["boxes"], result["text_labels"]):
            cls = next((i for i, c in enumerate(classes) if c in label), None)
            if cls is None:
                continue
            x0, y0, x1, y1 = box.tolist()
            # YOLO format: class x_center y_center width height, all normalized 0-1
            lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, max(0, y0 - 12)), label, fill="red")

        shutil.copy(path, out / "images" / path.name)
        (out / "labels" / f"{path.stem}.txt").write_text("\n".join(lines))
        if n < args.preview:
            draw_img.save(out / "preview" / path.name)
        total_boxes += len(lines)
        print(f"  {path.name}: {len(lines)} boxes")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        f"names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(classes)) + "\n")
    print(f"\n{total_boxes} boxes across {len(images)} images -> {out}")
    print("NEXT: eyeball preview/, import into CVAT and fix errors, split train/val,"
          "\nthen fine-tune: yolo train model=yolo26n.pt data=autolabel_dataset/data.yaml")


if __name__ == "__main__":
    main()
