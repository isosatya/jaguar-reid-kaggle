#!/usr/bin/env python3
"""
Crop raw jaguar images using Grounding DINO (Hugging Face).

Uses text-prompted detection only (no SAM). Prompts describe parts/visible
regions (jaguar fur, body, rosette, etc.) so the model does not expect a full
jaguar. Good for partial/occluded jaguars in vegetation.

Requires: pip install transformers torch Pillow
Model: IDEA-Research/grounding-dino-tiny (or grounding-dino-base for better accuracy).

Usage:
    python scripts/crop_jaguars_grounding_dino.py
    python scripts/crop_jaguars_grounding_dino.py --input-dir /path/to/images --limit 5
    python scripts/crop_jaguars_grounding_dino.py --resume
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from PIL import Image

# Default text prompts: parts/visible regions (Grounding DINO separates classes with " . ")
DEFAULT_TEXT_PROMPTS = [
    "a jaguar .",
    "jaguar fur .",
    "jaguar body .",
    "part of a jaguar .",
    "jaguar rosette .",
    "jaguar flank .",
    "big cat fur .",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prompt_to_slug(prompt: str) -> str:
    s = prompt.strip().replace(" ", "_").replace(".", "").strip()
    s = re.sub(r"[^\w\-]", "", s) or "prompt"
    return s


def expand_box(x0: float, y0: float, x1: float, y1: float, padding: float, w: int, h: int):
    bw, bh = x1 - x0, y1 - y0
    dx = bw * padding
    dy = bh * padding
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w, x1 + dx)
    y1 = min(h, y1 + dy)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def parse_args():
    p = argparse.ArgumentParser(description="Crop jaguar images with Grounding DINO (HF).")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw_gallery/jaguars_images"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed_gallery_grounding_dino"))
    p.add_argument(
        "--prompts",
        type=str,
        default=None,
        help='Comma-separated text prompts. Default: part-friendly jaguar phrases.',
    )
    p.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--min-score", type=float, default=0.28)
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--padding", type=float, default=0.08)
    p.add_argument("--max-area-ratio", type=float, default=0.90)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.allow_cpu else "cpu"
    if device == "cpu":
        print("Warning: running on CPU (slow). Use GPU or --allow-cpu to confirm.", file=sys.stderr)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts:
        text_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        text_prompts = list(DEFAULT_TEXT_PROMPTS)

    # Single list of phrases per image (Grounding DINO expects list of list for batch; we do one image at a time)
    text_labels = [text_prompts]

    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        image_paths = image_paths[: args.limit]
        print(f"Limit: processing {len(image_paths)} images")

    print(f"Loading Grounding DINO {args.model_id} on {device}...")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)
    print("Model loaded.")

    total_crops = 0
    skipped = 0
    for path in image_paths:
        try:
            stem = "_".join(path.relative_to(input_dir).with_suffix("").parts)
        except ValueError:
            stem = path.stem
        if args.resume and list(output_dir.glob(f"{stem}_crop_*{path.suffix}")):
            skipped += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue
        w, h = image.size

        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=args.min_score,
            text_threshold=0.25,
            target_sizes=[(h, w)],
        )
        result = results[0]
        boxes = result["boxes"].cpu()
        scores = result["scores"].cpu()
        labels = result.get("text_labels") or result.get("labels")
        if labels is not None and hasattr(labels, "cpu"):
            labels = labels.cpu()
        if boxes.shape[0] == 0:
            continue

        saved = 0
        for i in range(boxes.shape[0]):
            box = boxes[i].tolist()
            score = scores[i].item()
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            if area < args.min_area:
                continue
            x0, y0, x1, y1 = expand_box(x0, y0, x1, y1, args.padding, w, h)
            crop_area = (x1 - x0) * (y1 - y0)
            if w * h > 0 and (crop_area / (w * h)) > args.max_area_ratio:
                continue
            crop = image.crop((x0, y0, x1, y1))
            label_slug = "jaguar"
            if labels is not None and i < len(labels):
                try:
                    lab = labels[i]
                    if isinstance(lab, str):
                        label_slug = prompt_to_slug(lab)
                    else:
                        label_slug = f"class{lab}"
                except Exception:
                    pass
            ext = path.suffix.lower()
            out_name = f"{stem}_crop_{saved}_{label_slug}{ext}"
            out_path = output_dir / out_name
            crop.save(out_path, quality=95)
            total_crops += 1
            saved += 1
            print(f"  {path.name} -> {out_name} (score={score:.2f})")

    if skipped:
        print(f"Skipped {skipped} (already had crops).")
    print(f"Done. Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
