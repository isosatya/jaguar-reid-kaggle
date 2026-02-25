#!/usr/bin/env python3
"""
Crop raw jaguar images to jaguar-only regions using SAM 3.

Designed to run on a GPU pod (e.g. RunPod). By default refuses to run without
CUDA so you don't accidentally run heavy inference on a local machine. Use
--allow-cpu only for local testing (will be slow).

Usage (on RunPod / GPU machine):
    python scripts/crop_jaguars_sam3.py

Usage (local, skip processing):
    python scripts/crop_jaguars_sam3.py   # exits with instructions to use RunPod

Usage (local, allow CPU for testing):
    python scripts/crop_jaguars_sam3.py --allow-cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# SAM3 imports (heavy; only after device check when not --allow-cpu)
def _load_sam3():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    return build_sam3_image_model, Sam3Processor


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Crop images to jaguar-only regions using SAM 3 (intended for GPU/RunPod)."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw_gallery/jaguars_images"),
        help="Directory containing raw jaguar images",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_gallery"),
        help="Directory where cropped jaguar images are saved",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="jaguar",
        help='Text prompt for SAM 3 (e.g. "jaguar", "jaguar flank pattern", "rosette skin")',
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="Minimum detection score to keep a crop (0–1)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=2000,
        help="Minimum bounding box area in pixels to keep a detection",
    )
    p.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Fraction of box size to add as padding on each side (0–1)",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Device for inference. Default: cuda (use on RunPod).",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running on CPU if CUDA is not available (slow; for local testing only)",
    )
    return p.parse_args()


def get_device(args) -> str:
    """Resolve device. Refuse to run on local machine without GPU unless --allow-cpu."""
    if args.device == "cpu":
        return "cpu"
    if args.device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if args.allow_cpu:
            print("Warning: CUDA not available, falling back to CPU (slow).", file=sys.stderr)
            return "cpu"
        print(
            "CUDA is not available. This script is intended to run on a GPU pod (e.g. RunPod).\n"
            "Run this script on your GPU pod, or use --allow-cpu to run on this machine anyway (slow).",
            file=sys.stderr,
        )
        sys.exit(1)
    # auto: prefer cuda
    if torch.cuda.is_available():
        return "cuda"
    if args.allow_cpu:
        return "cpu"
    print(
        "CUDA is not available. Run on a GPU pod (e.g. RunPod) or use --allow-cpu.",
        file=sys.stderr,
    )
    sys.exit(1)


def expand_box(x0: float, y0: float, x1: float, y1: float, padding: float, w: int, h: int):
    """Expand box by padding fraction and clamp to image [0,w] x [0,h]."""
    bw, bh = x1 - x0, y1 - y0
    dx = bw * padding
    dy = bh * padding
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w, x1 + dx)
    y1 = min(h, y1 + dy)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def main():
    args = parse_args()
    device = get_device(args)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading SAM 3 model on device={device}...")
    build_sam3_image_model, Sam3Processor = _load_sam3()
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.min_score)
    print("Model loaded.")

    total_crops = 0
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue
        w, h = image.size
        state = processor.set_image(image)
        state = processor.set_text_prompt(args.prompt, state)
        boxes = state["boxes"]
        scores = state["scores"]
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.cpu()
        scores = scores.cpu()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        for i, (box, score) in enumerate(zip(boxes.tolist(), scores.tolist())):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            if area < args.min_area:
                continue
            x0, y0, x1, y1 = expand_box(x0, y0, x1, y1, args.padding, w, h)
            crop = image.crop((x0, y0, x1, y1))
            stem = path.stem
            ext = path.suffix.lower()
            out_name = f"{stem}_crop_{i}{ext}"
            out_path = output_dir / out_name
            crop.save(out_path, quality=95)
            total_crops += 1
            print(f"  {path.name} -> {out_name} (score={score:.2f})")

    print(f"Done. Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
