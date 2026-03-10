#!/usr/bin/env python3
"""
Crop raw jaguar images to jaguar-only regions using SAM 3.

Outputs contour crops (jaguar shape, transparent background) as PNG, not rectangular frames.
Images often show parts of jaguars with vegetation; prompts describe visible parts (fur, body, rosette, etc.).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# SAM3 imports (heavy; only after device check when not --allow-cpu)
def _load_sam3():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    return build_sam3_image_model, Sam3Processor


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Max number of reference crop examples to use when reference-crops-dir is set
MAX_REFERENCE_CROP_EXAMPLES = 5

# Prefer object-level prompts to avoid matching water/foliage; one part-friendly for occluded shots
DEFAULT_PROMPTS = [
    "a jaguar",
    "jaguar",
    "part of a jaguar",
    "jaguar body",
]


def prompt_to_slug(prompt: str) -> str:
    """Sanitize prompt for use in filenames (alphanumeric and underscores only)."""
    s = prompt.strip().replace(" ", "_").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^\w\-]", "", s)  # drop other non-word chars
    return s or "prompt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Crop images to jaguar-only regions using SAM 3 (intended for GPU/RunPod)."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw_gallery/jaguars_images"),
        help="Directory containing raw jaguar images (searched recursively)",
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
        default="a jaguar",
        help='Single prompt (used only with --single). Use "a jaguar" to avoid texture false matches.',
    )
    p.add_argument(
        "--prompts",
        type=str,
        default=None,
        metavar="P1,P2,...",
        help='Comma-separated list of prompts; overrides default list. Use e.g. --prompts "jaguar" for single prompt.',
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Use only --prompt (single prompt) instead of the default prompt list.",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum detection score (0-1). Higher reduces water/background false crops.",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=800,
        help="Minimum bounding box area in pixels to keep a detection (partial jaguars can be small).",
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
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N images (for dry run). Omit for full run.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip input images that already have at least one crop in the output dir (for resuming after pod stop).",
    )
    p.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.90,
        metavar="R",
        help="Skip saving a crop if its area is larger than this fraction of the image (avoids full-frame 'crops'). Default 0.90. Use 1.0 to allow full-frame.",
    )
    p.add_argument(
        "--reference-crops-dir",
        type=Path,
        default=Path("data/train_crops/train"),
        metavar="DIR",
        help="Directory of good crop examples. Default: data/train_crops/train (up to 5 random for reference mode).",
    )
    p.add_argument(
        "--reference-crop",
        type=Path,
        default=None,
        metavar="PATH",
        help="(Deprecated) Single reference crop path. Prefer --reference-crops-dir.",
    )
    p.add_argument("--no-reference", action="store_true", help="Use text prompts only; skip reference mode.")
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


def contour_crop_from_mask(image: Image.Image, mask, padding: float = 0.08) -> Image.Image | None:
    """Crop image to the mask contour (transparent outside). Returns RGBA PIL Image or None if mask empty.
    Uses continuous mask values for alpha so edges are soft, not hard rectangles."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask).squeeze().astype(np.float32)
    if mask.ndim != 2 or mask.size == 0:
        return None
    # Clip to [0,1] in case of logits or out-of-range
    mask = np.clip(mask, 0.0, 1.0)
    h, w = mask.shape
    if image.size != (w, h):
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.LANCZOS)) / 255.0
        h, w = mask.shape
    # Tight bbox from foreground (low threshold to include soft edges)
    binary = mask > 0.2
    ys = np.any(binary, axis=1)
    xs = np.any(binary, axis=0)
    y_where = np.where(ys)[0]
    x_where = np.where(xs)[0]
    if len(y_where) == 0 or len(x_where) == 0:
        return None
    y0, y1 = int(y_where[0]), int(y_where[-1]) + 1
    x0, x1 = int(x_where[0]), int(x_where[-1]) + 1
    bw, bh = x1 - x0, y1 - y0
    dx = max(1, int(bw * padding))
    dy = max(1, int(bh * padding))
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w, x1 + dx)
    y1 = min(h, y1 + dy)
    img_crop = np.array(image.crop((x0, y0, x1, y1)))
    if img_crop.ndim == 2:
        img_crop = np.stack([img_crop] * 3, axis=-1)
    mask_crop = mask[y0:y1, x0:x1].astype(np.float32)
    if mask_crop.shape[:2] != img_crop.shape[:2]:
        mask_crop = np.array(Image.fromarray((mask_crop * 255).astype(np.uint8)).resize((img_crop.shape[1], img_crop.shape[0]), Image.LANCZOS)) / 255.0
    # Continuous alpha: soft edges instead of hard rectangle
    alpha = (np.clip(mask_crop, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([img_crop[..., 0], img_crop[..., 1], img_crop[..., 2], alpha])
    return Image.fromarray(rgba, "RGBA")


def main():
    args = parse_args()
    device = get_device(args)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_reference_crop = False
    reference_crop_paths: list[Path] = []

    # Legacy: single file
    if args.reference_crop is not None:
        ref_path = args.reference_crop.resolve()
        if not ref_path.is_file():
            print(f"Error: reference crop not found: {ref_path}", file=sys.stderr)
            sys.exit(1)
        reference_crop_paths = [ref_path]
        use_reference_crop = True
        print(f"Using reference-crop mode (no text prompt): {ref_path.name}")
    elif not args.no_reference and args.reference_crops_dir is not None:
        ref_dir = args.reference_crops_dir.resolve()
        if not ref_dir.is_dir():
            print(
                f"Error: reference crops dir not found: {ref_dir}\n"
                "Create it and add crop images (e.g. from train), or run without reference mode (text prompts).",
                file=sys.stderr,
            )
            sys.exit(1)
        all_refs = sorted(
            p for p in ref_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not all_refs:
            print(
                f"Error: no images in reference crops dir: {ref_dir}\n"
                "Add crop images there, or run without reference mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        n = min(MAX_REFERENCE_CROP_EXAMPLES, len(all_refs))
        reference_crop_paths = random.sample(all_refs, n)
        use_reference_crop = True
        print(f"Using reference-crops mode ({len(reference_crop_paths)} examples from {ref_dir}):")
        for p in reference_crop_paths:
            print(f"  - {p.name}")

    if use_reference_crop:
        prompt_list = ["reference"]  # single pass, slug for filenames
    elif args.prompts is not None:
        prompt_list = [p.strip() for p in args.prompts.split(",") if p.strip()]
    elif args.single:
        prompt_list = [args.prompt]
    else:
        prompt_list = list(DEFAULT_PROMPTS)
    if len(prompt_list) > 1:
        print(f"Running with {len(prompt_list)} prompts: {prompt_list}")

    image_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
        print(f"Dry run: processing only {len(image_paths)} image(s) (--limit={args.limit})")

    print(f"Loading SAM 3 model on device={device}...")
    build_sam3_image_model, Sam3Processor = _load_sam3()
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.min_score)
    print("Model loaded.")

    total_crops = 0
    skipped = 0
    for prompt in prompt_list:
        prompt_slug = prompt_to_slug(prompt)
        if len(prompt_list) > 1:
            print(f"\n--- prompt: {prompt!r} (slug: {prompt_slug}) ---")
        for path in image_paths:
            try:
                stem = "_".join(path.relative_to(input_dir).with_suffix("").parts)
            except ValueError:
                stem = path.stem
            if args.resume:
                existing = list(output_dir.glob(f"{stem}_crop_*_{prompt_slug}.png"))
                if not existing and len(prompt_list) == 1:
                    existing = list(output_dir.glob(f"{stem}_crop_*.png"))
                if existing:
                    skipped += 1
                    continue
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Skip {path}: {e}", file=sys.stderr)
                continue
            w, h = image.size
            state = processor.set_image(image)
            if use_reference_crop:
                # Full-image box in normalized [cx, cy, w, h]; SAM3 segments main subject (no text).
                state = processor.add_geometric_prompt([0.5, 0.5, 1.0, 1.0], True, state)
            else:
                state = processor.set_text_prompt(prompt, state)
            boxes = state["boxes"]
            scores = state["scores"]
            masks = state.get("masks")  # [N, 1, H, W] or [N, H, W]
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
                image_area = w * h
                if image_area > 0 and (area / image_area) > args.max_area_ratio:
                    print(f"  {path.name} -> skip crop {i} (crop is {100*area/image_area:.0f}% of image, use --max-area-ratio 1.0 to allow)")
                    continue
                if masks is not None and i < masks.shape[0]:
                    m = masks[i].squeeze()
                    crop = contour_crop_from_mask(image, m, args.padding)
                else:
                    x0, y0, x1, y1 = expand_box(x0, y0, x1, y1, args.padding, w, h)
                    crop = image.crop((x0, y0, x1, y1)).convert("RGBA")
                if crop is None:
                    continue
                ext = ".png"
                out_name = f"{stem}_crop_{i}_{prompt_slug}{ext}"
                out_path = output_dir / out_name
                crop.save(out_path)
                total_crops += 1
                print(f"  {path.name} -> {out_name} (score={score:.2f})")

    if args.resume and skipped:
        print(f"Skipped {skipped} image(s) (already had crops).")
    print(f"Done. Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
