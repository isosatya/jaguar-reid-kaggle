#!/usr/bin/env python3
"""
Crop raw jaguar images using Grounded SAM: Grounding DINO (detection) + SAM2 (mask).

Outputs contour crops (jaguar shape, transparent background) as PNG, not rectangular frames.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    p = argparse.ArgumentParser(description="Crop jaguar images with Grounded SAM (Grounding DINO + SAM2).")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw_gallery/jaguars_images"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed_gallery_grounded_sam"))
    p.add_argument("--prompts", type=str, default=None)
    p.add_argument("--grounding-model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-model", type=str, default="facebook/sam2.1-hiera-small")
    p.add_argument("--min-score", type=float, default=0.28)
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--padding", type=float, default=0.08)
    p.add_argument("--max-area-ratio", type=float, default=0.90)
    p.add_argument("--max-fill-ratio", type=float, default=0.92, help="Skip when best mask fill ratio > this (1.0=allow squares).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument(
        "--no-sam-refine",
        action="store_true",
        help="Skip SAM2 refinement; only use Grounding DINO boxes (faster, less precise).",
    )
    return p.parse_args()


def mask_to_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) from a binary mask [H,W]. Returns None if mask is empty."""
    if mask.numel() == 0:
        return None
    ys = torch.any(mask, dim=1)
    xs = torch.any(mask, dim=0)
    y_where = torch.where(ys)[0]
    x_where = torch.where(xs)[0]
    if len(y_where) == 0 or len(x_where) == 0:
        return None
    y0, y1 = int(y_where[0].item()), int(y_where[-1].item()) + 1
    x0, x1 = int(x_where[0].item()), int(x_where[-1].item()) + 1
    return (x0, y0, x1, y1)


def pick_contour_mask(masks: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    """From [num_masks, H, W], pick the mask with lowest fill ratio (least rectangular).
    Returns (mask, fill_ratio). fill_ratio=1 means full rectangle."""
    if masks.dim() == 4:
        masks = masks[0]
    if masks.dim() == 2:
        return masks, 1.0
    if masks.dim() != 3 or masks.shape[0] == 0:
        m = masks[0] if masks.numel() > 0 else None
        return m, 1.0 if m is not None else 1.0
    n = masks.shape[0]
    best_idx = 0
    best_ratio = 1.0
    for i in range(n):
        m = masks[i] > 0.5
        area = m.float().sum().item()
        if area < 100:
            continue
        ys = torch.any(m, dim=1)
        xs = torch.any(m, dim=0)
        y_where = torch.where(ys)[0]
        x_where = torch.where(xs)[0]
        if len(y_where) == 0 or len(x_where) == 0:
            continue
        bw = int(x_where[-1].item() - x_where[0].item()) + 1
        bh = int(y_where[-1].item() - y_where[0].item()) + 1
        bbox_area = bw * bh
        if bbox_area <= 0:
            continue
        ratio = area / bbox_area
        if ratio < best_ratio:
            best_ratio = ratio
            best_idx = i
    return masks[best_idx], best_ratio


def contour_crop_from_mask(image: Image.Image, mask, padding: float = 0.08) -> Image.Image | None:
    """Crop image to the mask contour (transparent outside). Uses continuous mask for soft edges."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask).squeeze().astype(np.float32)
    if mask.ndim != 2 or mask.size == 0:
        return None
    mask = np.clip(mask, 0.0, 1.0)
    h, w = mask.shape
    if image.size != (w, h):
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.LANCZOS)) / 255.0
        h, w = mask.shape
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
    alpha = (np.clip(mask_crop, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([img_crop[..., 0], img_crop[..., 1], img_crop[..., 2], alpha])
    return Image.fromarray(rgba, "RGBA")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.allow_cpu else "cpu"
    if device == "cpu":
        print("Warning: CPU mode (slow).", file=sys.stderr)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_prompts = [p.strip() for p in args.prompts.split(",")] if args.prompts else list(DEFAULT_TEXT_PROMPTS)
    # Processor expects list of strings (one per image); phrases in one string separated by " . "
    text_for_batch = [" . ".join(p.strip() for p in text_prompts)]

    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        image_paths = image_paths[: args.limit]
        print(f"Limit: {len(image_paths)} images")

    print(f"Loading Grounding DINO {args.grounding_model}...")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    gd_processor = AutoProcessor.from_pretrained(args.grounding_model)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)

    sam_processor = None
    sam_model = None
    if not args.no_sam_refine:
        print(f"Loading SAM2 {args.sam_model}...")
        try:
            from transformers import Sam2Model, Sam2Processor as Sam2ProcessorHF
            sam_processor = Sam2ProcessorHF.from_pretrained(args.sam_model)
            sam_model = Sam2Model.from_pretrained(args.sam_model).to(device)
        except Exception as e:
            print(f"Warning: SAM2 load failed ({e}). Using Grounding DINO boxes only.", file=sys.stderr)
            args.no_sam_refine = True
    print("Models loaded.")

    total_crops = 0
    skipped = 0
    for path in image_paths:
        try:
            stem = "_".join(path.relative_to(input_dir).with_suffix("").parts)
        except ValueError:
            stem = path.stem
        if args.resume and list(output_dir.glob(f"{stem}_crop_*.png")):
            skipped += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue
        w, h = image.size

        # 1) Grounding DINO
        inputs_gd = gd_processor(images=image, text=text_for_batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_gd = gd_model(**inputs_gd)
        results = gd_processor.post_process_grounded_object_detection(
            outputs_gd,
            inputs_gd["input_ids"],
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
            if w * h > 0 and (area / (w * h)) > args.max_area_ratio:
                continue

            # 2) Optionally refine with SAM2 and save contour crop
            if sam_processor is not None and sam_model is not None:
                try:
                    input_boxes = [[[x0, y0, x1, y1]]]
                    inputs_sam = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs_sam = sam_model(**inputs_sam, multimask_output=True)
                    masks = sam_processor.post_process_masks(
                        outputs_sam.pred_masks.cpu(),
                        inputs_sam["original_sizes"],
                    )[0]
                    if masks.numel() > 0:
                        m, fill_ratio = pick_contour_mask(masks)
                        if m is None:
                            m = masks[0, 0] if masks.dim() == 4 else masks[0]
                        if fill_ratio > args.max_fill_ratio:
                            continue
                        m = m.numpy().astype(np.float32)
                        m = np.clip(m, 0.0, 1.0)
                        crop = contour_crop_from_mask(image, m, args.padding)
                        if crop is not None:
                            label_slug = "jaguar"
                            if labels is not None and i < len(labels):
                                try:
                                    lab = labels[i]
                                    label_slug = prompt_to_slug(lab) if isinstance(lab, str) else f"class{lab}"
                                except Exception:
                                    pass
                            out_name = f"{stem}_crop_{saved}_{label_slug}.png"
                            crop.save(output_dir / out_name)
                            total_crops += 1
                            saved += 1
                            print(f"  {path.name} -> {out_name} (score={score:.2f})")
                            continue
                except Exception as e:
                    pass  # fallback to box crop below if SAM fails

            x0, y0, x1, y1 = expand_box(float(x0), float(y0), float(x1), float(y1), args.padding, w, h)
            crop = image.crop((x0, y0, x1, y1)).convert("RGBA")
            label_slug = "jaguar"
            if labels is not None and i < len(labels):
                try:
                    lab = labels[i]
                    label_slug = prompt_to_slug(lab) if isinstance(lab, str) else f"class{lab}"
                except Exception:
                    pass
            out_name = f"{stem}_crop_{saved}_{label_slug}.png"
            crop.save(output_dir / out_name)
            total_crops += 1
            saved += 1
            print(f"  {path.name} -> {out_name} (score={score:.2f})")

    if skipped:
        print(f"Skipped {skipped} (resume).")
    print(f"Done. Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
