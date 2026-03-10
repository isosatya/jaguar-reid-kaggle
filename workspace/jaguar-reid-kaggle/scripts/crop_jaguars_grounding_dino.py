#!/usr/bin/env python3
"""
Crop raw jaguar images using Grounding DINO + SAM2 (Hugging Face).

Gets bounding boxes from Grounding DINO (text prompts), then refines each with SAM2
to obtain a segmentation mask. Saves contour crops (jaguar shape, transparent
background) as PNG, not rectangular frames.

Usage:
    python scripts/crop_jaguars_grounding_dino.py
    python scripts/crop_jaguars_grounding_dino.py --limit 5 --resume
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
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


def pick_contour_mask(masks: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    """From [num_masks, H, W] or [1, num_masks, H, W], pick the mask with lowest fill ratio. Returns (mask, ratio)."""
    if masks.dim() == 4:
        masks = masks[0]
    if masks.dim() == 2:
        return masks, 1.0
    if masks.dim() != 3 or masks.shape[0] == 0:
        return (masks[0] if masks.numel() > 0 else None), 1.0
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


def contour_crop_from_mask(image: Image.Image, mask, padding: float = 0.08, crisp_contour: bool = False, crisp_threshold: float = 0.5, exclude_green: bool = False) -> Image.Image | None:
    """Crop image to the mask contour (transparent outside). crisp_contour=False: soft edges; True: hard contour."""
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
    if exclude_green:
        img_arr = np.array(image)
        if img_arr.ndim >= 3 and img_arr.shape[-1] >= 3:
            r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
            green_strong = (g > r) & (g > b) & (g > 80)
            uncertain = (mask > 0.2) & (mask < 0.8)
            mask = np.where(green_strong & uncertain, 0.0, mask).astype(np.float32)
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
    if crisp_contour:
        alpha = (np.where(mask_crop > crisp_threshold, 255, 0)).astype(np.uint8)
    else:
        alpha = (np.clip(mask_crop, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([img_crop[..., 0], img_crop[..., 1], img_crop[..., 2], alpha])
    return Image.fromarray(rgba, "RGBA")


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
    p.add_argument("--max-fill-ratio", type=float, default=0.92, help="Skip when best mask fill ratio > this (1.0=allow squares).")
    p.add_argument("--crisp-contour", action="store_true", help="Binary alpha for hard contour (default: soft edges).")
    p.add_argument("--crisp-threshold", type=float, default=0.5, help="With --crisp-contour: mask > this is inside (0.6-0.7 = tighter).")
    p.add_argument("--exclude-green", action="store_true", help="Zero mask in green vegetation zones at uncertain boundaries.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--sam-model", type=str, default="facebook/sam2.1-hiera-small", help="SAM2 model for mask (contour) crops.")
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

    # Processor expects text as list of strings (one per image); phrases in one string separated by " . "
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
        print(f"Limit: processing {len(image_paths)} images")

    print(f"Loading Grounding DINO {args.model_id} on {device}...")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    print(f"Loading SAM2 {args.sam_model} for contour masks...")
    try:
        from transformers import Sam2Model, Sam2Processor as Sam2ProcessorHF
        sam_processor = Sam2ProcessorHF.from_pretrained(args.sam_model)
        sam_model = Sam2Model.from_pretrained(args.sam_model).to(device)
    except Exception as e:
        print(f"Error: SAM2 required for contour crops. {e}", file=sys.stderr)
        sys.exit(1)
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

        inputs = processor(images=image, text=text_for_batch, return_tensors="pt").to(device)
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
            if w * h > 0 and (area / (w * h)) > args.max_area_ratio:
                continue

            input_boxes = [[[x0, y0, x1, y1]]]
            cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
            input_points, input_labels = [[[[cx, cy]]]], [[[1]]]
            try:
                inputs_sam = sam_processor(
                    images=image,
                    input_boxes=input_boxes,
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt",
                ).to(device)
            except Exception:
                inputs_sam = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_sam = sam_model(**inputs_sam, multimask_output=True)
            masks = sam_processor.post_process_masks(
                outputs_sam.pred_masks.cpu(),
                inputs_sam["original_sizes"],
            )[0]
            if masks.numel() == 0:
                continue
            m, fill_ratio = pick_contour_mask(masks)
            if m is None:
                m = masks[0, 0] if masks.dim() == 4 else masks[0]
            if fill_ratio > args.max_fill_ratio:
                continue
            m = m.numpy().astype(np.float32)
            m = np.clip(m, 0.0, 1.0)
            crop = contour_crop_from_mask(image, m, args.padding, crisp_contour=args.crisp_contour, crisp_threshold=getattr(args, "crisp_threshold", 0.5), exclude_green=getattr(args, "exclude_green", False))
            if crop is None:
                continue

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
            out_name = f"{stem}_crop_{saved}_{label_slug}.png"
            crop.save(output_dir / out_name)
            total_crops += 1
            saved += 1
            print(f"  {path.name} -> {out_name} (score={score:.2f})")

    if skipped:
        print(f"Skipped {skipped} (already had crops).")
    print(f"Done. Saved {total_crops} crops to {output_dir}")


if __name__ == "__main__":
    main()
