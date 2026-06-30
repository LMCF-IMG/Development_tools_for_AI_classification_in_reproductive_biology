"""
Export Fiji/ImageJ ROIs (.roi / .zip not needed here) to ellipse annotations (JSONL)
and QC overlays, for ellipse-based (StarDist-like) training.

Assumes file naming:
  IMAGE:  <base>_<COUNT>.tif
  ROI:    <base>_<COUNT>_E1.tif.roi, <base>_<COUNT>_E2.tif.roi, ...

Example:
  q121_t0__001_046h20m_FOC_4.tif
  q121_t0__001_046h20m_FOC_4_E1.tif.roi
  ...

Edit PARAMETERS below and run in VS Code.
"""
#%%
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tifffile as tiff
from roifile import ImagejRoi

from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

print("CWD:", Path.cwd())
print("Script dir:", script_dir)
print("OK:", Path.cwd() == script_dir)

#%%
# =========================
# PARAMETERS (edit these)
# =========================

IMAGES_DIR = script_dir / "central_only"
ROIS_DIR = script_dir / "rois"
OUT_JSONL = script_dir / "jsonl" / "annotations.jsonl"   # musí být soubor, ne jen složka
WRITE_OVERLAYS = True
OVERLAY_DIR = script_dir / "overlays"

# How many overlays to write (None = all)
MAX_OVERLAYS: Optional[int] = None

# If True, require that number of ROIs == COUNT parsed from filename
STRICT_ROI_COUNT_MATCH = True

# Overlay drawing settings
ELLIPSE_THICKNESS = 2
CENTER_MARK_SIZE = 6
FONT_SCALE = 0.5

# =========================

#%%
@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float      # semi-axis (pixels)
    b: float      # semi-axis (pixels)
    theta: float  # radians, rotation of major axis (OpenCV convention mapped to radians)


COUNT_REGEX = re.compile(r"_(\d+)\.tif$", re.IGNORECASE)


def parse_count_from_image_name(image_name: str) -> int:
    """
    Parse last number before '.tif' as count, e.g. "..._4.tif" -> 4
    """
    m = COUNT_REGEX.search(image_name)
    if not m:
        raise ValueError(f"Cannot parse count from image filename: {image_name}")
    return int(m.group(1))


def list_roi_files_for_image(image_path: Path) -> List[Path]:
    """
    Find ROI files matching <image>.replace('.tif', '_E*.tif.roi') in ROIS_DIR.
    """
    stem = image_path.name  # includes .tif
    # ROI filenames: <image>.tif + _E*.tif.roi is not quite; your examples are: <image>_E1.tif.roi
    # So we match: image_path.name + "_E*.tif.roi" after removing the ".tif" suffix.
    base = re.sub(r"\.tif$", "", image_path.name, flags=re.IGNORECASE)
    pattern = f"{base}_E*.tif.roi"
    rois = sorted(ROIS_DIR.glob(pattern))
    return rois


def roi_to_ellipse(roi: ImagejRoi) -> Ellipse:
    """
    Convert an ImageJ ROI to an ellipse.
    Strategy:
      1) If ROI is an OVAL-type ROI (axis-aligned), use bounding box -> ellipse with theta=0.
      2) Otherwise, try roi.coordinates() -> fit ellipse with cv2.fitEllipse.

    This handles mixed "oval" vs "ellipse/freehand-ellipse" ROI types robustly.
    """
    # Try to use coordinates for everything first; for some OVAL ROIs coordinates may be sparse.
    coords = None
    try:
        coords = roi.coordinates()
    except Exception:
        coords = None

    if coords is not None:
        pts = np.asarray(coords)
        # roifile may return (row, col) or (x, y) depending on ROI;
        # empirically in many cases it's (y, x). We'll detect by range:
        # We'll assume image space is positive; without image shape we can’t be perfect,
        # so we simply treat the first column as x and second as y if that seems plausible.
        # Better: rely on cv2.fitEllipse needing (x,y) points.
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 5:
            # Heuristic: If most values in first column are integers and second too, proceed.
            # We'll attempt both interpretations and choose the one that yields a sensible ellipse.
            ell = _fit_ellipse_try_both(pts)
            if ell is not None:
                return ell

    # Fallback for axis-aligned oval: use bounding box
    # roi.left/top/width/height exist for many ROI types
    left = getattr(roi, "left", None)
    top = getattr(roi, "top", None)
    width = getattr(roi, "width", None)
    height = getattr(roi, "height", None)
    if None not in (left, top, width, height) and width > 0 and height > 0:
        cx = float(left) + float(width) / 2.0
        cy = float(top) + float(height) / 2.0
        a = float(width) / 2.0
        b = float(height) / 2.0
        theta = 0.0
        return Ellipse(cx=cx, cy=cy, a=a, b=b, theta=theta)

    raise ValueError(f"Unable to convert ROI to ellipse (type={getattr(roi, 'roitype', 'unknown')})")


def _fit_ellipse_try_both(pts: np.ndarray) -> Optional[Ellipse]:
    """
    Try fitting ellipse treating pts as (x,y) and as (y,x), choose plausible result.
    """
    def fit(pxy: np.ndarray) -> Optional[Ellipse]:
        p = pxy.astype(np.float32).reshape(-1, 1, 2)
        try:
            ((cx, cy), (MA, ma), angle_deg) = cv2.fitEllipse(p)  # MA,ma are diameters
        except Exception:
            return None

        a = float(MA) / 2.0
        b = float(ma) / 2.0
        # normalize so that a >= b (major axis first)
        # OpenCV’s angle corresponds to the rotation of the ellipse’s major axis
        theta = math.radians(float(angle_deg))

        # Plausibility checks: semi-axes positive and not crazy small
        if not (a > 0 and b > 0):
            return None
        if a < 1 or b < 1:
            return None

        # Ensure a >= b
        if b > a:
            a, b = b, a
            theta = (theta + math.pi / 2.0) % math.pi

        return Ellipse(cx=float(cx), cy=float(cy), a=a, b=b, theta=theta)

    ell_xy = fit(pts)              # treat as (x,y)
    ell_yx = fit(pts[:, ::-1])     # treat as (y,x) -> swap columns to (x,y)

    # If only one works, take it
    if ell_xy is None and ell_yx is None:
        return None
    if ell_xy is None:
        return ell_yx
    if ell_yx is None:
        return ell_xy

    # If both work, choose the one with smaller "weirdness":
    # Prefer ellipse whose center is closer to median of points.
    med = np.median(pts, axis=0)
    # compute distances
    d_xy = (ell_xy.cx - med[0]) ** 2 + (ell_xy.cy - med[1]) ** 2
    d_yx = (ell_yx.cx - med[0]) ** 2 + (ell_yx.cy - med[1]) ** 2
    return ell_xy if d_xy <= d_yx else ell_yx


def draw_overlay(gray: np.ndarray, ellipses: List[Ellipse], labels: List[str]) -> np.ndarray:
    """
    Draw ellipses + centers + labels over the image.
    Input gray is 2D uint8.
    Output is BGR uint8.
    """
    if gray.dtype != np.uint8:
        # Convert to 8-bit for display
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g = gray
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    for e, lab in zip(ellipses, labels):
        center = (int(round(e.cx)), int(round(e.cy)))
        axes = (int(round(e.a)), int(round(e.b)))
        angle_deg = math.degrees(e.theta)

        cv2.ellipse(bgr, center, axes, angle_deg, 0, 360, (0, 255, 0), ELLIPSE_THICKNESS)
        cv2.drawMarker(bgr, center, (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=CENTER_MARK_SIZE, thickness=2)

        cv2.putText(bgr, lab, (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 0), 1, cv2.LINE_AA)

    return bgr

#%%
def main() -> None:
    if WRITE_OVERLAYS:
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_DIR.glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No .tif images found in {IMAGES_DIR}")

    wrote_overlays = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for img_path in images:
            count = parse_count_from_image_name(img_path.name)
            roi_paths = list_roi_files_for_image(img_path)

            if STRICT_ROI_COUNT_MATCH and len(roi_paths) != count:
                raise ValueError(
                    f"ROI count mismatch for {img_path.name}: "
                    f"parsed count={count} but found {len(roi_paths)} ROI files"
                )

            # Load image (8-bit tif expected)
            img = tiff.imread(str(img_path))
            if img.ndim != 2:
                raise ValueError(f"Expected 2D grayscale TIFF: {img_path.name}, got shape {img.shape}")

            ellipses: List[Ellipse] = []
            labels: List[str] = []

            for rp in roi_paths:
                roi = ImagejRoi.fromfile(str(rp))
                e = roi_to_ellipse(roi)
                ellipses.append(e)

                # label from E-suffix if present
                m = re.search(r"_E(\d+)\.tif\.roi$", rp.name, flags=re.IGNORECASE)
                lab = f"E{m.group(1)}" if m else rp.stem
                labels.append(lab)

            record = {
                "image": img_path.name,
                "count": count,
                "ellipses": [
                    {"cx": e.cx, "cy": e.cy, "a": e.a, "b": e.b, "theta": e.theta}
                    for e in ellipses
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if WRITE_OVERLAYS and (MAX_OVERLAYS is None or wrote_overlays < MAX_OVERLAYS):
                overlay = draw_overlay(img.astype(np.uint8), ellipses, labels)
                out_png = OVERLAY_DIR / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(out_png), overlay)
                wrote_overlays += 1

    print(f"Done. Wrote JSONL: {OUT_JSONL}")
    if WRITE_OVERLAYS:
        print(f"Wrote overlays to: {OVERLAY_DIR} (count={wrote_overlays})")

#%%
if __name__ == "__main__":
    main()
# %%
