# M9_full_export_v2 — finální inference pro nové neanotované obrazy 
#%% M9_full_report_v2 — finální inference pro nové obrazy
# Výstup je do JEDNOHO společného adresáře a navíc vytváří souhrnné reporty.
#
# Pro každý vstupní obraz ukládá:
#   <image_stem>__orig.png
#   <image_stem>__fitted_ellipses_overlay.png
#   <image_stem>__instance_labels_color.png
#   <image_stem>__instance_labels_overlay.png
#   <image_stem>__fitted_ellipses.json
#
# Globálně ukládá:
#   summary.json
#   summary.csv
#   evaluation.json
#   evaluation_report.txt
#   confusion_matrix_classifier.png
#   confusion_matrix_exported.png
#   confusion_matrix_raw_candidates.png
#   count_mismatch_classifier.txt
#   count_mismatch_exported.txt
#   count_mismatch_raw_candidates.txt
#
# Poznámka:
#   - confusion matrix se počítá jen pro obrazy, kde lze z názvu vyčíst expected K
#   - Dice/IoU bez GT jsou zde SELF-CONSISTENCY metriky:
#       exportovaná maska vs maska rasterizovaná z fitted ellipse
#     => nejsou to GT metriky, ale kvalita fitu elipsy vůči segmentaci

from pathlib import Path
import csv
import json
import math
import re
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# =========================================================
# CESTY A NASTAVENÍ
# =========================================================
script_dir = Path(__file__).resolve().parent

SEG_CHECKPOINT_PATH = script_dir / "m6_train_val_aug" / "best_model_by_val_dice.pt"
COUNT_CHECKPOINT_PATH = script_dir / "best.pt"

# adresář s novými obrazy
INPUT_DIR = script_dir / "crops_used_for_inference_ellipses"

# JEDEN společný výstupní adresář
OUT_DIR = script_dir / "m9_full_report_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SLOTS = 4
PROB_THRESH = 0.5

# filtry kandidátů
ACTIVE_SLOT_MIN_MAXPROB = 0.75
ACTIVE_SLOT_MIN_PROBSUM = 800.0
MIN_COMPONENT_AREA_LOCAL = 250

COUNT_CLASSES = [1, 2, 4]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# pokud chceš jen subset, nastav integer; jinak None
MAX_IMAGES = None


# =========================================================
# SEGMENTAČNÍ MODEL
# =========================================================
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch + skip_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = conv_bn_relu(out_ch, out_ch, k=3, s=1, p=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetUNet18(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.input_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            self.input_conv.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.center = nn.Sequential(
            conv_bn_relu(512, 512, k=3, s=1, p=1),
            conv_bn_relu(512, 512, k=3, s=1, p=1),
        )

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.input_conv(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        z = self.center(x4)
        z = self.dec4(z, x3)
        z = self.dec3(z, x2)
        z = self.dec2(z, x1)
        z = self.dec1(z, x0)

        logits = self.final_conv(z)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# =========================================================
# COUNT MODEL
# =========================================================
def build_count_model(arch: str, n_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def build_count_transform(image_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])


# =========================================================
# IO
# =========================================================
def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nelze načíst obrázek: {path}")
    return img


def find_input_images(input_dir: Path):
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]
    return files


def parse_expected_k_from_filename(filename: str):
    """
    Hledá suffix _1 / _2 / _4 před příponou.
    Když nenajde, vrátí None.
    """
    m = re.search(r"_([124])\.(tif|tiff|png|jpg|jpeg|bmp)$", filename.lower())
    if m is None:
        return None
    return int(m.group(1))


# =========================================================
# POSTPROCESSING
# =========================================================
def keep_largest_component(mask_u8: np.ndarray, min_area: int = 150) -> np.ndarray:
    mask = (mask_u8 > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    best_label = None
    best_area = -1
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = lab

    if best_label is None or best_area < min_area:
        return np.zeros_like(mask, dtype=np.uint8)

    return (labels == best_label).astype(np.uint8)


def mask_to_fit_ellipse(mask_u8: np.ndarray):
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    if cv2.contourArea(cnt) < MIN_COMPONENT_AREA_LOCAL:
        return None

    (cx, cy), (major_d, minor_d), angle_deg = cv2.fitEllipse(cnt)

    a = float(major_d) * 0.5
    b = float(minor_d) * 0.5

    if b > a:
        a, b = b, a
        angle_deg += 90.0

    return {
        "cx": float(cx),
        "cy": float(cy),
        "a": float(a),
        "b": float(b),
        "theta": float(angle_deg),
        "area_px": int(mask_u8.sum()),
    }


# =========================================================
# INSTANČNÍ LABEL IMAGE
# =========================================================
def make_instance_label_image(pred_masks_4hw: np.ndarray) -> np.ndarray:
    h, w = pred_masks_4hw.shape[1:]
    label_img = np.zeros((h, w), dtype=np.uint8)

    for s in range(pred_masks_4hw.shape[0]):
        mask = pred_masks_4hw[s] > 0
        label_img[mask] = s + 1

    return label_img


def label_to_color_image(label_img: np.ndarray) -> np.ndarray:
    lut = np.array([
        [0,   0,   0],     # 0 background
        [0,   0, 255],     # 1 red
        [0, 255,   0],     # 2 green
        [255, 0,   0],     # 3 blue
        [0, 255, 255],     # 4 yellow
    ], dtype=np.uint8)

    return lut[np.clip(label_img, 0, 4)]


def make_instance_overlay(gray: np.ndarray, label_img: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = label_to_color_image(label_img)

    overlay = base.copy()
    fg = label_img > 0
    overlay[fg] = cv2.addWeighted(base[fg], 1.0 - alpha, color[fg], alpha, 0.0)

    for lab in [1, 2, 3, 4]:
        ys, xs = np.where(label_img == lab)
        if len(xs) == 0:
            continue
        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))
        cv2.putText(
            overlay,
            str(lab),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


# =========================================================
# ELIPSA: vykreslení + rasterizace
# =========================================================
def ellipse_to_dense_points(cx, cy, a, b, theta_deg, n=360):
    if abs(theta_deg) <= 2 * math.pi + 1e-6:
        theta_deg = math.degrees(theta_deg)

    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float32)
    th = np.deg2rad(theta_deg).astype(np.float32)

    ct = np.cos(t)
    st = np.sin(t)
    c = np.cos(th)
    s = np.sin(th)

    x = cx + a * ct * c - b * st * s
    y = cy + a * ct * s + b * st * c

    return np.stack([x, y], axis=1).astype(np.float32)


def ellipse_to_mask(shape_hw, ellipse_dict, n=360):
    h, w = shape_hw
    pts = ellipse_to_dense_points(
        ellipse_dict["cx"],
        ellipse_dict["cy"],
        ellipse_dict["a"],
        ellipse_dict["b"],
        ellipse_dict["theta"],
        n=n,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], color=255, lineType=cv2.LINE_8)
    return (mask > 0).astype(np.uint8)


def draw_fitted_ellipse_overlay(gray: np.ndarray, fitted_ellipses: list[dict], title: str = "") -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for i, e in enumerate(fitted_ellipses, start=1):
        pts = ellipse_to_dense_points(
            e["cx"], e["cy"], e["a"], e["b"], e["theta"], n=360
        )
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        cv2.polylines(canvas, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        cxy = (int(round(e["cx"])), int(round(e["cy"])))
        cv2.circle(canvas, cxy, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(i),
            (cxy[0] + 3, cxy[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            canvas,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


# =========================================================
# METRIKY
# =========================================================
def dice_score(pred_u8: np.ndarray, gt_u8: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_u8.astype(np.float32)
    gt = gt_u8.astype(np.float32)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (denom + eps)


def iou_score(pred_u8: np.ndarray, gt_u8: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_u8.astype(np.float32)
    gt = gt_u8.astype(np.float32)
    inter = float((pred * gt).sum())
    union = float(((pred + gt) > 0).sum())
    return (inter + eps) / (union + eps)


def confusion_matrix_counts(y_true: list[int], y_pred: list[int], classes: list[int]):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_report_from_cm(cm: np.ndarray, classes: list[int]):
    total = int(cm.sum())
    diag = np.diag(cm)

    accuracy = float(diag.sum() / total) if total > 0 else 0.0

    per_class = {}
    recalls = []

    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        recalls.append(recall)

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
    }


def save_confusion_matrix_figure(cm: np.ndarray, classes: list[int], out_path: Path, title: str):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted K")
    ax.set_ylabel("Expected K")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================
# COUNT INFERENCE
# =========================================================
@torch.no_grad()
def run_count_classifier(model: nn.Module, gray: np.ndarray, tfm, classes: list[int]):
    x = tfm(gray).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    k_pred = int(classes[pred_idx])

    return k_pred, probs.tolist()


# =========================================================
# SEGMENTATION INFERENCE
# =========================================================
@torch.no_grad()
def run_segmentation_topk(model: nn.Module, gray: np.ndarray, k_selected: int):
    x = gray.astype(np.float32) / 255.0
    x = torch.from_numpy(x[None, None, :, :]).to(DEVICE)

    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # 4,H,W
    raw_bin_masks = (probs >= PROB_THRESH).astype(np.uint8)

    candidates = []
    for s in range(MAX_SLOTS):
        cleaned = keep_largest_component(raw_bin_masks[s], min_area=MIN_COMPONENT_AREA_LOCAL)

        mask_area = int(cleaned.sum())
        prob_sum = float(probs[s].sum())
        max_prob = float(probs[s].max())

        if mask_area < MIN_COMPONENT_AREA_LOCAL:
            continue
        if max_prob < ACTIVE_SLOT_MIN_MAXPROB:
            continue
        if prob_sum < ACTIVE_SLOT_MIN_PROBSUM:
            continue

        fit_e = mask_to_fit_ellipse(cleaned)
        if fit_e is None:
            continue

        candidates.append({
            "orig_slot": int(s),
            "mask": cleaned,
            "prob": probs[s],
            "mask_area": mask_area,
            "prob_sum": prob_sum,
            "max_prob": max_prob,
            "ellipse": fit_e,
        })

    candidates = sorted(
        candidates,
        key=lambda d: (d["mask_area"], d["prob_sum"], d["max_prob"]),
        reverse=True,
    )

    selected = candidates[:k_selected]

    export_probs = np.zeros_like(probs, dtype=np.float32)
    export_masks = np.zeros_like(raw_bin_masks, dtype=np.uint8)
    fitted_ellipses = []

    for new_slot, item in enumerate(selected):
        export_probs[new_slot] = item["prob"]
        export_masks[new_slot] = item["mask"]

        e = dict(item["ellipse"])
        e["slot"] = int(new_slot)
        e["orig_slot"] = int(item["orig_slot"])
        e["mask_area"] = int(item["mask_area"])
        e["prob_sum"] = float(item["prob_sum"])
        e["max_prob"] = float(item["max_prob"])
        fitted_ellipses.append(e)

    return export_probs, export_masks, fitted_ellipses, len(candidates)


# =========================================================
# CSV EXPORT
# =========================================================
def row_to_csv_row(row: dict):
    out = {
        "image_name": row["image_name"],
        "image_path": row["image_path"],
        "k_expected_from_filename": row["k_expected_from_filename"],
        "k_classifier": row["k_classifier"],
        "classifier_prob_1": row["classifier_probs"][0] if len(row["classifier_probs"]) > 0 else None,
        "classifier_prob_2": row["classifier_probs"][1] if len(row["classifier_probs"]) > 1 else None,
        "classifier_prob_4": row["classifier_probs"][2] if len(row["classifier_probs"]) > 2 else None,
        "raw_candidate_count": row["raw_candidate_count"],
        "raw_candidate_mapped_k": row["raw_candidate_mapped_k"],
        "exported_topk_count": row["exported_topk_count"],
        "mean_fit_dice_mask_vs_ellipse": row["mean_fit_dice_mask_vs_ellipse"],
        "mean_fit_iou_mask_vs_ellipse": row["mean_fit_iou_mask_vs_ellipse"],
        "orig_path": row["orig_path"],
        "fitted_ellipses_overlay_path": row["fitted_ellipses_overlay_path"],
        "instance_labels_color_path": row["instance_labels_color_path"],
        "instance_labels_overlay_path": row["instance_labels_overlay_path"],
        "fitted_ellipses_json_path": row["fitted_ellipses_json_path"],
    }

    for slot in range(MAX_SLOTS):
        if slot < len(row["fitted_ellipses"]):
            e = row["fitted_ellipses"][slot]
            out[f"ellipse_{slot+1}_cx"] = e.get("cx")
            out[f"ellipse_{slot+1}_cy"] = e.get("cy")
            out[f"ellipse_{slot+1}_a"] = e.get("a")
            out[f"ellipse_{slot+1}_b"] = e.get("b")
            out[f"ellipse_{slot+1}_theta"] = e.get("theta")
            out[f"ellipse_{slot+1}_mask_area"] = e.get("mask_area")
            out[f"ellipse_{slot+1}_prob_sum"] = e.get("prob_sum")
            out[f"ellipse_{slot+1}_max_prob"] = e.get("max_prob")
            out[f"ellipse_{slot+1}_orig_slot"] = e.get("orig_slot")
            out[f"ellipse_{slot+1}_fit_dice"] = (
                row["slot_fit_dices_mask_vs_ellipse"][slot]
                if slot < len(row["slot_fit_dices_mask_vs_ellipse"]) else None
            )
            out[f"ellipse_{slot+1}_fit_iou"] = (
                row["slot_fit_ious_mask_vs_ellipse"][slot]
                if slot < len(row["slot_fit_ious_mask_vs_ellipse"]) else None
            )
        else:
            out[f"ellipse_{slot+1}_cx"] = None
            out[f"ellipse_{slot+1}_cy"] = None
            out[f"ellipse_{slot+1}_a"] = None
            out[f"ellipse_{slot+1}_b"] = None
            out[f"ellipse_{slot+1}_theta"] = None
            out[f"ellipse_{slot+1}_mask_area"] = None
            out[f"ellipse_{slot+1}_prob_sum"] = None
            out[f"ellipse_{slot+1}_max_prob"] = None
            out[f"ellipse_{slot+1}_orig_slot"] = None
            out[f"ellipse_{slot+1}_fit_dice"] = None
            out[f"ellipse_{slot+1}_fit_iou"] = None

    return out


# =========================================================
# MAIN
# =========================================================
print(f"DEVICE: {DEVICE}")

if not SEG_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Seg checkpoint not found: {SEG_CHECKPOINT_PATH}")
if not COUNT_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Count checkpoint not found: {COUNT_CHECKPOINT_PATH}")
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

# segmentation model
seg_model = ResNetUNet18(in_channels=1, out_channels=4).to(DEVICE)
seg_state = torch.load(SEG_CHECKPOINT_PATH, map_location=DEVICE)
seg_model.load_state_dict(seg_state)
seg_model.eval()

# count model
count_ckpt = torch.load(COUNT_CHECKPOINT_PATH, map_location=DEVICE)
count_classes = list(count_ckpt["classes"])
count_arch = str(count_ckpt["arch"])
count_image_size = int(count_ckpt["image_size"])

count_model = build_count_model(count_arch, n_classes=len(count_classes)).to(DEVICE)
count_model.load_state_dict(count_ckpt["model"])
count_model.eval()

count_tfm = build_count_transform(count_image_size)

print(f"Seg checkpoint:   {SEG_CHECKPOINT_PATH}")
print(f"Count checkpoint: {COUNT_CHECKPOINT_PATH}")
print(f"Count arch: {count_arch}")
print(f"Count classes: {count_classes}")
print(f"Count image_size: {count_image_size}")
print(f"Input dir: {INPUT_DIR}")

input_images = find_input_images(INPUT_DIR)
print(f"Found input images: {len(input_images)}")

summary_rows = []

# confusion matrix jen pro obrazy, kde je K v názvu
y_expected = []
y_classifier = []
y_exported = []
y_raw_mapped = []

count_mismatch_classifier = [
    "Images where classifier-predicted K does NOT match filename K",
    "=" * 80,
    "",
]
count_mismatch_exported = [
    "Images where exported instance count does NOT match filename K",
    "=" * 80,
    "",
]
count_mismatch_raw = [
    "Images where raw candidate count does NOT match filename K",
    "=" * 80,
    "",
]

# self-consistency metriky
ellipse_fit_all_dice = []
ellipse_fit_all_iou = []
ellipse_fit_per_k = defaultdict(lambda: {"dice": [], "iou": []})

for i, img_path in enumerate(input_images, start=1):
    gray = load_gray_image(img_path)
    h, w = gray.shape[:2]

    expected_k = parse_expected_k_from_filename(img_path.name)

    k_classifier, classifier_probs = run_count_classifier(
        count_model, gray, count_tfm, count_classes
    )

    pred_probs, pred_masks, fitted_ellipses, raw_candidate_count = run_segmentation_topk(
        seg_model, gray, k_selected=k_classifier
    )

    exported_count = len(fitted_ellipses)

    if raw_candidate_count <= 1:
        raw_mapped_k = 1
    elif raw_candidate_count == 2:
        raw_mapped_k = 2
    else:
        raw_mapped_k = 4

    # =====================================================
    # Self-consistency Dice/IoU: mask vs fitted ellipse
    # =====================================================
    slot_fit_dices = []
    slot_fit_ious = []

    for s in range(exported_count):
        pred_mask = pred_masks[s]
        ellipse_mask = ellipse_to_mask((h, w), fitted_ellipses[s], n=360)

        d = dice_score(pred_mask, ellipse_mask)
        j = iou_score(pred_mask, ellipse_mask)

        slot_fit_dices.append(float(d))
        slot_fit_ious.append(float(j))

    mean_fit_dice = float(np.mean(slot_fit_dices)) if slot_fit_dices else None
    mean_fit_iou = float(np.mean(slot_fit_ious)) if slot_fit_ious else None

    if mean_fit_dice is not None:
        ellipse_fit_all_dice.append(mean_fit_dice)
        ellipse_fit_all_iou.append(mean_fit_iou)

        if expected_k is not None:
            ellipse_fit_per_k[expected_k]["dice"].append(mean_fit_dice)
            ellipse_fit_per_k[expected_k]["iou"].append(mean_fit_iou)

    # =====================================================
    # Export obrazů
    # =====================================================
    stem = img_path.stem

    out_orig = OUT_DIR / f"{stem}__orig.png"
    cv2.imwrite(str(out_orig), gray)

    overlay = draw_fitted_ellipse_overlay(
        gray,
        fitted_ellipses,
        title=f"{img_path.name} | Kcls={k_classifier} | raw={raw_candidate_count}",
    )
    out_fit = OUT_DIR / f"{stem}__fitted_ellipses_overlay.png"
    cv2.imwrite(str(out_fit), overlay)

    instance_labels = make_instance_label_image(pred_masks)

    instance_labels_color = label_to_color_image(instance_labels)
    out_labels_color = OUT_DIR / f"{stem}__instance_labels_color.png"
    cv2.imwrite(str(out_labels_color), instance_labels_color)

    instance_overlay = make_instance_overlay(gray, instance_labels, alpha=0.45)
    out_labels_overlay = OUT_DIR / f"{stem}__instance_labels_overlay.png"
    cv2.imwrite(str(out_labels_overlay), instance_overlay)

    out_json = OUT_DIR / f"{stem}__fitted_ellipses.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(fitted_ellipses, f, indent=2, ensure_ascii=False)

    # =====================================================
    # confusion matrix inputs + mismatch logy
    # =====================================================
    if expected_k is not None:
        y_expected.append(expected_k)
        y_classifier.append(k_classifier)
        y_exported.append(exported_count if exported_count in COUNT_CLASSES else (4 if exported_count >= 3 else exported_count))
        y_raw_mapped.append(raw_mapped_k)

        if k_classifier != expected_k:
            count_mismatch_classifier.append(
                f"{img_path.name} | expected_from_filename={expected_k} | classifier_pred={k_classifier} | classifier_probs={classifier_probs}"
            )

        if exported_count != expected_k:
            count_mismatch_exported.append(
                f"{img_path.name} | expected_from_filename={expected_k} | exported_topk_count={exported_count}"
            )

        if raw_candidate_count != expected_k:
            count_mismatch_raw.append(
                f"{img_path.name} | expected_from_filename={expected_k} | raw_candidate_count={raw_candidate_count} | raw_mapped_class={raw_mapped_k}"
            )

    row = {
        "image_name": img_path.name,
        "image_path": str(img_path),
        "k_expected_from_filename": int(expected_k) if expected_k is not None else None,
        "k_classifier": int(k_classifier),
        "classifier_probs": [float(x) for x in classifier_probs],
        "raw_candidate_count": int(raw_candidate_count),
        "raw_candidate_mapped_k": int(raw_mapped_k),
        "exported_topk_count": int(exported_count),
        "fitted_ellipses": fitted_ellipses,
        "slot_fit_dices_mask_vs_ellipse": slot_fit_dices,
        "slot_fit_ious_mask_vs_ellipse": slot_fit_ious,
        "mean_fit_dice_mask_vs_ellipse": mean_fit_dice,
        "mean_fit_iou_mask_vs_ellipse": mean_fit_iou,
        "orig_path": str(out_orig),
        "fitted_ellipses_overlay_path": str(out_fit),
        "instance_labels_color_path": str(out_labels_color),
        "instance_labels_overlay_path": str(out_labels_overlay),
        "fitted_ellipses_json_path": str(out_json),
    }
    summary_rows.append(row)

    print(
        f"[{i:04d}/{len(input_images):04d}] {img_path.name} | "
        f"Kexp={expected_k} | Kcls={k_classifier} | raw_n={raw_candidate_count} | exported={exported_count} | "
        f"fitDice={mean_fit_dice}"
    )

# =========================================================
# SOUHRNY
# =========================================================
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, indent=2, ensure_ascii=False)

# summary.csv
csv_rows = [row_to_csv_row(r) for r in summary_rows]
if csv_rows:
    fieldnames = list(csv_rows[0].keys())
    with open(OUT_DIR / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

# confusion matrices
if y_expected:
    cm_classifier = confusion_matrix_counts(y_expected, y_classifier, COUNT_CLASSES)
    cm_exported = confusion_matrix_counts(y_expected, y_exported, COUNT_CLASSES)
    cm_raw = confusion_matrix_counts(y_expected, y_raw_mapped, COUNT_CLASSES)

    clf_report_classifier = classification_report_from_cm(cm_classifier, COUNT_CLASSES)
    clf_report_exported = classification_report_from_cm(cm_exported, COUNT_CLASSES)
    clf_report_raw = classification_report_from_cm(cm_raw, COUNT_CLASSES)

    save_confusion_matrix_figure(
        cm_classifier,
        COUNT_CLASSES,
        OUT_DIR / "confusion_matrix_classifier.png",
        title="Count confusion matrix — classifier vs filename K",
    )
    save_confusion_matrix_figure(
        cm_exported,
        COUNT_CLASSES,
        OUT_DIR / "confusion_matrix_exported.png",
        title="Count confusion matrix — exported count vs filename K",
    )
    save_confusion_matrix_figure(
        cm_raw,
        COUNT_CLASSES,
        OUT_DIR / "confusion_matrix_raw_candidates.png",
        title="Count confusion matrix — raw candidates vs filename K",
    )
else:
    cm_classifier = None
    cm_exported = None
    cm_raw = None
    clf_report_classifier = None
    clf_report_exported = None
    clf_report_raw = None

if len(count_mismatch_classifier) == 3:
    count_mismatch_classifier.append("No mismatches found.")
if len(count_mismatch_exported) == 3:
    count_mismatch_exported.append("No mismatches found.")
if len(count_mismatch_raw) == 3:
    count_mismatch_raw.append("No mismatches found.")

with open(OUT_DIR / "count_mismatch_classifier.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(count_mismatch_classifier))

with open(OUT_DIR / "count_mismatch_exported.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(count_mismatch_exported))

with open(OUT_DIR / "count_mismatch_raw_candidates.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(count_mismatch_raw))

# self-consistency souhrn
fit_metrics = {
    "mean_dice_all_mask_vs_ellipse": float(np.mean(ellipse_fit_all_dice)) if ellipse_fit_all_dice else None,
    "std_dice_all_mask_vs_ellipse": float(np.std(ellipse_fit_all_dice)) if ellipse_fit_all_dice else None,
    "mean_iou_all_mask_vs_ellipse": float(np.mean(ellipse_fit_all_iou)) if ellipse_fit_all_iou else None,
    "std_iou_all_mask_vs_ellipse": float(np.std(ellipse_fit_all_iou)) if ellipse_fit_all_iou else None,
    "per_k": {},
}

for k in sorted(ellipse_fit_per_k.keys()):
    fit_metrics["per_k"][k] = {
        "mean_dice": float(np.mean(ellipse_fit_per_k[k]["dice"])) if ellipse_fit_per_k[k]["dice"] else None,
        "std_dice": float(np.std(ellipse_fit_per_k[k]["dice"])) if ellipse_fit_per_k[k]["dice"] else None,
        "mean_iou": float(np.mean(ellipse_fit_per_k[k]["iou"])) if ellipse_fit_per_k[k]["iou"] else None,
        "std_iou": float(np.std(ellipse_fit_per_k[k]["iou"])) if ellipse_fit_per_k[k]["iou"] else None,
        "n": int(len(ellipse_fit_per_k[k]["dice"])),
    }

evaluation = {
    "count_classes": COUNT_CLASSES,
    "n_images_total": len(input_images),
    "n_images_with_expected_k_from_filename": len(y_expected),
    "confusion_matrix_classifier": cm_classifier.tolist() if cm_classifier is not None else None,
    "confusion_matrix_exported": cm_exported.tolist() if cm_exported is not None else None,
    "confusion_matrix_raw_candidates": cm_raw.tolist() if cm_raw is not None else None,
    "classification_metrics_classifier": clf_report_classifier,
    "classification_metrics_exported": clf_report_exported,
    "classification_metrics_raw_candidates": clf_report_raw,
    "self_consistency_mask_vs_ellipse": fit_metrics,
}

with open(OUT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2, ensure_ascii=False)

# =========================================================
# HUMAN-READABLE REPORT
# =========================================================
lines = []
lines.append("M9 FULL REPORT V2")
lines.append("=" * 80)
lines.append("")
lines.append(f"Seg checkpoint:   {SEG_CHECKPOINT_PATH}")
lines.append(f"Count checkpoint: {COUNT_CHECKPOINT_PATH}")
lines.append(f"Count arch: {count_arch}")
lines.append(f"Count classes: {count_classes}")
lines.append(f"Count image_size: {count_image_size}")
lines.append(f"Input dir: {INPUT_DIR}")
lines.append(f"Output dir: {OUT_DIR}")
lines.append(f"Total input images: {len(input_images)}")
lines.append(f"Images with expected K parseable from filename: {len(y_expected)}")
lines.append("")
lines.append("FILES WRITTEN")
lines.append("-" * 80)
lines.append("Per-image:")
lines.append("  <stem>__orig.png")
lines.append("  <stem>__fitted_ellipses_overlay.png")
lines.append("  <stem>__instance_labels_color.png")
lines.append("  <stem>__instance_labels_overlay.png")
lines.append("  <stem>__fitted_ellipses.json")
lines.append("")
lines.append("Global:")
lines.append("  summary.json")
lines.append("  summary.csv")
lines.append("  evaluation.json")
lines.append("  evaluation_report.txt")
lines.append("  confusion_matrix_classifier.png")
lines.append("  confusion_matrix_exported.png")
lines.append("  confusion_matrix_raw_candidates.png")
lines.append("  count_mismatch_classifier.txt")
lines.append("  count_mismatch_exported.txt")
lines.append("  count_mismatch_raw_candidates.txt")
lines.append("")
lines.append("IMPORTANT NOTE")
lines.append("-" * 80)
lines.append("For unknown images without annotations, no true segmentation accuracy can be computed.")
lines.append("Dice/IoU reported below are SELF-CONSISTENCY metrics:")
lines.append("  predicted mask vs rasterized fitted ellipse")
lines.append("These measure how well the fitted ellipse approximates the predicted mask,")
lines.append("NOT how well the prediction matches biological ground truth.")
lines.append("")

if clf_report_classifier is not None:
    lines.append("COUNT CONFUSION MATRIX — CLASSIFIER vs FILENAME K")
    lines.append("-" * 80)
    lines.append(str(cm_classifier))
    lines.append(f"Accuracy:          {clf_report_classifier['accuracy']:.6f}")
    lines.append(f"Balanced accuracy: {clf_report_classifier['balanced_accuracy']:.6f}")
    for k in COUNT_CLASSES:
        m = clf_report_classifier["per_class"][k]
        lines.append(
            f"  K={k}: precision={m['precision']:.6f} recall={m['recall']:.6f} f1={m['f1']:.6f} support={m['support']}"
        )
    lines.append("")

    lines.append("COUNT CONFUSION MATRIX — EXPORTED COUNT vs FILENAME K")
    lines.append("-" * 80)
    lines.append(str(cm_exported))
    lines.append(f"Accuracy:          {clf_report_exported['accuracy']:.6f}")
    lines.append(f"Balanced accuracy: {clf_report_exported['balanced_accuracy']:.6f}")
    for k in COUNT_CLASSES:
        m = clf_report_exported["per_class"][k]
        lines.append(
            f"  K={k}: precision={m['precision']:.6f} recall={m['recall']:.6f} f1={m['f1']:.6f} support={m['support']}"
        )
    lines.append("")

    lines.append("COUNT CONFUSION MATRIX — RAW CANDIDATES vs FILENAME K")
    lines.append("-" * 80)
    lines.append(str(cm_raw))
    lines.append(f"Accuracy:          {clf_report_raw['accuracy']:.6f}")
    lines.append(f"Balanced accuracy: {clf_report_raw['balanced_accuracy']:.6f}")
    for k in COUNT_CLASSES:
        m = clf_report_raw["per_class"][k]
        lines.append(
            f"  K={k}: precision={m['precision']:.6f} recall={m['recall']:.6f} f1={m['f1']:.6f} support={m['support']}"
        )
    lines.append("")
else:
    lines.append("No confusion matrix was computed because expected K could not be parsed from filenames.")
    lines.append("")

lines.append("SELF-CONSISTENCY OF SEGMENTATION vs FITTED ELLIPSES")
lines.append("-" * 80)
lines.append(
    f"Mean Dice all (mask vs ellipse): {fit_metrics['mean_dice_all_mask_vs_ellipse']}"
)
lines.append(
    f"Std  Dice all (mask vs ellipse): {fit_metrics['std_dice_all_mask_vs_ellipse']}"
)
lines.append(
    f"Mean IoU  all (mask vs ellipse): {fit_metrics['mean_iou_all_mask_vs_ellipse']}"
)
lines.append(
    f"Std  IoU  all (mask vs ellipse): {fit_metrics['std_iou_all_mask_vs_ellipse']}"
)
lines.append("")
lines.append("Per-class by expected K from filename:")
for k in sorted(fit_metrics["per_k"].keys()):
    m = fit_metrics["per_k"][k]
    lines.append(
        f"  K={k}: meanDice={m['mean_dice']}, stdDice={m['std_dice']}, "
        f"meanIoU={m['mean_iou']}, stdIoU={m['std_iou']}, n={m['n']}"
    )
lines.append("")

lines.append("FILES WITH CLASSIFIER COUNT MISMATCH")
lines.append("-" * 80)
if len(count_mismatch_classifier) > 3:
    lines.extend(count_mismatch_classifier[3:])
else:
    lines.append("No mismatches found.")
lines.append("")

lines.append("FILES WITH EXPORTED COUNT MISMATCH")
lines.append("-" * 80)
if len(count_mismatch_exported) > 3:
    lines.extend(count_mismatch_exported[3:])
else:
    lines.append("No mismatches found.")
lines.append("")

lines.append("FILES WITH RAW CANDIDATE COUNT MISMATCH")
lines.append("-" * 80)
if len(count_mismatch_raw) > 3:
    lines.extend(count_mismatch_raw[3:])
else:
    lines.append("No mismatches found.")

with open(OUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nDONE")
print(f"Outputs: {OUT_DIR.resolve()}")
print(f"Summary JSON: {OUT_DIR / 'summary.json'}")
print(f"Summary CSV: {OUT_DIR / 'summary.csv'}")
print(f"Evaluation JSON: {OUT_DIR / 'evaluation.json'}")
print(f"Evaluation report: {OUT_DIR / 'evaluation_report.txt'}")
print(f"Mismatch classifier: {OUT_DIR / 'count_mismatch_classifier.txt'}")
print(f"Mismatch exported: {OUT_DIR / 'count_mismatch_exported.txt'}")
print(f"Mismatch raw: {OUT_DIR / 'count_mismatch_raw_candidates.txt'}")
#%%