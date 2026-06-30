from pathlib import Path
import json, math, random, itertools, copy
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

script_dir = Path(__file__).resolve().parent
ANNOTATIONS_JSONL = script_dir / "jsonl" / "annotations.jsonl"
IMAGES_DIR = script_dir / "central_only"
OUT_DIR = script_dir / "m6_train_val_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SLOTS = 4
N_ELLIPSE_POINTS = 360
VAL_FRACTION = 0.2
BATCH_SIZE = 2
NUM_EPOCHS = 120
LR = 3e-4
WEIGHT_DECAY = 1e-4
USE_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 8
SCHEDULER_MIN_LR = 1e-6
LAMBDA_BCE = 1.0
LAMBDA_DICE = 1.0
PREVIEW_THRESH = 0.5
SAVE_VAL_PREVIEW_EVERY = 10
MAX_VAL_PREVIEWS = 12
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 1e-4
USE_AUG = True
ROT_DEG = 25.0
SCALE_MIN = 0.85
SCALE_MAX = 1.15
SHIFT_PX = 24.0
DO_HFLIP = True
MIN_FRACTION_INSIDE = 0.5
AUG_BORDER_MODE = cv2.BORDER_REFLECT_101

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def load_records(jsonl_path: Path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSONL error line {line_i}: {e}") from e
            records.append(rec)
    return records


def find_image_path(rec: dict, images_dir: Path) -> Path:
    for k in ["image", "image_path", "img", "img_path", "filename", "file_name", "path"]:
        if k in rec:
            p = Path(rec[k])
            return p if p.is_absolute() else images_dir / p
    raise KeyError(f"Missing image path key: {list(rec.keys())}")


def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def get_count(rec: dict) -> int:
    for k in ["count", "K", "cells", "cell_count"]:
        if k in rec:
            return int(rec[k])
    raise KeyError(f"Missing count key: {list(rec.keys())}")


def get_ellipses(rec: dict):
    ells = rec.get("ellipses")
    if not isinstance(ells, list):
        raise KeyError(f"Missing/invalid 'ellipses': {list(rec.keys())}")
    return ells


def ellipse_params(e: dict):
    def pick(d, names):
        for n in names:
            if n in d:
                return d[n]
        raise KeyError(f"Missing any of {names} in ellipse {d}")
    cx = float(pick(e, ["cx", "x", "center_x"]))
    cy = float(pick(e, ["cy", "y", "center_y"]))
    a = float(pick(e, ["a", "rx", "semi_major", "axis_a"]))
    b = float(pick(e, ["b", "ry", "semi_minor", "axis_b"]))
    theta = float(pick(e, ["theta", "angle", "phi"]))
    return cx, cy, a, b, theta


def ellipse_area(e: dict) -> float:
    _, _, a, b, _ = ellipse_params(e)
    return float(math.pi * a * b)


def sort_ellipses_by_size_desc(ellipses: list) -> list:
    return sorted(ellipses, key=ellipse_area, reverse=True)


def ellipse_boundary_points(e: dict, n_points: int = 360) -> np.ndarray:
    cx, cy, a, b, theta_deg = ellipse_params(e)
    if abs(theta_deg) <= 2 * math.pi + 1e-6:
        theta_deg = math.degrees(theta_deg)
    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False, dtype=np.float32)
    th = np.deg2rad(theta_deg).astype(np.float32)
    ct, st = np.cos(t), np.sin(t)
    c, s = np.cos(th), np.sin(th)
    x = cx + a * ct * c - b * st * s
    y = cy + a * ct * s + b * st * c
    return np.stack([x, y], axis=1).astype(np.float32)


def contour_to_mask(shape_hw, pts_xy: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_i = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], color=255, lineType=cv2.LINE_8)
    return (mask > 0).astype(np.uint8)


def make_instance_slots_from_contours(shape_hw, ellipses: list, max_slots: int = 4, n_points: int = 360):
    h, w = shape_hw
    slots = np.zeros((max_slots, h, w), dtype=np.uint8)
    used = sort_ellipses_by_size_desc(ellipses)[:max_slots]
    for i, e in enumerate(used):
        slots[i] = contour_to_mask((h, w), ellipse_boundary_points(e, n_points=n_points))
    valid_slots = np.zeros((max_slots,), dtype=np.uint8)
    valid_slots[:len(used)] = 1
    return slots, used, valid_slots


def random_affine_params(rng: random.Random):
    angle = rng.uniform(-ROT_DEG, ROT_DEG)
    scale = rng.uniform(SCALE_MIN, SCALE_MAX)
    tx = rng.uniform(-SHIFT_PX, SHIFT_PX)
    ty = rng.uniform(-SHIFT_PX, SHIFT_PX)
    do_hflip = DO_HFLIP and (rng.random() < 0.5)
    return angle, scale, tx, ty, do_hflip


def build_affine_matrix(w, h, angle_deg, scale, tx, ty, do_hflip):
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    if do_hflip:
        Fm = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        A = np.eye(3, dtype=np.float32)
        A[:2, :] = M.astype(np.float32)
        M = (Fm @ A)[:2, :]
    return M.astype(np.float32)


def apply_affine_to_image(img: np.ndarray, M_2x3: np.ndarray):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M_2x3, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=AUG_BORDER_MODE)


def transform_point_affine(x, y, M_2x3):
    q = M_2x3 @ np.array([x, y, 1.0], dtype=np.float32)
    return float(q[0]), float(q[1])


def transform_points_affine(pts_xy: np.ndarray, M_2x3: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1), dtype=np.float32)], axis=1)
    return (pts_h @ M_2x3.T).astype(np.float32)


def transformed_center_and_contour(e: dict, M_2x3: np.ndarray, n_points: int = 360):
    cx, cy, _, _, _ = ellipse_params(e)
    return transform_point_affine(cx, cy, M_2x3), transform_points_affine(ellipse_boundary_points(e, n_points=n_points), M_2x3)


def contour_inside_image(pts_xy: np.ndarray, w: int, h: int, min_fraction_inside: float = 0.5) -> bool:
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return float(np.mean(inside)) >= min_fraction_inside


def build_augmented_slots(shape_hw, ellipses: list, M_2x3: np.ndarray, max_slots: int = 4, n_points: int = 360, min_fraction_inside: float = 0.5):
    h, w = shape_hw
    kept = []
    for e in sort_ellipses_by_size_desc(ellipses):
        center2, pts2 = transformed_center_and_contour(e, M_2x3, n_points=n_points)
        if contour_inside_image(pts2, w, h, min_fraction_inside=min_fraction_inside):
            kept.append({"center": center2, "pts": pts2, "orig_area": ellipse_area(e)})
    used = kept[:max_slots]
    slots = np.zeros((max_slots, h, w), dtype=np.uint8)
    for i, item in enumerate(used):
        slots[i] = contour_to_mask((h, w), item["pts"])
    valid_slots = np.zeros((max_slots,), dtype=np.uint8)
    valid_slots[:len(used)] = 1
    return slots, valid_slots, len(used)


def stratified_split_indices(records, val_fraction=0.2, rng=None):
    rng = random.Random(RANDOM_SEED) if rng is None else rng
    by_k = defaultdict(list)
    for i, rec in enumerate(records):
        by_k[get_count(rec)].append(i)
    train_indices, val_indices = [], []
    for k in sorted(by_k.keys()):
        idxs = by_k[k][:]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices, by_k


class EllipseSlotDataset(Dataset):
    def __init__(self, records, images_dir: Path, indices: list[int], training: bool, use_aug: bool, base_seed: int):
        self.records = records
        self.images_dir = images_dir
        self.indices = indices
        self.training = training
        self.use_aug = use_aug
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec_i = self.indices[i]
        rec = self.records[rec_i]
        img_path = find_image_path(rec, self.images_dir)
        img = load_gray_image(img_path)
        h, w = img.shape[:2]
        ellipses = get_ellipses(rec)
        K = get_count(rec)
        if self.training and self.use_aug:
            rng = random.Random(self.base_seed + rec_i + random.randint(0, 10_000_000))
            M = build_affine_matrix(w, h, *random_affine_params(rng))
            img_aug = apply_affine_to_image(img, M)
            slots_u8, valid_slots, used_count = build_augmented_slots((h, w), ellipses, M_2x3=M, max_slots=MAX_SLOTS, n_points=N_ELLIPSE_POINTS, min_fraction_inside=MIN_FRACTION_INSIDE)
            if used_count == 0:
                img_final = img
                slots_u8, _, valid_slots = make_instance_slots_from_contours((h, w), ellipses, max_slots=MAX_SLOTS, n_points=N_ELLIPSE_POINTS)
            else:
                img_final = img_aug
        else:
            img_final = img
            slots_u8, _, valid_slots = make_instance_slots_from_contours((h, w), ellipses, max_slots=MAX_SLOTS, n_points=N_ELLIPSE_POINTS)
        x = img_final.astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(x[None]),
            "slots": torch.from_numpy(slots_u8.astype(np.float32)),
            "valid_slots": torch.from_numpy(valid_slots.astype(np.float32)),
            "count": torch.tensor(K, dtype=torch.long),
            "record_idx": torch.tensor(rec_i, dtype=torch.long),
            "image_name": img_path.name,
        }


def convblock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Decoder(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = convblock(in_ch + skip_ch, out_ch)
        self.conv2 = convblock(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class ResNetUNet18(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.input_conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            self.input_conv.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.center = nn.Sequential(convblock(512, 512), convblock(512, 512))
        self.dec4 = Decoder(512, 256, 256)
        self.dec3 = Decoder(256, 128, 128)
        self.dec2 = Decoder(128, 64, 64)
        self.dec1 = Decoder(64, 64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.input_conv(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        z = self.center(x4)
        z = self.dec4(z, x3)
        z = self.dec3(z, x2)
        z = self.dec2(z, x1)
        z = self.dec1(z, x0)
        logits = self.final_conv(z)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_loss_from_probs_torch(pred_probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    inter = (pred_probs * target).sum()
    denom = pred_probs.sum() + target.sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


def dice_score_binary_torch(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    inter = (pred_bin * target).sum()
    denom = pred_bin.sum() + target.sum()
    return (2.0 * inter + eps) / (denom + eps)


def slot_pair_loss_torch(pred_logits_hw: torch.Tensor, target_hw: torch.Tensor, lambda_bce=1.0, lambda_dice=1.0):
    bce = F.binary_cross_entropy_with_logits(pred_logits_hw, target_hw)
    if float(target_hw.sum().item()) <= 0.0:
        dice = torch.zeros((), dtype=pred_logits_hw.dtype, device=pred_logits_hw.device)
        total = lambda_bce * bce
    else:
        dice = dice_loss_from_probs_torch(torch.sigmoid(pred_logits_hw), target_hw)
        total = lambda_bce * bce + lambda_dice * dice
    return total, bce, dice


def permutation_invariant_loss_and_dice_torch(pred_logits_b4hw: torch.Tensor, target_b4hw: torch.Tensor, lambda_bce=1.0, lambda_dice=1.0, thr=0.5):
    batch_losses, batch_dices, debug_rows = [], [], []
    perms = list(itertools.permutations(range(MAX_SLOTS)))
    for b in range(pred_logits_b4hw.shape[0]):
        pred_logits, target = pred_logits_b4hw[b], target_b4hw[b]
        best_loss, best_dice_mean, best_debug = None, None, None
        pred_probs = torch.sigmoid(pred_logits)
        pred_bin = (pred_probs >= thr).float()
        for perm in perms:
            slot_losses, slot_bces, slot_dices_loss, slot_dices_bin = [], [], [], []
            for pred_i in range(MAX_SLOTS):
                tgt_i = perm[pred_i]
                total, bce, dice_loss_val = slot_pair_loss_torch(pred_logits[pred_i], target[tgt_i], lambda_bce=lambda_bce, lambda_dice=lambda_dice)
                dice_bin = dice_score_binary_torch(pred_bin[pred_i], target[tgt_i])
                slot_losses.append(total); slot_bces.append(bce); slot_dices_loss.append(dice_loss_val); slot_dices_bin.append(dice_bin)
            loss_mean = torch.stack(slot_losses).mean()
            bce_mean = torch.stack(slot_bces).mean()
            dice_loss_mean = torch.stack(slot_dices_loss).mean()
            dice_bin_mean = torch.stack(slot_dices_bin).mean()
            if best_loss is None or loss_mean.item() < best_loss.item():
                best_loss, best_dice_mean = loss_mean, dice_bin_mean
                best_debug = {"perm": tuple(int(x) for x in perm), "loss_total": float(loss_mean.item()), "loss_bce": float(bce_mean.item()), "loss_dice": float(dice_loss_mean.item()), "dice_bin": float(dice_bin_mean.item())}
        batch_losses.append(best_loss); batch_dices.append(best_dice_mean); debug_rows.append(best_debug)
    return torch.stack(batch_losses).mean(), torch.stack(batch_dices).mean(), debug_rows


def slots_to_grid_rgb(slots_4hw: np.ndarray, is_prob: bool) -> np.ndarray:
    tiles = []
    for i in range(4):
        x = slots_4hw[i]
        tile = np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        tiles.append(tile)
    return cv2.cvtColor(np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])]), cv2.COLOR_GRAY2RGB)


def put_text_multiline(img_rgb: np.ndarray, lines: list[str], x=8, y=20, dy=18):
    out = img_rgb.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def make_preview(image_1hw: np.ndarray, gt_4hw: np.ndarray, pred_prob_4hw: np.ndarray, info_lines: list[str], thr=0.5):
    img_rgb = cv2.cvtColor(np.round(np.clip(image_1hw[0], 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    gt_rgb = cv2.resize(slots_to_grid_rgb(gt_4hw, False), (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_rgb = cv2.resize(slots_to_grid_rgb(pred_prob_4hw, True), (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_bin_rgb = cv2.resize(slots_to_grid_rgb((pred_prob_4hw >= thr).astype(np.float32), False), (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.vstack([put_text_multiline(img_rgb, info_lines), put_text_multiline(gt_rgb, ["GT slots"]), put_text_multiline(pred_rgb, ["Pred probs"]), put_text_multiline(pred_bin_rgb, [f"Pred bin thr={thr:.2f}"])])


def aggregate_k_metrics(batch_counts, batch_debug_rows, out_dict):
    for cnt, dbg in zip(batch_counts, batch_debug_rows):
        k = int(cnt)
        out_dict[k]["loss_sum"] += float(dbg["loss_total"])
        out_dict[k]["dice_sum"] += float(dbg["dice_bin"])
        out_dict[k]["n"] += 1


def finalize_k_metrics(k_dict):
    result = {}
    for k in sorted(k_dict.keys()):
        n = max(k_dict[k]["n"], 1)
        result[k] = {"loss": k_dict[k]["loss_sum"] / n, "dice": k_dict[k]["dice_sum"] / n, "n": int(k_dict[k]["n"])}
    return result


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = dice_sum = 0.0
    n_batches = 0
    k_metrics = defaultdict(lambda: {"loss_sum": 0.0, "dice_sum": 0.0, "n": 0})
    for batch in loader:
        x, y = batch["image"].to(device), batch["slots"].to(device)
        logits = model(x)
        loss, dice, debug_rows = permutation_invariant_loss_and_dice_torch(logits, y, lambda_bce=LAMBDA_BCE, lambda_dice=LAMBDA_DICE, thr=PREVIEW_THRESH)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        loss_sum += float(loss.item()); dice_sum += float(dice.item()); n_batches += 1
        aggregate_k_metrics([int(v.item()) for v in batch["count"]], debug_rows, k_metrics)
    return loss_sum / max(n_batches, 1), dice_sum / max(n_batches, 1), finalize_k_metrics(k_metrics)


@torch.no_grad()
def evaluate_model(model, loader, device, save_previews=False, preview_dir=None, epoch=None):
    model.eval()
    loss_sum = dice_sum = 0.0
    n_batches = 0
    k_metrics = defaultdict(lambda: {"loss_sum": 0.0, "dice_sum": 0.0, "n": 0})
    previews_saved = 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["slots"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        loss, dice, debug_rows = permutation_invariant_loss_and_dice_torch(logits, y, lambda_bce=LAMBDA_BCE, lambda_dice=LAMBDA_DICE, thr=PREVIEW_THRESH)
        loss_sum += float(loss.item()); dice_sum += float(dice.item()); n_batches += 1
        aggregate_k_metrics([int(v.item()) for v in batch["count"]], debug_rows, k_metrics)
        if save_previews and preview_dir is not None and previews_saved < MAX_VAL_PREVIEWS:
            x_np, y_np, probs_np = x.cpu().numpy(), y.cpu().numpy(), probs.cpu().numpy()
            for bi in range(x_np.shape[0]):
                if previews_saved >= MAX_VAL_PREVIEWS:
                    break
                rec_idx = int(batch["record_idx"][bi].item())
                K = int(batch["count"][bi].item())
                dbg = debug_rows[bi]
                info_lines = [f"epoch={epoch} rec_idx={rec_idx} K={K}", batch["image_name"][bi], f"perm={dbg['perm']}", f"loss={dbg['loss_total']:.5f} bce={dbg['loss_bce']:.5f} diceLoss={dbg['loss_dice']:.5f}", f"diceBin={dbg['dice_bin']:.5f}"]
                preview = make_preview(x_np[bi], y_np[bi], probs_np[bi], info_lines, thr=PREVIEW_THRESH)
                cv2.imwrite(str(preview_dir / f"rec{rec_idx:03d}_K{K}.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                previews_saved += 1
    return loss_sum / max(n_batches, 1), dice_sum / max(n_batches, 1), finalize_k_metrics(k_metrics)


def main():
    records = load_records(ANNOTATIONS_JSONL)
    rng = random.Random(RANDOM_SEED)
    train_indices, val_indices, by_k = stratified_split_indices(records, val_fraction=VAL_FRACTION, rng=rng)
    print(f"Počet záznamů: {len(records)}")
    print("Rozdělení count:")
    for k in sorted(by_k.keys()):
        print(f"  K={k}: {len(by_k[k])}")
    train_ds = EllipseSlotDataset(records, IMAGES_DIR, train_indices, training=True, use_aug=USE_AUG, base_seed=RANDOM_SEED)
    val_ds = EllipseSlotDataset(records, IMAGES_DIR, val_indices, training=False, use_aug=False, base_seed=RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    model = ResNetUNet18(in_channels=1, out_channels=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR) if USE_SCHEDULER else None
    history = []
    best_val_loss, best_val_loss_epoch = float("inf"), -1
    best_val_dice, best_val_dice_epoch = -1.0, -1
    best_per_k = {1: {"best_dice": -1.0, "epoch": -1}, 2: {"best_dice": -1.0, "epoch": -1}, 4: {"best_dice": -1.0, "epoch": -1}}
    epochs_without_improvement = 0
    preview_root = OUT_DIR / "val_previews"; preview_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice, train_k = train_one_epoch(model, train_loader, optimizer, DEVICE)
        save_previews = (epoch == 1) or (epoch % SAVE_VAL_PREVIEW_EVERY == 0) or (epoch == NUM_EPOCHS)
        preview_dir = preview_root / f"epoch_{epoch:04d}" if save_previews else None
        if save_previews:
            preview_dir.mkdir(parents=True, exist_ok=True)
        val_loss, val_dice, val_k = evaluate_model(model, val_loader, DEVICE, save_previews=save_previews, preview_dir=preview_dir, epoch=epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": int(epoch), "lr": float(current_lr), "train_loss": float(train_loss), "train_dice": float(train_dice), "val_loss": float(val_loss), "val_dice": float(val_dice), "train_by_k": train_k, "val_by_k": val_k})
        improved_loss = val_loss < (best_val_loss - EARLY_STOP_MIN_DELTA)
        improved_dice = val_dice > best_val_dice
        if improved_loss:
            best_val_loss, best_val_loss_epoch = val_loss, epoch
            torch.save(copy.deepcopy(model.state_dict()), OUT_DIR / "best_model_by_val_loss.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if improved_dice:
            best_val_dice, best_val_dice_epoch = val_dice, epoch
            torch.save(copy.deepcopy(model.state_dict()), OUT_DIR / "best_model_by_val_dice.pt")
        for k in [1, 2, 4]:
            if k in val_k and val_k[k]["dice"] > best_per_k[k]["best_dice"]:
                best_per_k[k]["best_dice"] = float(val_k[k]["dice"])
                best_per_k[k]["epoch"] = int(epoch)
        torch.save(model.state_dict(), OUT_DIR / "last_model.pt")
        if scheduler is not None:
            scheduler.step(val_loss)
        if epoch <= 10 or epoch % 5 == 0:
            print(f"epoch {epoch:04d} | lr={current_lr:.2e} | train_loss={train_loss:.5f} train_dice={train_dice:.5f} | val_loss={val_loss:.5f} val_dice={val_dice:.5f} | best_val_loss={best_val_loss:.5f} @ {best_val_loss_epoch} | best_val_dice={best_val_dice:.5f} @ {best_val_dice_epoch}")
            for k in sorted(val_k.keys()):
                print(f"    val K={k}: loss={val_k[k]['loss']:.5f} dice={val_k[k]['dice']:.5f} n={val_k[k]['n']} | best_dice={best_per_k[k]['best_dice']:.5f} @ {best_per_k[k]['epoch']}")
        if USE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: no val_loss improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"train_indices": train_indices, "val_indices": val_indices, "best_val_loss": float(best_val_loss), "best_val_loss_epoch": int(best_val_loss_epoch), "best_val_dice": float(best_val_dice), "best_val_dice_epoch": int(best_val_dice_epoch), "best_per_k": best_per_k, "history": history}, f, indent=2, ensure_ascii=False)
    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    train_dices = [r["train_dice"] for r in history]
    val_dices = [r["val_dice"] for r in history]
    plt.figure(figsize=(8, 5)); plt.plot(epochs, train_losses, label="train_loss"); plt.plot(epochs, val_losses, label="val_loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("M6 train/val loss with aug"); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR / "loss_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5)); plt.plot(epochs, train_dices, label="train_dice"); plt.plot(epochs, val_dices, label="val_dice"); plt.xlabel("epoch"); plt.ylabel("dice"); plt.title("M6 train/val dice with aug"); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR / "dice_curve.png", dpi=150); plt.close()
    print("\nDONE")
    print(f"Best val loss: {best_val_loss:.6f} @ epoch {best_val_loss_epoch}")
    print(f"Best val dice: {best_val_dice:.6f} @ epoch {best_val_dice_epoch}")
    for k in [1, 2, 4]:
        print(f"Best K={k} val dice: {best_per_k[k]['best_dice']:.6f} @ epoch {best_per_k[k]['epoch']}")
    print(f"Výstupy: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
