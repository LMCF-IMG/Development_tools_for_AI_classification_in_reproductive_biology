#%% 
# ============================================================
# cellcount_cnn_vscode_verbose.py
# - verbose terminal progress for scanning, loading, augmentation, training, eval and inference
# - robust error reporting (file paths + tracebacks)
# ============================================================

import sys
import time
import random
import traceback
import faulthandler
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import inspect
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# tqdm is optional; code falls back to plain prints if not installed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

#%% 
# ============================================================
# CONFIG (upravujte zde ve VS Code)
# ============================================================

MODE = "predict"  # "train" nebo "predict"

# Split po embryích (doporučeno fixně). Embryo ID je z prvních 3 číslic: 001->1.
VAL_EMBRYO_IDS = [7]       # např. [7]
TEST_EMBRYO_IDS = [8]      # např. [8]
# Pokud chcete náhodný split: nastavte VAL_EMBRYO_IDS=None a TEST_EMBRYO_IDS=None.

IMG_ROOT = r"d:/Programovani/Human_Embryo_Classification_TV/crops"  # složka s cropy (400x400, 8-bit)
OUT_DIR = r"d:/Programovani/Human_Embryo_Classification_TV/runs/t_1-6_v-7_t-8"

# Přípony souborů, které bereme
EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Uvažované třídy (počty buněk)
ALLOWED_COUNTS = {1, 2, 4}

# Model / trénink
ARCH = "resnet18"          # "resnet18" nebo "efficientnet_b0"
PRETRAINED = True
IMAGE_SIZE = 400
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4            # při debugování doporučuji 0 (lepší traceback)
FREEZE_BACKBONE_EPOCHS = 0

# Vyvážení tříd
USE_CLASS_WEIGHTS = True
USE_OVERSAMPLING = False
OVERSAMPLE_POWER = 1.0

# Early-stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.001

# Predikce
CKPT_PATH = f"{OUT_DIR}/best.pt"
PREDICT_OUTPUT_CSV = f"{OUT_DIR}/predictions.csv" 
PREDICT_GLOB = None  # např. "**/*.tif" nebo None => vše v IMG_ROOT

# ===== VERBOSE / DEBUG =====
LOG_TO_FILE = True                 # log i do OUT_DIR/run.log
LOG_LEVEL = "INFO"                 # "DEBUG" | "INFO" | "WARNING" | "ERROR"
SHOW_TQDM = True                   # progress bary (vyžaduje tqdm)
PRINT_EVERY_N_BATCHES = 20         # když tqdm není, tisk po N batchech
SHOW_GPU_MEMORY = True             # tisk VRAM (CUDA)
FAIL_FAST_ON_BAD_SAMPLE = True     # True: chyba v __getitem__ ukončí běh; False: sample se přeskočí (collate_fn)
MAX_BAD_SAMPLES = 50               # limit pro přeskočené vadné vzorky (jen když FAIL_FAST...=False)

# rychlá diagnostika datasetu (načtení pár sample + uložení augmentovaných ukázek)
RUN_DATASET_SANITY_CHECK = True
SANITY_N_SAMPLES = 16
SAVE_AUGMENTED_DEBUG = True
AUG_DEBUG_N_SAVE = 12              # kolik augmentovaných vstupů uložit do OUT_DIR/debug_aug/

#%% 
# ============================================================
# Logging / Utils
# ============================================================

def setup_logging(out_dir: Path) -> logging.Logger:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger = logging.getLogger("cellcount")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if LOG_TO_FILE:
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def gpu_mem_string() -> str:
    if not torch.cuda.is_available():
        return "GPU: n/a"
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    return f"GPU MB alloc={alloc:.0f} reserved={reserved:.0f} max_alloc={max_alloc:.0f}"


def parse_embryo_id_and_count(filename: str) -> Tuple[int, int]:
    """
    Formát: 001_039h20m_FOC_1.tif
      embryo_id = 1  (první token před prvním "_", přesně 3 číslice)
      count = 1      (poslední token po posledním "_" ve stému)
    """
    p = Path(filename)
    stem = p.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad filename format (too few '_'): {filename}")

    embryo_str = parts[0]
    count_str = parts[-1]

    if len(embryo_str) != 3 or not embryo_str.isdigit():
        raise ValueError(f"Bad embryo id token (expected 3 digits): {filename}")
    if not count_str.isdigit():
        raise ValueError(f"Bad count token (expected digits at end): {filename}")

    embryo_id = int(embryo_str)
    count = int(count_str)
    return embryo_id, count


def collect_labeled_records(img_root: str, logger: logging.Logger) -> List[dict]:
    root = Path(img_root)
    if not root.exists():
        raise FileNotFoundError(f"IMG_ROOT does not exist: {img_root}")

    files = [fp for fp in root.rglob("*") if fp.is_file()]
    logger.info(f"Scanning {len(files)} files under: {root.resolve()}")

    records: List[dict] = []
    skipped_ext = 0
    skipped_parse = 0
    skipped_class = 0

    iterator = files
    if SHOW_TQDM and tqdm is not None:
        iterator = tqdm(files, desc="Scanning files", unit="file")

    for fp in iterator:
        if fp.suffix.lower() not in EXTENSIONS:
            skipped_ext += 1
            continue

        try:
            embryo_id, count = parse_embryo_id_and_count(fp.name)
        except Exception:
            skipped_parse += 1
            continue

        if count not in ALLOWED_COUNTS:
            skipped_class += 1
            continue

        records.append({
            "filepath": str(fp.relative_to(root)),
            "embryo_id": embryo_id,
            "cell_count": count
        })

    if len(records) == 0:
        raise RuntimeError(
            f"No usable images found in {img_root}. "
            "Check EXTENSIONS, ALLOWED_COUNTS, and filename format like 001_..._1.tif."
        )

    logger.info(f"Collected {len(records)} labeled images.")
    logger.info(f"Skipped: ext={skipped_ext}, parse={skipped_parse}, class_not_used={skipped_class}")
    return records


def counts_by_class(records: List[dict], classes: List[int]) -> Dict[int, int]:
    d = {c: 0 for c in classes}
    for r in records:
        d[int(r["cell_count"])] += 1
    return d


def split_by_embryos(
    records: List[dict],
    val_embryos: Optional[List[int]],
    test_embryos: Optional[List[int]],
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[dict], List[dict], List[dict], List[int], List[int]]:
    embryo_ids = sorted({r["embryo_id"] for r in records})
    logger.info(f"Found embryos: {embryo_ids} (n={len(embryo_ids)})")

    if val_embryos is not None and test_embryos is not None:
        val_set = set(val_embryos)
        test_set = set(test_embryos)
        if val_set & test_set:
            raise ValueError(f"VAL_EMBRYO_IDS and TEST_EMBRYO_IDS overlap: {sorted(val_set & test_set)}")
        unknown = (val_set | test_set) - set(embryo_ids)
        if unknown:
            raise ValueError(f"VAL/TEST embryos not found in data: {sorted(unknown)}")
    else:
        rng = np.random.default_rng(seed)
        ids = embryo_ids.copy()
        rng.shuffle(ids)
        test_set = {ids[0]}
        val_set = {ids[1]}
        logger.info(f"Random split selected: VAL={sorted(val_set)}, TEST={sorted(test_set)}")

    train_set = set(embryo_ids) - val_set - test_set
    if len(train_set) == 0:
        raise ValueError("No embryos left for training after VAL/TEST selection.")

    train = [r for r in records if r["embryo_id"] in train_set]
    val = [r for r in records if r["embryo_id"] in val_set]
    test = [r for r in records if r["embryo_id"] in test_set]

    return train, val, test, sorted(list(val_set)), sorted(list(test_set))


def format_confusion_matrix(cm: np.ndarray, classes: List[int]) -> str:
    header = " " * 10 + " ".join([f"pred_{c:>4}" for c in classes])
    lines = [header]
    for i, c in enumerate(classes):
        row = " ".join([f"{cm[i, j]:>8d}" for j in range(len(classes))])
        lines.append(f"true_{c:<4} {row}")
    return "\n".join(lines)


def write_epoch_report(
    report_path: Path,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_acc: float,
    val_f1: float,
    classes: List[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lr: float,
    improved: bool,
    epochs_no_improve: int,
    best_f1: float,
) -> None:
    rep = classification_report(
        y_true,
        y_pred,
        target_names=[str(c) for c in classes],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Epoch {epoch:03d}/{epochs}\n")
        f.write(f"LR: {lr:.6g}\n")
        f.write(f"Train loss: {train_loss:.6f}\n")
        f.write(f"VAL acc: {val_acc:.4f}\n")
        f.write(f"VAL macroF1: {val_f1:.4f}\n")
        f.write(f"Improved: {improved} | epochs_no_improve: {epochs_no_improve} | best_f1: {best_f1:.4f}\n\n")
        f.write("classification_report (VAL):\n")
        f.write(rep + "\n")
        f.write("\nconfusion_matrix (VAL):\n")
        f.write(format_confusion_matrix(cm, classes) + "\n")


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) normalized tensor. Convert back to uint8 RGB for debug saving.
    """
    x = x.detach().cpu().float()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return x

#%% 
# ============================================================
# Dataset (+ optional skipping bad samples)
# ============================================================

class CellCountDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        img_root: str,
        class_to_index: Dict[int, int],
        transform=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.records = records
        self.root = Path(img_root)
        self.class_to_index = class_to_index
        self.transform = transform
        self.logger = logger

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img_path = self.root / r["filepath"]

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            img = Image.open(img_path).convert("L")  # 8-bit grayscale

            count = int(r["cell_count"])
            y = self.class_to_index[count]

            if self.transform is not None:
                img = self.transform(img)

            return img, y

        except Exception as e:
            msg = f"Dataset __getitem__ failed for idx={idx}, file='{img_path}': {repr(e)}"
            if self.logger is not None:
                self.logger.error(msg)
                self.logger.debug("Traceback:\n" + traceback.format_exc())

            if FAIL_FAST_ON_BAD_SAMPLE:
                raise RuntimeError(msg) from e
            else:
                # return None and let collate_fn filter it out
                return None


def filtering_collate_fn(batch, bad_counter: Dict[str, int], logger: logging.Logger):
    """
    Collate that filters out None samples (only if FAIL_FAST_ON_BAD_SAMPLE=False).
    """
    filtered = [b for b in batch if b is not None]
    bad = len(batch) - len(filtered)
    if bad > 0:
        bad_counter["count"] += bad
        logger.warning(f"Skipped {bad} bad sample(s) in a batch. Total skipped so far: {bad_counter['count']}")
        if bad_counter["count"] > MAX_BAD_SAMPLES:
            raise RuntimeError(f"Too many bad samples skipped ({bad_counter['count']} > {MAX_BAD_SAMPLES}).")

    if len(filtered) == 0:
        return None

    xs, ys = zip(*filtered)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

#%% 
# ============================================================
# Model
# ============================================================

def build_model(num_classes: int, arch: str, pretrained: bool) -> nn.Module:
    import torchvision.models as M

    arch = arch.lower().strip()
    if arch == "resnet18":
        try:
            weights = M.ResNet18_Weights.DEFAULT if pretrained else None
            model = M.resnet18(weights=weights)
        except Exception:
            model = M.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        try:
            weights = M.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = M.efficientnet_b0(weights=weights)
        except Exception:
            model = M.efficientnet_b0(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError("ARCH must be 'resnet18' or 'efficientnet_b0'")


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

#%% 
# ============================================================
# Transforms
# ============================================================

def build_transforms(image_size: int, train: bool) -> object:
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),

            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR),

            T.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),

            T.ColorJitter(brightness=0.12, contrast=0.18),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),

            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

#%% 
# ============================================================
# Training / Eval
# ============================================================

def compute_class_weights(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_sampler_for_oversampling(y_idx: np.ndarray, num_classes: int, power: float) -> WeightedRandomSampler:
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_w = (1.0 / counts) ** float(power)
    sample_w = class_w[y_idx]
    sample_w = torch.tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


#%%
# ============================================================
# PyTorch API compatibility helpers
# - AMP: silence torch.cuda.amp deprecation warnings by preferring torch.amp
# - Scheduler: avoid keyword-arg incompatibilities across torch versions
# ============================================================

def _amp_device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def make_grad_scaler(device: str):
    """Create an AMP GradScaler compatible with both old and new PyTorch."""
    use_cuda = str(device).startswith("cuda")

    # Newer PyTorch: torch.amp.GradScaler(...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if "device_type" in sig.parameters:
                return torch.amp.GradScaler(device_type="cuda" if use_cuda else "cpu")
        except Exception:
            pass

        # Some builds accept the device type as the first positional argument
        try:
            return torch.amp.GradScaler("cuda" if use_cuda else "cpu")
        except Exception:
            pass

    # Older PyTorch: torch.cuda.amp.GradScaler(enabled=...)
    return torch.cuda.amp.GradScaler(enabled=use_cuda)


def autocast_ctx(device: str, enabled: bool = True):
    """Return an autocast context manager compatible with both old and new PyTorch."""
    if not enabled:
        return nullcontext()

    dev = _amp_device_type(device)

    # Newer PyTorch: torch.amp.autocast(device_type=...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            sig = inspect.signature(torch.amp.autocast)
            if "device_type" in sig.parameters:
                return torch.amp.autocast(device_type=dev, enabled=True)
        except Exception:
            pass

        # Some builds accept the device type as the first positional argument
        try:
            return torch.amp.autocast(dev, enabled=True)
        except Exception:
            pass

    # Older PyTorch: CUDA-only autocast
    if dev == "cuda":
        return torch.cuda.amp.autocast(enabled=True)

    return nullcontext()


def make_plateau_scheduler(optimizer, **kwargs):
    """Create ReduceLROnPlateau, filtering unsupported kwargs (e.g., verbose)."""
    cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    sig = inspect.signature(cls.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(optimizer, **filtered)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str, logger: logging.Logger, tag: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []

    iterator = loader
    if SHOW_TQDM and tqdm is not None:
        iterator = tqdm(loader, desc=f"{tag} eval", unit="batch")

    for batch in iterator:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro"))
    return acc, macro_f1, y_true_np, y_pred_np


def train_one_epoch(model, loader, optimizer, criterion, device: str, logger: logging.Logger, epoch: int, epochs: int) -> float:
    model.train()
    scaler = make_grad_scaler(device)

    running_loss = 0.0
    n = 0

    iterator = loader
    if SHOW_TQDM and tqdm is not None:
        iterator = tqdm(loader, desc=f"Train {epoch}/{epochs}", unit="batch")

    for bi, batch in enumerate(iterator):
        if batch is None:
            continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device, enabled=str(device).startswith("cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        running_loss += float(loss.item()) * bs
        n += bs
        avg_loss = running_loss / max(1, n)

        if SHOW_TQDM and tqdm is not None:
            postfix = {"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"}
            if SHOW_GPU_MEMORY and torch.cuda.is_available():
                postfix["maxMB"] = f"{torch.cuda.max_memory_allocated()/(1024**2):.0f}"
            iterator.set_postfix(postfix)
        else:
            if (bi + 1) % PRINT_EVERY_N_BATCHES == 0:
                extra = f" | {gpu_mem_string()}" if (SHOW_GPU_MEMORY and torch.cuda.is_available()) else ""
                logger.info(f"[Train] epoch={epoch}/{epochs} batch={bi+1} loss={loss.item():.4f} avg_loss={avg_loss:.4f}{extra}")

    return running_loss / max(1, n)


def dataset_sanity_check(ds: Dataset, out_dir: Path, logger: logging.Logger, prefix: str) -> None:
    """
    Load first N samples sequentially to catch I/O / transform errors early.
    Optionally save a few augmented images.
    """
    n = min(SANITY_N_SAMPLES, len(ds))
    logger.info(f"Running dataset sanity check: {prefix}, n={n}")

    save_dir = out_dir / "debug_aug" / prefix
    if SAVE_AUGMENTED_DEBUG:
        save_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for i in range(n):
        sample = ds[i]  # will raise with explicit file path if broken
        if sample is None:
            continue
        x, y = sample
        if SAVE_AUGMENTED_DEBUG and n_saved < AUG_DEBUG_N_SAVE:
            try:
                arr = tensor_to_uint8_image(x)
                Image.fromarray(arr).save(save_dir / f"{prefix}_idx{i:04d}_y{int(y)}.png")
                n_saved += 1
            except Exception as e:
                logger.warning(f"Failed to save debug augmented image for idx={i}: {repr(e)}")

    logger.info(f"Sanity check done: {prefix}. Saved {n_saved} augmented samples to: {save_dir if SAVE_AUGMENTED_DEBUG else '(disabled)'}")


def run_training() -> None:
    faulthandler.enable()
    set_seed(SEED)

    out_dir = ensure_out_dir(OUT_DIR)
    logger = setup_logging(out_dir)
    logger.info("=== START TRAINING ===")
    logger.info(f"Config: IMG_ROOT={IMG_ROOT}, OUT_DIR={out_dir.resolve()}")
    logger.info(f"Classes used: {sorted(ALLOWED_COUNTS)} | arch={ARCH} | image_size={IMAGE_SIZE} | batch={BATCH_SIZE} | workers={NUM_WORKERS}")
    logger.info(f"Class weights: {USE_CLASS_WEIGHTS} | Oversampling: {USE_OVERSAMPLING} | EarlyStop: {USE_EARLY_STOPPING}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(gpu_mem_string())

    report_path = out_dir / "epoch_reports_val.txt"
    final_path = out_dir / "final_eval_best.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Reset epoch report file
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Epoch-wise validation reports\n")
        f.write(f"IMG_ROOT={IMG_ROOT}\n")
        f.write(f"ARCH={ARCH}, PRETRAINED={PRETRAINED}, IMAGE_SIZE={IMAGE_SIZE}\n")
        f.write(f"ALLOWED_COUNTS={sorted(ALLOWED_COUNTS)}\n")
        f.write(f"USE_CLASS_WEIGHTS={USE_CLASS_WEIGHTS}, USE_OVERSAMPLING={USE_OVERSAMPLING}\n")
        f.write(f"EARLY_STOPPING: USE={USE_EARLY_STOPPING}, PATIENCE={EARLY_STOPPING_PATIENCE}, MIN_DELTA={EARLY_STOPPING_MIN_DELTA}\n")
        f.write("=" * 80 + "\n")

    # Load & split
    t0 = time.time()
    records = collect_labeled_records(IMG_ROOT, logger)
    logger.info(f"Scan+index time: {time.time() - t0:.2f}s")

    classes = sorted(list(ALLOWED_COUNTS))  # [1,2,4]
    class_to_index = {c: i for i, c in enumerate(classes)}

    train_rec, val_rec, test_rec, used_val, used_test = split_by_embryos(
        records,
        val_embryos=VAL_EMBRYO_IDS if VAL_EMBRYO_IDS is not None else None,
        test_embryos=TEST_EMBRYO_IDS if TEST_EMBRYO_IDS is not None else None,
        seed=SEED,
        logger=logger,
    )

    logger.info(f"Train embryos: {sorted({r['embryo_id'] for r in train_rec})}")
    logger.info(f"Val embryos:   {used_val}")
    logger.info(f"Test embryos:  {used_test}")
    logger.info(f"Train counts: {counts_by_class(train_rec, classes)}")
    logger.info(f"Val counts:   {counts_by_class(val_rec, classes)}")
    logger.info(f"Test counts:  {counts_by_class(test_rec, classes)}")

    # Ensure each class appears in TRAIN
    train_classes_present = {r["cell_count"] for r in train_rec}
    missing_in_train = sorted(set(classes) - train_classes_present)
    if missing_in_train:
        raise RuntimeError(
            f"TRAIN split is missing class(es) {missing_in_train}. "
            "Change VAL_EMBRYO_IDS / TEST_EMBRYO_IDS so TRAIN contains all classes."
        )

    # Transforms
    tf_train = build_transforms(IMAGE_SIZE, train=True)
    tf_eval = build_transforms(IMAGE_SIZE, train=False)

    # Datasets
    ds_train = CellCountDataset(train_rec, IMG_ROOT, class_to_index, transform=tf_train, logger=logger)
    ds_val = CellCountDataset(val_rec, IMG_ROOT, class_to_index, transform=tf_eval, logger=logger)
    ds_test = CellCountDataset(test_rec, IMG_ROOT, class_to_index, transform=tf_eval, logger=logger)

    # Optional dataset sanity check (catches I/O/augment issues early)
    if RUN_DATASET_SANITY_CHECK:
        dataset_sanity_check(ds_train, out_dir, logger, prefix="train")
        dataset_sanity_check(ds_val, out_dir, logger, prefix="val")

    # Sampling / weights
    y_train_idx = np.array([class_to_index[int(r["cell_count"])] for r in train_rec], dtype=np.int64)

    sampler = None
    if USE_OVERSAMPLING:
        sampler = build_sampler_for_oversampling(y_train_idx, num_classes=len(classes), power=OVERSAMPLE_POWER)
        logger.info(f"Oversampling ENABLED (power={OVERSAMPLE_POWER}).")
    else:
        logger.info("Oversampling DISABLED.")

    # Collate behavior for bad samples
    bad_counter = {"count": 0}
    collate = None
    if not FAIL_FAST_ON_BAD_SAMPLE:
        collate = lambda batch: filtering_collate_fn(batch, bad_counter=bad_counter, logger=logger)

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    # Model
    model = build_model(num_classes=len(classes), arch=ARCH, pretrained=PRETRAINED).to(device)
    logger.info(f"Model created: {ARCH} | num_classes={len(classes)}")

    if FREEZE_BACKBONE_EPOCHS > 0:
        freeze_backbone(model)
        logger.info(f"Freezing backbone for first {FREEZE_BACKBONE_EPOCHS} epoch(s).")

    # Loss
    if USE_CLASS_WEIGHTS:
        w = compute_class_weights(y_train_idx, num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        logger.info(f"Using class weights: {w.detach().cpu().numpy().round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Class weights DISABLED.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = make_plateau_scheduler(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # Train loop
    for epoch in range(1, EPOCHS + 1):
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = make_plateau_scheduler(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                verbose=True,
            )
            logger.info("Unfroze full backbone; reset optimizer/scheduler.")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_epoch = time.time()
        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, device, logger, epoch, EPOCHS)
        val_acc, val_f1, y_true_val, y_pred_val = eval_model(model, dl_val, device, logger, tag="VAL")
        epoch_time = time.time() - t_epoch

        current_lr = float(optimizer.param_groups[0]["lr"])
        msg = f"Epoch {epoch:03d}/{EPOCHS} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f} | lr={current_lr:.2e} | time={epoch_time:.1f}s"
        if SHOW_GPU_MEMORY and torch.cuda.is_available():
            msg += f" | {gpu_mem_string()}"
        logger.info(msg)

        # Step scheduler + log LR reduction (more informative than scheduler.verbose)
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_f1)
        lr_after = float(optimizer.param_groups[0]["lr"])
        if lr_after < lr_before:
            logger.info(f"[LR] ReduceLROnPlateau reduced LR: {lr_before:.3e} -> {lr_after:.3e}")

        # Save last
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "classes": classes,
            "arch": ARCH,
            "image_size": IMAGE_SIZE,
            "val_embryos": used_val,
            "test_embryos": used_test,
        }, last_path)

        # Early-stopping bookkeeping
        improved = (val_f1 > best_f1 + EARLY_STOPPING_MIN_DELTA) if USE_EARLY_STOPPING else (val_f1 > best_f1)

        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "classes": classes,
                "arch": ARCH,
                "image_size": IMAGE_SIZE,
                "val_embryos": used_val,
                "test_embryos": used_test,
            }, best_path)
            logger.info(f"Saved new BEST -> {best_path} (macroF1={best_f1:.4f})")
        else:
            if USE_EARLY_STOPPING:
                epochs_no_improve += 1

        # Per-epoch report to text
        write_epoch_report(
            report_path=report_path,
            epoch=epoch,
            epochs=EPOCHS,
            train_loss=train_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            classes=classes,
            y_true=y_true_val,
            y_pred=y_pred_val,
            lr=current_lr,
            improved=improved,
            epochs_no_improve=epochs_no_improve,
            best_f1=best_f1,
        )

        if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early-stopping: no improvement in val_macroF1 for {EARLY_STOPPING_PATIENCE} epoch(s). "
                f"Best macroF1={best_f1:.4f} at epoch {best_epoch}."
            )
            break

    # Load best and evaluate on VAL + TEST (final)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_acc, val_f1, y_true_val, y_pred_val = eval_model(model, dl_val, device, logger, tag="VAL(best)")
    test_acc, test_f1, y_true_test, y_pred_test = eval_model(model, dl_test, device, logger, tag="TEST(best)")

    val_rep = classification_report(y_true_val, y_pred_val, target_names=[str(c) for c in classes], digits=4, zero_division=0)
    test_rep = classification_report(y_true_test, y_pred_test, target_names=[str(c) for c in classes], digits=4, zero_division=0)

    with final_path.open("w", encoding="utf-8") as f:
        f.write("FINAL EVAL (BEST checkpoint)\n")
        f.write(f"Best epoch: {int(ckpt.get('epoch', -1))}\n")
        f.write(f"VAL embryos: {ckpt.get('val_embryos')}\n")
        f.write(f"TEST embryos: {ckpt.get('test_embryos')}\n\n")
        f.write(f"[VAL]  acc={val_acc:.4f} | macroF1={val_f1:.4f}\n{val_rep}\n\n")
        f.write(f"[TEST] acc={test_acc:.4f} | macroF1={test_f1:.4f}\n{test_rep}\n")

    logger.info("=== FINAL EVAL (BEST) ===")
    logger.info(f"[VAL]  acc={val_acc:.4f} | macroF1={val_f1:.4f}")
    logger.info(f"[TEST] acc={test_acc:.4f} | macroF1={test_f1:.4f}")
    logger.info(f"Saved epoch reports: {report_path}")
    logger.info(f"Saved final eval:    {final_path}")
    logger.info(f"Best checkpoint:     {best_path}")
    logger.info(f"Last checkpoint:     {last_path}")
    logger.info("=== TRAINING DONE ===")

#%% 
# ============================================================
# Prediction / Inference
# ============================================================

@torch.no_grad()
def run_prediction() -> None:
    faulthandler.enable()
    out_dir = ensure_out_dir(OUT_DIR)
    logger = setup_logging(out_dir)
    logger.info("=== START PREDICTION ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(gpu_mem_string())

    root = Path(IMG_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"IMG_ROOT does not exist: {IMG_ROOT}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "resnet18")
    image_size = int(ckpt.get("image_size", 400))

    class_to_index = {c: i for i, c in enumerate(classes)}
    index_to_class = {i: c for c, i in class_to_index.items()}

    model = build_model(num_classes=len(classes), arch=arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {CKPT_PATH}")
    logger.info(f"Model: {arch} | classes={classes} | image_size={image_size}")

    # Collect files for prediction
    if PREDICT_GLOB is None:
        files = [fp for fp in root.rglob("*") if fp.is_file() and fp.suffix.lower() in EXTENSIONS]
    else:
        files = [fp for fp in root.glob(PREDICT_GLOB) if fp.is_file()]
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No files found for prediction.")

    logger.info(f"Predicting {len(files)} files...")

    # Build records; label is optional
    records = []
    for fp in files:
        embryo_id = None
        true_count = None
        try:
            embryo_id, true_count = parse_embryo_id_and_count(fp.name)
        except Exception:
            pass

        records.append({
            "filepath": str(fp.relative_to(root)),
            "embryo_id": embryo_id,
            "true_cell_count": true_count
        })

    # Dataset needs a label; use dummy label if missing/outside classes
    dummy_count = classes[0]
    fixed_records = []
    for r in records:
        fixed_records.append({
            "filepath": r["filepath"],
            "embryo_id": (r["embryo_id"] if r["embryo_id"] is not None else -1),
            "cell_count": (r["true_cell_count"] if (r["true_cell_count"] in classes) else dummy_count),
        })

    tf = build_transforms(image_size, train=False)
    ds = CellCountDataset(fixed_records, IMG_ROOT, class_to_index, transform=tf, logger=logger)

    bad_counter = {"count": 0}
    collate = None
    if not FAIL_FAST_ON_BAD_SAMPLE:
        collate = lambda batch: filtering_collate_fn(batch, bad_counter=bad_counter, logger=logger)

    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    all_probs = []
    all_pred = []

    iterator = dl
    if SHOW_TQDM and tqdm is not None:
        iterator = tqdm(dl, desc="Inference", unit="batch")

    for batch in iterator:
        if batch is None:
            continue
        x, _ = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_pred.extend(pred_idx.cpu().numpy().tolist())

    probs_np = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.zeros((0, len(classes)), dtype=np.float32)
    pred_counts = [index_to_class[int(i)] for i in all_pred]

    # Save CSV
    out_lines = []
    header = ["filepath", "embryo_id", "true_cell_count", "pred_cell_count"] + [f"p_{c}" for c in classes]
    out_lines.append(",".join(header))

    for i, r in enumerate(records):
        embryo_str = "" if r["embryo_id"] is None else str(r["embryo_id"])
        true_str = "" if r["true_cell_count"] is None else str(r["true_cell_count"])
        row = [r["filepath"], embryo_str, true_str, str(pred_counts[i])]
        row += [f"{probs_np[i, j]:.6f}" for j in range(len(classes))]
        out_lines.append(",".join(row))

    Path(PREDICT_OUTPUT_CSV).write_text("\n".join(out_lines), encoding="utf-8")
    logger.info(f"Saved predictions to: {PREDICT_OUTPUT_CSV}")
    logger.info("=== PREDICTION DONE ===")

#%% 
def main():
    try:
        if MODE == "train":
            run_training()
        elif MODE == "predict":
            run_prediction()
        else:
            raise ValueError("MODE must be 'train' or 'predict'")
    except Exception:
        # Print full traceback to terminal and to log file if enabled
        tb = traceback.format_exc()
        # logger may not exist yet; fallback to stderr
        print("\n[FATAL] Unhandled exception:\n" + tb, file=sys.stderr)
        # if OUT_DIR exists, try write crash log
        try:
            out_dir = ensure_out_dir(OUT_DIR)
            crash_path = out_dir / "crash_traceback.txt"
            crash_path.write_text(tb, encoding="utf-8")
            print(f"[FATAL] Crash traceback saved to: {crash_path}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

# %%
