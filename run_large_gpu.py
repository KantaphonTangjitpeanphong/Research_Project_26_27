r"""
run_large_gpu.py - Large CNN on the GPU
=======================================
A-Level Computer Science research project:
"When training an AI CNN model, is a GPU or a CPU better value for money once
accounting for time, accuracy, electricity, and hardware, and does the answer
change as the model gets bigger?"

This is one of the four run_*.py scripts (small/large model x CPU/GPU). They
are identical apart from the RUN SETTINGS block below, so all four use the same
data split, seeds, hyperparameter trials, training loop and measurements; only
the model and the device change. Written for Windows 11 + an NVIDIA GPU with
the PyTorch cu128 build.

HOW TO RUN (in the project folder, with the virtual environment activated)
    python run_large_gpu.py --smoke   quick test: 1 trial, 2 epochs, 1 in 20 images,
                                      written to results\smoke\ (wiped every time)
    python run_large_gpu.py           the real experiment

Training starts straight away and never waits for input. If a run stops
(crash, power cut, Ctrl+C), run the same command again: finished trials are
kept and the unfinished trial carries on from its last checkpoint. To redo a
run from nothing, delete its folder in results\.

    Ctrl+C once   finish the current epoch, save a checkpoint, stop
    Ctrl+C twice  quit immediately (work since the last checkpoint is lost)

OUTPUT, in results\<RUN_NAME>\
    models\<Size> CNN <DEVICE> Trained <N>t_<E>E.pt
                                the best model: highest validation accuracy over
                                every epoch of every trial, plus its settings
    summary_results_<RUN_NAME>.csv  one row: means and totals over all trials
    full_results_<RUN_NAME>.csv     one row per epoch per trial
    trials_<RUN_NAME>.csv           one row per trial: hyperparameters, time, kWh, cost
    test_report_<RUN_NAME>.txt      test accuracy, per-class metrics, confusion matrix
    run_info.json                   hardware, library versions, settings, sessions
    optuna.db                       Optuna study of finished trials (used for resuming)
    checkpoint.pt                   the unfinished trial (deleted when it finishes)
    codecarbon\emissions.csv        raw CodeCarbon log, one row per trial session
    error_log.txt                   only written if something crashed
splits.json in the project folder holds the train/validation split shared by
all four scripts. Whichever script runs first creates it; the rest reuse it.

Trials are numbered from 0 (trial N uses seed 42 + N), epochs from 1.
"""

# =============================================================================
# RUN SETTINGS - the only part that is different in each of the four scripts
# =============================================================================
RUN_NAME = "large_gpu"   # name of this run's folder inside results
MODEL_SIZE = "large"     # "small" = SmallCNN, "large" = LargeCNN
DEVICE_TYPE = "gpu"      # "gpu" = NVIDIA GPU via CUDA (required), "cpu" = CPU only
N_TRIALS = 50            # Optuna trials
EPOCHS = 100             # epochs per trial (no early stopping, no pruning)

# =============================================================================
# SHARED SETTINGS - keep these the same in all four scripts
# =============================================================================
# Folder that contains the dataset's Training and Testing folders, for example
#     DATA_DIR = r"C:\Users\<you>\Datasets\brain-tumor-mri-dataset"
# None = find it with kagglehub like the older scripts did (downloads it the
# first time, then reuses the cached copy). The folder used is printed at startup.
DATA_DIR = None

ELECTRICITY_PRICE = 4.18     # THB per kWh. Placeholder: check your latest bill.
CPU_THREADS = 6              # CPU scripts: one PyTorch thread per physical core (Ryzen 5 7600)

IMG_SIZE = 64                # images are made greyscale and resized to 64x64
VAL_FRACTION = 0.15          # validation share of the official Training folder (stratified)
SPLIT_SEED = 42
BASE_SEED = 42               # trial N is seeded with BASE_SEED + N
HP_SEED = 42                 # seed of Optuna's RandomSampler

LR_RANGE = (1e-4, 1e-2)               # learning rate, log scale
BATCH_SIZES = [32, 64, 128]
WEIGHT_DECAY_RANGE = (1e-6, 1e-3)     # log scale
DROPOUT_RANGE = (0.1, 0.5)

AUG_FLIP_PROB = 0.5          # training images only: random horizontal flip
AUG_MAX_ROTATION_DEG = 10    # and a random rotation of up to +/- 10 degrees
EVAL_BATCH_SIZE = 256        # validation/test batch size (does not change any result)

MEASURE_POWER_SECS = 5       # CodeCarbon and nvidia-smi sampling interval
CHECKPOINT_EVERY_SEC = 120   # save the unfinished trial at most this often (0 = every epoch)
ALLOW_TF32 = False           # False = true FP32 maths on the GPU, the same as the CPU
CUDNN_BENCHMARK = False      # True can be faster, but adds one-off tuning time to
                             # whichever trial first uses each batch size
DISABLE_QUICK_EDIT = True    # see disable_quick_edit()

SMOKE_TRIALS = 1             # --smoke: 1 trial x 2 epochs on 1 in 20 images
SMOKE_EPOCHS = 2
SMOKE_DATA_STEP = 20

# =============================================================================

import os
import sys

if DEVICE_TYPE == "cpu":
    # Hide the GPU from PyTorch before torch is imported, so nothing can run on it.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import math
import platform
import random
import shutil
import signal
import subprocess
import threading
import time
import traceback
import warnings
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import psutil  # installed together with codecarbon
except ImportError:
    psutil = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

IS_WINDOWS = sys.platform == "win32"
BASE_DIR = Path(__file__).resolve().parent
SCRIPT = Path(__file__).name
SPLITS_FILE = BASE_DIR / "splits.json"
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEVICE = torch.device("cuda" if DEVICE_TYPE == "gpu" else "cpu")
PROCESS = psutil.Process() if psutil else None

# Measurements that are added up over sessions when a trial is resumed.
SUMMED_MEASUREMENTS = ("energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
                       "wall_time_sec", "checkpoint_sec", "nvsmi_gpu_kwh")
ENERGY_FIELDS = ("energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "emissions_kg")

STOP_REQUESTED = False   # set by the first Ctrl+C


class SetupError(Exception):
    """A setup problem (data, CUDA, ...) that needs fixing before training can run."""


# =============================================================================
# SMALL HELPERS
# =============================================================================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_hms(seconds):
    if is_missing(seconds):
        return "?"
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def fmt_num(value, spec):
    return "?" if is_missing(value) else format(value, spec)


def add_or_none(a, b):
    """a + b, or None if either is unknown."""
    return None if a is None or b is None else a + b


def max_or_none(a, b):
    if a is None:
        return b
    return a if b is None else max(a, b)


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def json_safe(value):
    return None if is_missing(value) else float(value)


def replace_with_retry(src, dst, attempts=10):
    """os.replace, retried for a while: on Windows, antivirus or OneDrive can hold a
    freshly written file open for a moment."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


def write_json(path, data, indent=2):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    replace_with_retry(tmp, path)


def atomic_torch_save(obj, path):
    """Write to a temporary file first, so a power cut never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    replace_with_retry(tmp, path)


def safe_write_csv(df, path):
    """Rewrite a results CSV. If it is open in Excel (Windows locks it), warn and carry on;
    every CSV is rewritten in full after each trial, so nothing is lost."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        replace_with_retry(tmp, path, attempts=3)
    except PermissionError:
        tqdm.write(f"Warning: could not update {path.name} - is it open in Excel? "
                   "Training carries on; it is rewritten after the next trial.")
        tmp.unlink(missing_ok=True)


# =============================================================================
# WINDOWS HELPERS
# =============================================================================
def keep_pc_awake(enable):
    """Stop Windows going to sleep during training (the screen may still switch off)."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_state.argtypes = [ctypes.c_uint]
        set_state.restype = ctypes.c_uint
        set_state((ES_CONTINUOUS | ES_SYSTEM_REQUIRED) if enable else ES_CONTINUOUS)
    except Exception as e:
        print(f"Warning: could not change the Windows sleep setting ({e})")


def _console_input():
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    STD_INPUT_HANDLE = -10 & 0xFFFFFFFF
    return kernel32, kernel32.GetStdHandle(STD_INPUT_HANDLE), ctypes, wintypes


def disable_quick_edit():
    """Turn off the console's QuickEdit mode while training. With it on, clicking in a
    classic console window freezes the program until a key is pressed, which would
    silently add time to a trial. Returns the old mode so it can be restored."""
    if not IS_WINDOWS:
        return None
    try:
        kernel32, handle, ctypes, wintypes = _console_input()
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return None  # not a classic console (e.g. an IDE terminal): nothing to do
        ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
        kernel32.SetConsoleMode(handle, (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE)
        return mode.value
    except Exception:
        return None


def restore_console_mode(old_mode):
    if old_mode is None:
        return
    try:
        kernel32, handle, _, _ = _console_input()
        kernel32.SetConsoleMode(handle, old_mode)
    except Exception:
        pass


def beep(times=3, freq=1000):
    for _ in range(times):
        try:
            if IS_WINDOWS:
                import winsound
                winsound.Beep(freq, 400)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass
        time.sleep(0.2)


def notify(title, message):
    """Windows toast notification if plyer is installed; silently skipped otherwise."""
    try:
        from plyer import notification
        notification.notify(title=title, message=message[:250], app_name="CNN training", timeout=15)
    except Exception:
        pass


def cpu_name():
    if IS_WINDOWS:
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def run_nvidia_smi(fields):
    """nvidia-smi's answer for --query-gpu=<fields> on GPU 0, or None if it fails."""
    flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0  # no console window flashing up
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def handle_ctrl_c(signum, frame):
    """First Ctrl+C: stop cleanly after the current epoch. Second: stop right now."""
    global STOP_REQUESTED
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    STOP_REQUESTED = True
    tqdm.write("\nCtrl+C: the current epoch will finish, then a checkpoint is saved and the "
               "script stops. Press Ctrl+C again to quit immediately.")


# =============================================================================
# MODELS - both end in global average pooling, so neither has a big dense layer
# =============================================================================
def conv_bn_relu(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class SmallCNN(nn.Module):
    """3 blocks of Conv3x3-BatchNorm-ReLU-MaxPool (16, 32, 64 channels), then GAP -> Dropout -> Linear."""

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (16, 32, 64):
            layers += conv_bn_relu(c_in, c_out) + [nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


class LargeCNN(nn.Module):
    """5 blocks of 2 x (Conv3x3-BatchNorm-ReLU) + MaxPool (32, 64, 128, 256, 512 channels),
    then GAP -> Dropout -> Linear."""

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (32, 64, 128, 256, 512):
            layers += conv_bn_relu(c_in, c_out) + conv_bn_relu(c_out, c_out) + [nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


MODEL_CLASSES = {"small": SmallCNN, "large": LargeCNN}


def build_model(dropout, num_classes):
    return MODEL_CLASSES[MODEL_SIZE](num_classes=num_classes, dropout=dropout)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# DATA - every image is read once and kept in memory, so disk speed never
# affects the timings
# =============================================================================
def find_dataset_root():
    """The folder that contains Training and Testing."""
    if DATA_DIR:
        base = Path(DATA_DIR)
        if not base.exists():
            raise SetupError(f"DATA_DIR does not exist: {base}")
    else:
        try:
            import kagglehub
        except ImportError:
            raise SetupError("DATA_DIR is not set and kagglehub is not installed. Set DATA_DIR at the "
                             "top of this script to the folder that contains Training and Testing.") from None
        print("DATA_DIR is not set, so asking kagglehub for the dataset ...")
        base = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    candidates = [base]
    candidates += sorted(p for p in base.glob("*") if p.is_dir())
    candidates += sorted(p for p in base.glob("*/*") if p.is_dir())
    for folder in candidates:
        if (folder / "Training").is_dir() and (folder / "Testing").is_dir():
            return folder
    raise SetupError(f"Could not find Training and Testing folders inside {base}. Check DATA_DIR.")


def list_images(folder, class_names=None):
    """(class_names, [(relative path, label), ...]) for the Training or Testing folder."""
    if class_names is None:
        class_names = sorted(p.name for p in folder.iterdir() if p.is_dir())
    entries = []
    for label, name in enumerate(class_names):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise SetupError(f"Missing class folder: {class_dir}")
        files = sorted((p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                       key=lambda p: p.name.lower())
        entries += [(f"{name}/{p.name}", label) for p in files]
    return class_names, entries


def load_images(folder, entries, description):
    """Read, greyscale and resize images into a uint8 tensor [N, 1, H, W] plus labels."""
    pixels = np.empty((len(entries), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, (rel_path, _) in enumerate(tqdm(entries, desc=description, unit="img", leave=False)):
        with Image.open(folder / rel_path) as img:
            img = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            pixels[i] = np.asarray(img)
    labels = torch.tensor([label for _, label in entries], dtype=torch.long)
    return torch.from_numpy(pixels).unsqueeze(1), labels


def prepare_data(cfg):
    root = find_dataset_root()
    print(f"Dataset folder: {root}")
    train_dir, test_dir = root / "Training", root / "Testing"
    class_names, all_train = list_images(train_dir)
    _, test_entries = list_images(test_dir, class_names)

    preloaded = None
    if SPLITS_FILE.exists():
        split = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
        if split["class_names"] != class_names:
            raise SetupError(f"{SPLITS_FILE.name} was made for classes {split['class_names']}, "
                             f"but the dataset has {class_names}.")
        if split["img_size"] != IMG_SIZE:
            raise SetupError(f"{SPLITS_FILE.name} was made for {split['img_size']}px images, not {IMG_SIZE}px.")
        print(f"Using the saved train/validation split in {SPLITS_FILE.name}")
    else:
        print(f"No {SPLITS_FILE.name} yet: creating the train/validation split (only happens once) ...")
        labels = [label for _, label in all_train]
        train_idx, val_idx = train_test_split(np.arange(len(all_train)), test_size=VAL_FRACTION,
                                              stratify=labels, random_state=SPLIT_SEED)
        train_entries = [all_train[i] for i in sorted(train_idx)]
        val_entries = [all_train[i] for i in sorted(val_idx)]
        x_train, y_train = load_images(train_dir, train_entries, "Loading training images")
        scaled = x_train.double() / 255
        split = {
            "created": now_iso(), "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
            "img_size": IMG_SIZE, "class_names": class_names,
            # normalisation statistics, from the training images only
            "norm_mean": scaled.mean().item(), "norm_std": scaled.std().item(),
            "n_train": len(train_entries), "n_val": len(val_entries),
            "train": train_entries, "val": val_entries,
        }
        del scaled
        write_json(SPLITS_FILE, split, indent=1)
        preloaded = (x_train, y_train)

    train_entries = [(rel, int(label)) for rel, label in split["train"]]
    val_entries = [(rel, int(label)) for rel, label in split["val"]]
    missing = [rel for rel, _ in train_entries + val_entries if not (train_dir / rel).is_file()]
    if missing:
        raise SetupError(f"{len(missing)} images listed in {SPLITS_FILE.name} are missing from {train_dir} "
                         f"(first one: {missing[0]}). Point DATA_DIR at the same copy of the dataset.")

    if cfg.smoke:  # every 20th image; the lists are sorted by class, so each class stays represented
        train_entries = train_entries[::SMOKE_DATA_STEP]
        val_entries = val_entries[::SMOKE_DATA_STEP]
        test_entries = test_entries[::SMOKE_DATA_STEP]
        preloaded = None

    if preloaded is not None:
        x_train, y_train = preloaded
    else:
        x_train, y_train = load_images(train_dir, train_entries, "Loading training images")
    x_val, y_val = load_images(train_dir, val_entries, "Loading validation images")
    x_test, y_test = load_images(test_dir, test_entries, "Loading test images")
    return SimpleNamespace(root=root, class_names=class_names, norm=(split["norm_mean"], split["norm_std"]),
                           x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                           x_test=x_test, y_test=y_test)


# =============================================================================
# HYPERPARAMETERS (Optuna)
# =============================================================================
def suggest_hyperparameters(trial):
    return {
        "lr": trial.suggest_float("lr", *LR_RANGE, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "weight_decay": trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True),
        "dropout": trial.suggest_float("dropout", *DROPOUT_RANGE),
    }


def search_space():
    return {
        "lr": optuna.distributions.FloatDistribution(*LR_RANGE, log=True),
        "batch_size": optuna.distributions.CategoricalDistribution(BATCH_SIZES),
        "weight_decay": optuna.distributions.FloatDistribution(*WEIGHT_DECAY_RANGE, log=True),
        "dropout": optuna.distributions.FloatDistribution(*DROPOUT_RANGE),
    }


def build_trial_plan(n_trials):
    """Hyperparameters for trials 0 .. n-1, drawn from Optuna's RandomSampler(seed=HP_SEED).

    All draws are made up front in a throw-away in-memory study, so trial N gets the
    same settings in all four scripts (the 10 large_cpu trials match the first 10
    large_gpu trials) and keeps them when a run is resumed. Asking a freshly seeded
    sampler after a restart would start its sequence again and repeat earlier
    trials' settings. A random sampler is used rather than TPE because TPE adapts
    to results, so the CPU and GPU runs would end up testing different settings."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=HP_SEED))
    plan = []
    for _ in range(n_trials):
        trial = study.ask()
        plan.append(suggest_hyperparameters(trial))
        study.tell(trial, 0.0)
    return plan


def open_study(cfg):
    """The run's Optuna study in SQLite. Each finished trial is added to it straight
    away, so a re-run knows which trials are already done."""
    storage = f"sqlite:///{(cfg.out_dir / 'optuna.db').as_posix()}"
    return optuna.create_study(study_name=RUN_NAME, storage=storage, direction="maximize",
                               sampler=optuna.samplers.RandomSampler(seed=HP_SEED), load_if_exists=True)


def completed_trials(study):
    """{trial number: finished Optuna trial}"""
    done = {}
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if "trial" in t.user_attrs:
            done[int(t.user_attrs["trial"])] = t
    return done


def record_trial(study, result, epoch_log, params):
    study.add_trial(optuna.trial.create_trial(
        params=params, distributions=search_space(), value=result["best_val_acc"],
        user_attrs={"trial": result["trial"], "result": result, "epoch_log": epoch_log}))


# =============================================================================
# TRAINING
# =============================================================================
def sync():
    """Wait for the GPU to finish queued work, so the clock measures real work."""
    if DEVICE_TYPE == "gpu":
        torch.cuda.synchronize()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU


def to_device(x, y):
    """Move a uint8 image batch and its labels to the training device
    (pinned memory + non_blocking on the GPU)."""
    if DEVICE_TYPE == "gpu":
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    return x, y


def normalise(x, norm):
    return (x - norm[0]) / norm[1]


def augment(x, gen):
    """Random horizontal flip + small rotation, applied on the training device.
    The random numbers come from the trial's own CPU generator, so the CPU and the
    GPU run of a trial get exactly the same flips and angles."""
    n = x.size(0)
    flip = (torch.rand(n, generator=gen) < AUG_FLIP_PROB).to(x.device)
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    angle = (torch.rand(n, generator=gen) * 2 - 1) * math.radians(AUG_MAX_ROTATION_DEG)
    cos, sin, zero = torch.cos(angle), torch.sin(angle), torch.zeros(n)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def train_one_epoch(model, data, batch_size, optimizer, criterion, gen):
    """One pass over the training images. Batches are sliced straight from the
    in-memory tensor (no DataLoader worker processes, which avoids Windows
    multiprocessing problems). Returns (mean loss, accuracy)."""
    model.train()
    n = data.x_train.size(0)
    order = torch.randperm(n, generator=gen)
    loss_sum = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), dtype=torch.long, device=DEVICE)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        x, y = to_device(data.x_train[idx], data.y_train[idx])
        x = normalise(augment(x.float().div_(255), gen), data.norm)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * y.size(0)  # kept on the device: no GPU sync per batch
        correct += (out.argmax(1) == y).sum()
    return loss_sum.item() / n, correct.item() / n


@torch.no_grad()
def evaluate(model, x_all, y_all, norm, criterion, want_predictions=False):
    """(mean loss, accuracy, predictions or None) on a validation/test set."""
    model.eval()
    n = x_all.size(0)
    loss_sum = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), dtype=torch.long, device=DEVICE)
    predictions = []
    for start in range(0, n, EVAL_BATCH_SIZE):
        x, y = to_device(x_all[start:start + EVAL_BATCH_SIZE], y_all[start:start + EVAL_BATCH_SIZE])
        out = model(normalise(x.float().div_(255), norm))
        loss_sum += criterion(out, y) * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum()
        if want_predictions:
            predictions.append(pred.cpu())
    return loss_sum.item() / n, correct.item() / n, (torch.cat(predictions) if want_predictions else None)


# =============================================================================
# MEASUREMENT
# =============================================================================
class GpuPowerSampler:
    """Cross-check for CodeCarbon: reads nvidia-smi power.draw every few seconds in a
    background thread and integrates it (trapezium rule) into kWh."""

    def __init__(self, interval):
        self.interval = interval
        self.kwh = 0.0
        self.samples = 0
        self._last = None  # (time, watts)
        self._stop = threading.Event()
        self._thread = None

    def _read_watts(self):
        text = run_nvidia_smi("power.draw")
        try:
            return float(text.splitlines()[0])
        except (AttributeError, ValueError, IndexError):
            return None  # nvidia-smi missing, or it reported [N/A]

    def _sample(self):
        watts = self._read_watts()
        now = time.perf_counter()
        if watts is None:
            if self._last is None:
                return
            watts = self._last[1]  # carry the last reading over the gap
        else:
            self.samples += 1
        if self._last is not None:
            t0, w0 = self._last
            self.kwh += (w0 + watts) / 2 * (now - t0) / 3.6e6
        self._last = (now, watts)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=15)
            self._thread = None
            self._sample()
        return self.kwh


def codecarbon_final(tracker):
    """Energy measured by a stopped CodeCarbon tracker."""
    data = getattr(tracker, "final_emissions_data", None)
    if data is None:
        return dict.fromkeys(ENERGY_FIELDS)
    return {"energy_kwh": float(data.energy_consumed), "cpu_energy_kwh": float(data.cpu_energy),
            "gpu_energy_kwh": float(data.gpu_energy), "ram_energy_kwh": float(data.ram_energy),
            "emissions_kg": float(data.emissions)}


def codecarbon_so_far(tracker):
    """Energy a running tracker has measured so far (only needed for mid-trial
    checkpoints). CodeCarbon has no public call for this, so its running totals
    are read directly; if that ever stops working the values are None."""
    try:
        return {"energy_kwh": float(tracker._total_energy.kWh),
                "cpu_energy_kwh": float(tracker._total_cpu_energy.kWh),
                "gpu_energy_kwh": float(tracker._total_gpu_energy.kWh),
                "ram_energy_kwh": float(tracker._total_ram_energy.kWh),
                "emissions_kg": None}
    except Exception:
        return dict.fromkeys(ENERGY_FIELDS)


class TrialMeter:
    """Time and energy of one session of one trial: one CodeCarbon tracker, plus the
    nvidia-smi cross-check on the GPU."""

    def __init__(self, cfg, trial_index):
        self.tracker = EmissionsTracker(
            project_name=f"{RUN_NAME}_trial{trial_index:03d}",
            output_dir=str(cfg.codecarbon_dir), output_file="emissions.csv",
            measure_power_secs=MEASURE_POWER_SECS, log_level="error",
            save_to_file=True, allow_multiple_runs=True)
        self.gpu_sampler = GpuPowerSampler(MEASURE_POWER_SECS) if DEVICE_TYPE == "gpu" else None
        self.checkpoint_sec = 0.0
        self.t0 = None
        self.final = None

    def start(self):
        self.tracker.start()
        if self.gpu_sampler:
            self.gpu_sampler.start()
        self.t0 = time.perf_counter()

    def read(self, final=False):
        """This session's measurements so far; final=True stops measuring first."""
        if self.final is not None:
            return self.final
        wall = time.perf_counter() - self.t0
        if final:
            self.tracker.stop()
            values = codecarbon_final(self.tracker)
            nvsmi = self.gpu_sampler.stop() if self.gpu_sampler else None
        else:
            values = codecarbon_so_far(self.tracker)
            nvsmi = self.gpu_sampler.kwh if self.gpu_sampler else None
        if self.gpu_sampler and self.gpu_sampler.samples == 0:
            nvsmi = None  # nvidia-smi could not be read
        values.update(wall_time_sec=wall, checkpoint_sec=self.checkpoint_sec, nvsmi_gpu_kwh=nvsmi)
        if final:
            self.final = values
        return values

    def abort(self):
        """Stop measuring after an error, ignoring any further errors."""
        if self.t0 is not None and self.final is None:
            try:
                self.read(final=True)
            except Exception:
                pass


def process_ram_mb():
    return PROCESS.memory_info().rss / 1024**2 if PROCESS else None


class Progress:
    """Elapsed time and ETA for this session: average time per epoch so far x epochs left."""

    def __init__(self, epochs_left):
        self.epochs_left = epochs_left
        self.epochs_done = 0
        self.t0 = None
        self.announced = False

    def start(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()

    def epoch_finished(self):
        self.epochs_done += 1
        self.epochs_left = max(self.epochs_left - 1, 0)

    def elapsed(self):
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def eta(self):
        return self.elapsed() / self.epochs_done * self.epochs_left if self.epochs_done else None

    def announce_estimate(self):
        """Printed once, after the first epoch of this session. Never waits for input."""
        self.announced = True
        eta = self.eta()
        finish = (datetime.now() + timedelta(seconds=eta)).strftime("%a %d %b %H:%M")
        tqdm.write(f"\n  Estimated time for the {self.epochs_left} epochs still to run: {eta / 3600:.1f} hours "
                   f"(finishing around {finish}).\n  This is rough: the batch size changes between trials and "
                   "the first epoch includes warm-up. The ETA on each epoch line keeps updating.\n")


# =============================================================================
# CHECKPOINTS
# =============================================================================
def capture_rng(gen):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if DEVICE_TYPE == "gpu" else None,
            "data": gen.get_state()}


def restore_rng(rng, gen):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if DEVICE_TYPE == "gpu" and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    gen.set_state(rng["data"])


def new_trial_state():
    """Everything about an unfinished trial that must survive a restart."""
    earlier = dict.fromkeys(SUMMED_MEASUREMENTS, 0.0)
    if DEVICE_TYPE != "gpu":
        earlier["nvsmi_gpu_kwh"] = None
    return {"epoch": 0, "best_val_acc": -1.0, "best_epoch": 0, "best_weights": None, "epoch_log": [],
            "sessions": 1, "peak_ram_mb": None, "peak_vram_mb": None, "earlier_sessions": earlier}


def save_checkpoint(cfg, trial_index, params, model, optimizer, scheduler, gen, state, meter):
    """Save the unfinished trial: model, optimiser, scheduler, RNG states, epoch, and the
    time/energy measured so far. After a crash, the epochs after the last checkpoint
    are redone, so every epoch's time and energy is counted exactly once.

    This runs every CHECKPOINT_EVERY_SEC rather than after every epoch: the large
    model's checkpoint is ~75 MB, and writing it after every 1-2 second GPU epoch
    would add noticeable time and energy to the GPU results. A power cut loses at
    most ~2 minutes of work."""
    started = time.perf_counter()
    so_far = meter.read()
    earlier = {k: add_or_none(state["earlier_sessions"][k], so_far[k]) for k in SUMMED_MEASUREMENTS}
    atomic_torch_save({
        "run_name": RUN_NAME, "trial": trial_index, "params": params, "epochs": cfg.epochs,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng(gen), "state": dict(state, earlier_sessions=earlier), "saved_at": now_iso(),
    }, cfg.checkpoint_path)
    meter.checkpoint_sec += time.perf_counter() - started


def load_checkpoint(cfg, plan, done):
    """The saved unfinished trial, or None. Checkpoints that do not belong to an
    unfinished trial of this run are removed."""
    path = cfg.checkpoint_path
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    except Exception as e:
        broken = path.with_name("checkpoint.unreadable.pt")
        replace_with_retry(path, broken)
        print(f"Warning: could not read the checkpoint ({e}). Moved it to {broken.name}; "
              "that trial starts again from epoch 1.")
        return None
    i = ckpt.get("trial")
    if (ckpt.get("run_name") != RUN_NAME or i is None or i in done or i >= len(plan)
            or ckpt.get("params") != plan[i] or ckpt.get("epochs") != cfg.epochs):
        path.unlink()
        if i not in done:
            print("Note: removed a checkpoint that does not match this run's settings.")
        return None
    return ckpt


# =============================================================================
# ONE TRIAL
# =============================================================================
def run_trial(cfg, data, trial_index, params, progress, checkpoint=None):
    """Train one trial for cfg.epochs epochs (or carry on from a checkpoint).
    Returns (result, epoch_log, best_weights), or None if stopped with Ctrl+C."""
    seed = BASE_SEED + trial_index
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)  # batch order + augmentation, identical on CPU and GPU
    # The weights are created on the CPU before moving, so they start identical on both devices.
    model = build_model(params["dropout"], len(data.class_names)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    if checkpoint is None:
        state = new_trial_state()
    else:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"], gen)
        state = checkpoint["state"]
        state["sessions"] += 1
        tqdm.write(f"Resuming trial {trial_index} after epoch {state['epoch']} (session {state['sessions']}).")

    if DEVICE_TYPE == "gpu":
        torch.cuda.reset_peak_memory_stats()
    meter = TrialMeter(cfg, trial_index)
    try:
        meter.start()
        last_save = time.perf_counter()
        for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
            progress.start()
            sync()
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, data, params["batch_size"], optimizer, criterion, gen)
            sync()
            train_sec = time.perf_counter() - t0
            val_loss, val_acc, _ = evaluate(model, data.x_val, data.y_val, data.norm, criterion)
            sync()
            val_sec = time.perf_counter() - t0 - train_sec
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            if val_acc > state["best_val_acc"]:  # keep the weights of this trial's best epoch
                state["best_val_acc"], state["best_epoch"] = val_acc, epoch
                state["best_weights"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["peak_ram_mb"] = max_or_none(state["peak_ram_mb"], process_ram_mb())
            state["epoch_log"].append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
                "duration_sec": train_sec, "val_sec": val_sec, "epoch_sec": train_sec + val_sec,
                "session": state["sessions"],
            })
            state["epoch"] = epoch
            progress.epoch_finished()
            tqdm.write(f"trial {trial_index:>2} epoch {epoch:>3}/{cfg.epochs} | train loss {train_loss:.4f} "
                       f"acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} | "
                       f"{train_sec + val_sec:6.2f}s | elapsed {fmt_hms(progress.elapsed())} | "
                       f"ETA {fmt_hms(progress.eta())}")
            if not progress.announced:
                progress.announce_estimate()

            if STOP_REQUESTED:
                meter.read(final=True)
                save_checkpoint(cfg, trial_index, params, model, optimizer, scheduler, gen, state, meter)
                tqdm.write(f"Checkpoint saved after epoch {epoch} of trial {trial_index}.")
                return None
            if epoch < cfg.epochs and time.perf_counter() - last_save >= cfg.checkpoint_every:
                save_checkpoint(cfg, trial_index, params, model, optimizer, scheduler, gen, state, meter)
                last_save = time.perf_counter()
        measured = meter.read(final=True)
    except BaseException:
        meter.abort()
        raise

    if DEVICE_TYPE == "gpu":
        state["peak_vram_mb"] = max_or_none(state["peak_vram_mb"], torch.cuda.max_memory_allocated() / 1024**2)
    totals = {k: add_or_none(state["earlier_sessions"][k], measured[k]) for k in SUMMED_MEASUREMENTS}
    energy = totals["energy_kwh"]
    # CO2 = energy x grid carbon intensity; this session's intensity is used for the whole trial
    intensity = None
    if measured["emissions_kg"] is not None and measured["energy_kwh"]:
        intensity = measured["emissions_kg"] / measured["energy_kwh"]
    log = state["epoch_log"]
    last = log[-1]
    result = {
        "trial": trial_index, "run_name": RUN_NAME, "model": MODEL_SIZE, "device": DEVICE_TYPE,
        "lr": params["lr"], "batch_size": params["batch_size"],
        "weight_decay": params["weight_decay"], "dropout": params["dropout"],
        "seed": seed, "epochs": cfg.epochs, "param_count": count_parameters(model),
        "best_val_acc": state["best_val_acc"], "best_epoch": state["best_epoch"],
        "final_val_acc": last["val_acc"], "final_val_loss": last["val_loss"],
        "final_train_acc": last["train_acc"], "final_train_loss": last["train_loss"],
        "train_time_sec": sum(e["duration_sec"] for e in log),   # training passes only
        "val_time_sec": sum(e["val_sec"] for e in log),
        "checkpoint_sec": totals["checkpoint_sec"],
        "wall_time_sec": totals["wall_time_sec"],                # everything; matches the energy figure
        "energy_kwh": energy, "cpu_energy_kwh": totals["cpu_energy_kwh"],
        "gpu_energy_kwh": totals["gpu_energy_kwh"], "ram_energy_kwh": totals["ram_energy_kwh"],
        "emissions_kg": energy * intensity if energy is not None and intensity is not None else None,
        "nvsmi_gpu_kwh": totals["nvsmi_gpu_kwh"],
        "cost_thb": energy * ELECTRICITY_PRICE if energy is not None else None,
        "peak_ram_mb": state["peak_ram_mb"], "peak_vram_mb": state["peak_vram_mb"],
        "resumed": state["sessions"] > 1, "sessions": state["sessions"],
        "finished_at": now_iso(),
    }
    return result, log, state["best_weights"]


# =============================================================================
# RESULTS
# =============================================================================
def save_best_model(cfg, data, result, params, best_weights):
    """Everything needed to rebuild and use the model later (e.g. from a test_model.py)."""
    atomic_torch_save({
        "state_dict": best_weights,
        "model_class": MODEL_CLASSES[MODEL_SIZE].__name__,
        "model_kwargs": {"num_classes": len(data.class_names), "dropout": params["dropout"]},
        "hyperparameters": dict(params),
        "class_names": list(data.class_names),
        "img_size": IMG_SIZE, "in_channels": 1,
        "normalisation": {"mean": data.norm[0], "std": data.norm[1]},
        "trial": result["trial"], "epoch": result["best_epoch"], "val_acc": result["best_val_acc"],
        "run_name": RUN_NAME, "trained_on": DEVICE_TYPE.upper(), "saved_at": now_iso(),
    }, cfg.model_path)


def summarise(trials_df, test_acc):
    """One row of means and totals over all finished trials."""
    energy = numeric(trials_df, "energy_kwh")
    wall = numeric(trials_df, "wall_time_sec")
    best_acc = numeric(trials_df, "best_val_acc")
    nvsmi = numeric(trials_df, "nvsmi_gpu_kwh")
    best = trials_df.loc[best_acc.idxmax()]
    return {
        "run_name": RUN_NAME, "model": MODEL_SIZE, "device": DEVICE_TYPE,
        "n_trials": len(trials_df), "epochs_per_trial": int(trials_df["epochs"].iloc[0]),
        "param_count": int(trials_df["param_count"].iloc[0]),
        "mean_energy_kwh": energy.mean(), "std_energy_kwh": energy.std(),
        "total_energy_kwh": energy.sum(min_count=1),
        "mean_time_sec": wall.mean(), "std_time_sec": wall.std(), "total_time_sec": wall.sum(min_count=1),
        "mean_train_time_sec": numeric(trials_df, "train_time_sec").mean(),
        "mean_final_val_acc": numeric(trials_df, "final_val_acc").mean(),
        "mean_best_val_acc": best_acc.mean(), "std_best_val_acc": best_acc.std(),
        "best_val_acc": best["best_val_acc"], "best_trial": int(best["trial"]), "best_epoch": int(best["best_epoch"]),
        "test_acc": test_acc,
        "est_cost_thb": energy.mean() * ELECTRICITY_PRICE,          # per trial, as in the older scripts
        "total_cost_thb": energy.sum(min_count=1) * ELECTRICITY_PRICE,
        "electricity_price_thb_per_kwh": ELECTRICITY_PRICE,
        "mean_nvsmi_gpu_kwh": nvsmi.mean(), "total_nvsmi_gpu_kwh": nvsmi.sum(min_count=1),
        "peak_ram_mb": numeric(trials_df, "peak_ram_mb").max(),
        "peak_vram_mb": numeric(trials_df, "peak_vram_mb").max(),
        "n_resumed_trials": int(trials_df["resumed"].sum()),
    }


def export_results(cfg, done, test_acc=None):
    """Rewrite the three results CSVs from the Optuna study. Returns the trials table."""
    if not done:
        return None
    order = sorted(done)
    trials_df = pd.DataFrame([done[i].user_attrs["result"] for i in order])
    epoch_rows = [{"run_name": RUN_NAME, "model": MODEL_SIZE, "device": DEVICE_TYPE, "trial": i, **row}
                  for i in order for row in done[i].user_attrs["epoch_log"]]
    safe_write_csv(trials_df, cfg.out_dir / f"trials_{RUN_NAME}.csv")
    safe_write_csv(pd.DataFrame(epoch_rows), cfg.out_dir / f"full_results_{RUN_NAME}.csv")
    safe_write_csv(pd.DataFrame([summarise(trials_df, test_acc)]), cfg.out_dir / f"summary_results_{RUN_NAME}.csv")
    return trials_df


def final_test(cfg, data):
    """Evaluate the saved best model once on the untouched Testing folder."""
    saved = torch.load(cfg.model_path, map_location="cpu", weights_only=True)
    model = build_model(saved["model_kwargs"]["dropout"], len(data.class_names)).to(DEVICE)
    model.load_state_dict(saved["state_dict"])
    _, test_acc, predictions = evaluate(model, data.x_test, data.y_test, data.norm, nn.CrossEntropyLoss(),
                                        want_predictions=True)
    labels = list(range(len(data.class_names)))
    y_true, y_pred = data.y_test.numpy(), predictions.numpy()
    report = classification_report(y_true, y_pred, labels=labels, target_names=data.class_names,
                                   digits=4, zero_division=0)
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                          index=[f"true {c}" for c in data.class_names],
                          columns=[f"pred {c}" for c in data.class_names])
    hp = saved["hyperparameters"]
    text = "\n".join([
        f"Test report - {RUN_NAME} ({MODEL_SIZE} CNN trained on the {DEVICE_TYPE.upper()})"
        + ("  [SMOKE TEST]" if cfg.smoke else ""),
        f"Created: {now_iso()}",
        f"Model file: {cfg.model_path}",
        f"Best trial: {saved['trial']} (epoch {saved['epoch']}), validation accuracy {saved['val_acc']:.4f}",
        f"Hyperparameters: lr={hp['lr']:.6g}, batch_size={hp['batch_size']}, "
        f"weight_decay={hp['weight_decay']:.6g}, dropout={hp['dropout']:.4f}",
        f"Test images: {len(y_true)} (Testing folder: never used for training or model selection)",
        "",
        f"Test accuracy: {test_acc:.4f}",
        "",
        "Per-class metrics:",
        report,
        "Confusion matrix (rows = true class, columns = predicted class):",
        matrix.to_string(),
        "",
    ])
    (cfg.out_dir / f"test_report_{RUN_NAME}.txt").write_text(text, encoding="utf-8")
    return test_acc, saved


def run_totals(trials_df):
    kwh = numeric(trials_df, "energy_kwh").sum(min_count=1)
    return {"total_time_sec": numeric(trials_df, "wall_time_sec").sum(),
            "total_energy_kwh": kwh, "total_cost_thb": kwh * ELECTRICITY_PRICE,
            "total_nvsmi_gpu_kwh": numeric(trials_df, "nvsmi_gpu_kwh").sum(min_count=1),
            "n_resumed_trials": int(trials_df["resumed"].sum())}


def print_complete_banner(cfg, trials_df, totals, test_acc, saved):
    rows = [
        ("Run", f"{RUN_NAME}  ({MODEL_SIZE} CNN on the {DEVICE_TYPE.upper()})" + ("  SMOKE TEST" if cfg.smoke else "")),
        ("Trials x epochs", f"{len(trials_df)} x {cfg.epochs}"),
        ("Total time", f"{fmt_hms(totals['total_time_sec'])}  (all trials, all sessions)"),
        ("Total energy", f"{fmt_num(totals['total_energy_kwh'], '.4f')} kWh  (CodeCarbon)"),
    ]
    if DEVICE_TYPE == "gpu":
        rows.append(("GPU energy", f"{fmt_num(totals['total_nvsmi_gpu_kwh'], '.4f')} kWh  (nvidia-smi cross-check)"))
    rows += [
        ("Total cost", f"{fmt_num(totals['total_cost_thb'], '.2f')} THB  at {ELECTRICITY_PRICE} THB/kWh"),
        ("Best val accuracy", f"{saved['val_acc']:.4f}  (trial {saved['trial']}, epoch {saved['epoch']})"),
        ("Test accuracy", f"{test_acc:.4f}"),
        ("Saved model", str(cfg.model_path)),
    ]
    if totals["n_resumed_trials"]:
        rows.append(("Resumed trials", f"{totals['n_resumed_trials']} (time/energy split over sessions; "
                                       "flag them in the analysis)"))
    line = "#" * 76
    print(f"\n{line}\n#{'TRAINING COMPLETE'.center(74)}#\n{line}")
    for label, value in rows:
        print(f"  {label:<18} {value}")
    print(line + "\n")


# =============================================================================
# RUN INFO
# =============================================================================
def library_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "optuna", "codecarbon", "numpy", "pandas", "scikit-learn",
                 "pillow", "tqdm", "psutil", "plyer", "kagglehub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info():
    info = {"os": platform.platform(), "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads()}
    if psutil:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    if DEVICE_TYPE == "gpu":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["nvidia_driver"] = run_nvidia_smi("driver_version")
    return info


def settings_dict(cfg):
    names = ("N_TRIALS", "EPOCHS", "DATA_DIR", "ELECTRICITY_PRICE", "CPU_THREADS", "IMG_SIZE", "VAL_FRACTION",
             "SPLIT_SEED", "BASE_SEED", "HP_SEED", "LR_RANGE", "BATCH_SIZES", "WEIGHT_DECAY_RANGE",
             "DROPOUT_RANGE", "AUG_FLIP_PROB", "AUG_MAX_ROTATION_DEG", "EVAL_BATCH_SIZE", "MEASURE_POWER_SECS",
             "CHECKPOINT_EVERY_SEC", "ALLOW_TF32", "CUDNN_BENCHMARK")
    settings = {name: globals()[name] for name in names}
    settings.update(trials_this_run=cfg.n_trials, epochs_this_run=cfg.epochs, optimiser="AdamW",
                    loss="CrossEntropyLoss", lr_schedule="CosineAnnealingLR(T_max=epochs)",
                    sampler=f"Optuna RandomSampler(seed={HP_SEED})", precision="FP32 (no mixed precision)")
    return settings


def load_run_info(cfg):
    try:
        return json.loads((cfg.out_dir / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_run_info(cfg, info):
    try:
        write_json(cfg.out_dir / "run_info.json", info)
    except OSError as e:
        print(f"Warning: could not write run_info.json ({e})")


def log_error(cfg, error_text):
    try:
        with open(cfg.out_dir / "error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n===== {now_iso()} =====\n{error_text}\n")
    except OSError:
        pass


# =============================================================================
# MAIN
# =============================================================================
def setup_device():
    """Check and configure the device. Returns a one-line description."""
    if DEVICE_TYPE == "gpu":
        if not torch.cuda.is_available():
            raise SetupError(
                "CUDA is not available, so this GPU script cannot run.\n"
                f"  PyTorch {torch.__version__} (built for CUDA {torch.version.cuda}), Python: {sys.executable}\n"
                "  Most likely the virtual environment is not activated (wrong Python), or this is a\n"
                "  CPU-only PyTorch build. You need the cu128 build (version ending in +cu128).")
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # newer PyTorch warns that these flags are being renamed
            torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
            torch.backends.cudnn.allow_tf32 = ALLOW_TF32
        return (f"{torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} (PyTorch {torch.__version__}) | "
                f"driver {run_nvidia_smi('driver_version') or '?'} | TF32 {'on' if ALLOW_TF32 else 'off (pure FP32)'}")
    torch.set_num_threads(CPU_THREADS)
    return (f"{cpu_name()} | {torch.get_num_threads()} PyTorch threads | "
            f"GPU hidden from PyTorch: {'yes' if not torch.cuda.is_available() else 'NO'}")


def print_startup_summary(cfg, data, device_text, param_counts, n_done, checkpoint):
    line = "=" * 76
    title = f"{MODEL_SIZE.capitalize()} CNN on the {DEVICE_TYPE.upper()}   (run: {RUN_NAME})"
    if cfg.smoke:
        title += "   SMOKE TEST"
    print(f"\n{line}\n  {title}\n{line}")
    print(f"  Device        {device_text}")
    print(f"  Model         {MODEL_CLASSES[MODEL_SIZE].__name__}: {param_counts[MODEL_SIZE]:,} parameters "
          f"(SmallCNN {param_counts['small']:,} / LargeCNN {param_counts['large']:,})")
    print(f"  Trials        {cfg.n_trials} x {cfg.epochs} epochs ({n_done} already finished)")
    if checkpoint is not None:
        print(f"  Resuming      trial {checkpoint['trial']} from epoch {checkpoint['state']['epoch'] + 1}")
    print(f"  Images        {len(data.y_train)} train / {len(data.y_val)} validation / {len(data.y_test)} test, "
          f"{IMG_SIZE}x{IMG_SIZE} greyscale")
    print(f"  Classes       {', '.join(data.class_names)}")
    print(f"  Results       {cfg.out_dir}")
    print(f"  Electricity   {ELECTRICITY_PRICE} THB/kWh  (placeholder: check your latest bill)")
    print(line)


def run_experiment(cfg, info):
    """Returns "complete" or "stopped"."""
    device_text = setup_device()
    data = prepare_data(cfg)
    plan = build_trial_plan(cfg.n_trials)
    study = open_study(cfg)
    done = completed_trials(study)
    checkpoint = load_checkpoint(cfg, plan, done)
    n_classes = len(data.class_names)
    param_counts = {size: count_parameters(cls(num_classes=n_classes)) for size, cls in MODEL_CLASSES.items()}

    info.update(hardware=hardware_info(), libraries=library_versions(), param_count=param_counts[MODEL_SIZE],
                data={"dataset_folder": str(data.root), "splits_file": str(SPLITS_FILE),
                      "class_names": data.class_names, "n_train": len(data.y_train), "n_val": len(data.y_val),
                      "n_test": len(data.y_test), "norm_mean": data.norm[0], "norm_std": data.norm[1]},
                trial_plan=plan)
    save_run_info(cfg, info)
    finished_here = [i for i in done if i < cfg.n_trials]
    print_startup_summary(cfg, data, device_text, param_counts, len(finished_here), checkpoint)
    if done:
        export_results(cfg, done)  # brings the CSVs up to date after an interrupted run

    epochs_left = cfg.epochs * (cfg.n_trials - len(finished_here))
    if checkpoint is not None:
        epochs_left -= checkpoint["state"]["epoch"]
    progress = Progress(epochs_left)

    trial_bar = tqdm(total=cfg.n_trials, initial=len(finished_here), desc="Trials", unit="trial", dynamic_ncols=True)
    try:
        for i in range(cfg.n_trials):
            if i in done:
                continue
            if STOP_REQUESTED:
                tqdm.write(f"Stopped before trial {i}. Run 'python {SCRIPT}' again to carry on.")
                return "stopped"
            resume_from = None
            if checkpoint is not None and checkpoint["trial"] == i:
                resume_from, checkpoint = checkpoint, None
            outcome = run_trial(cfg, data, i, plan[i], progress, resume_from)
            if outcome is None:
                tqdm.write(f"Stopped. Run 'python {SCRIPT}' again to carry on from the checkpoint.")
                return "stopped"
            result, epoch_log, best_weights = outcome

            best_so_far = max((t.user_attrs["result"]["best_val_acc"] for t in done.values()), default=-1.0)
            if result["best_val_acc"] > best_so_far:
                save_best_model(cfg, data, result, plan[i], best_weights)
                tqdm.write(f"  New best model: trial {i}, epoch {result['best_epoch']}, "
                           f"val acc {result['best_val_acc']:.4f}")
            record_trial(study, result, epoch_log, plan[i])
            done = completed_trials(study)
            cfg.checkpoint_path.unlink(missing_ok=True)
            export_results(cfg, done)
            tqdm.write(f"  -> trial {i} finished: best val acc {result['best_val_acc']:.4f} "
                       f"(epoch {result['best_epoch']}) | {fmt_hms(result['wall_time_sec'])} | "
                       f"{fmt_num(result['energy_kwh'], '.5f')} kWh | {fmt_num(result['cost_thb'], '.3f')} THB")
            trial_bar.update(1)
    finally:
        trial_bar.close()

    if not cfg.model_path.exists():
        raise SetupError(f"The best model file is missing: {cfg.model_path}")
    print("\nAll trials finished. Testing the best model once on the untouched Testing folder ...")
    test_acc, saved = final_test(cfg, data)
    trials_df = export_results(cfg, done, test_acc)
    totals = run_totals(trials_df)
    info["results"] = {
        "trials_finished": len(trials_df), "total_time_sec": json_safe(totals["total_time_sec"]),
        "total_energy_kwh": json_safe(totals["total_energy_kwh"]),
        "total_cost_thb": json_safe(totals["total_cost_thb"]),
        "total_nvsmi_gpu_kwh": json_safe(totals["total_nvsmi_gpu_kwh"]),
        "best_val_acc": saved["val_acc"], "best_trial": saved["trial"], "best_epoch": saved["epoch"],
        "test_acc": test_acc, "model_file": str(cfg.model_path), "resumed_trials": totals["n_resumed_trials"],
    }
    save_run_info(cfg, info)
    print_complete_banner(cfg, trials_df, totals, test_acc, saved)
    if cfg.smoke:
        print(f"This was a smoke test: its files in {cfg.out_dir} are not part of the experiment.")
        print(f"Start the real run with:  python {SCRIPT}\n")
    beep(3)
    notify(f"Training complete: {RUN_NAME}",
           f"Test acc {test_acc:.3f} | {fmt_hms(totals['total_time_sec'])} | "
           f"{fmt_num(totals['total_energy_kwh'], '.3f')} kWh | {fmt_num(totals['total_cost_thb'], '.2f')} THB")
    return "complete"


def make_config(smoke):
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    if smoke:
        out_dir = BASE_DIR / "results" / "smoke" / RUN_NAME
    else:
        out_dir = BASE_DIR / "results" / RUN_NAME
    model_name = f"{MODEL_SIZE.capitalize()} CNN {DEVICE_TYPE.upper()} Trained {n_trials}t_{epochs}E.pt"
    return SimpleNamespace(
        smoke=smoke, n_trials=n_trials, epochs=epochs, out_dir=out_dir,
        codecarbon_dir=out_dir / "codecarbon", model_path=out_dir / "models" / model_name,
        checkpoint_path=out_dir / "checkpoint.pt",
        checkpoint_every=0 if smoke else CHECKPOINT_EVERY_SEC)  # smoke tests exercise the checkpoint code


def main():
    for stream in (sys.stdout, sys.stderr):  # never crash on a character the console cannot show
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=f"Train the {MODEL_SIZE} CNN on the {DEVICE_TYPE.upper()}.")
    parser.add_argument("--smoke", action="store_true",
                        help="quick test: 1 trial, 2 epochs, 1 in 20 images, written to results/smoke/")
    cfg = make_config(parser.parse_args().smoke)

    if cfg.smoke and cfg.out_dir.exists():
        shutil.rmtree(cfg.out_dir)  # smoke tests always start clean
    for folder in (cfg.out_dir, cfg.codecarbon_dir, cfg.model_path.parent):
        folder.mkdir(parents=True, exist_ok=True)

    info = load_run_info(cfg)
    session = {"started": now_iso(), "ended": None, "status": "running", "command": " ".join(sys.argv)}
    info.setdefault("first_started", session["started"])
    info.setdefault("sessions", []).append(session)
    info.update(run_name=RUN_NAME, model=MODEL_SIZE, device=DEVICE_TYPE, smoke=cfg.smoke, settings=settings_dict(cfg))
    save_run_info(cfg, info)

    console_mode = disable_quick_edit() if DISABLE_QUICK_EDIT else None
    keep_pc_awake(True)
    signal.signal(signal.SIGINT, handle_ctrl_c)
    status, exit_code = "crashed", 1
    try:
        status = run_experiment(cfg, info)
        exit_code = 0
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
        print("\nStopped immediately. Work since the last checkpoint is lost; "
              f"run 'python {SCRIPT}' again to carry on.")
    except Exception:
        error_text = traceback.format_exc()
        log_error(cfg, error_text)
        print(f"\n{'!' * 76}\n  TRAINING CRASHED\n{'!' * 76}\n{error_text}")
        print(f"The error was also written to {cfg.out_dir / 'error_log.txt'}")
        print(f"Fix the problem, then run 'python {SCRIPT}' again: finished trials are kept.")
        beep(3, freq=440)
        notify(f"Training crashed: {RUN_NAME}", error_text.strip().splitlines()[-1])
    finally:
        keep_pc_awake(False)
        restore_console_mode(console_mode)
        session.update(ended=now_iso(), status=status)
        info["last_ended"] = session["ended"]
        save_run_info(cfg, info)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
