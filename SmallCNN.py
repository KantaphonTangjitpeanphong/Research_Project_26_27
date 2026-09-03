"""
CPU vs GPU Training Efficiency Experiment — SMALL CNN ONLY
Brain Tumor MRI Classification
============================================================================
Run this file directly: python run_small.py
Standalone — does not depend on run_medium.py or run_large.py.
 
Recommended order:
  1. Run once with only RUN_SANITY_CHECK = True   -> confirms pipeline works
  2. Run once with only RUN_TIMING_CHECK = True   -> gives real per-epoch times
  3. Update EPOCHS / TRIALS below using those real timings
  4. Run with RUN_FULL_GRID = True                -> the actual experiment
"""
 
import time
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
import kagglehub
 
MODEL_NAME = "small"
 
# ============================================================
# CONFIG — edit before running
# ============================================================
RUN_SANITY_CHECK = True
RUN_TIMING_CHECK = True
RUN_FULL_GRID = False     # long-running — enable deliberately
 
BATCH_SIZE = 128
IMG_SIZE = 64
 
EPOCHS = 1
TRIALS = {"cpu": 40, "cuda": 40}   # Small is cheap on both — 40/40 is fine
 
ELECTRICITY_RATE_THB_PER_KWH = 3.95  # verify current rate at mea.or.th before final analysis
 
# ============================================================
# DEVICE SETUP
# ============================================================
device_gpu = "cuda" if torch.cuda.is_available() else None
device_cpu = "cpu"
 
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if device_gpu:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected — CPU-only combinations will still run.")
print("=" * 60)
 
# ============================================================
# DATASET
# ============================================================
DATA_PATH = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print(f"Dataset downloaded to: {DATA_PATH}")
 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
 
train_data = datasets.ImageFolder(f"{DATA_PATH}/Training", transform=transform)
test_data = datasets.ImageFolder(f"{DATA_PATH}/Testing", transform=transform)
 
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
 
NUM_CLASSES = len(train_data.classes)
print(f"Classes: {train_data.classes}")
print(f"Train images: {len(train_data)}, Test images: {len(test_data)}")
 
# ============================================================
# MODEL — Small only
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
 
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
 
 
# ============================================================
# SHAPE CHECK
# ============================================================
_test_model = SmallCNN()
_X, _y = next(iter(trainloader))
_out = _test_model(_X)
assert _out.shape == (_X.shape[0], NUM_CLASSES), f"Shape mismatch: {_out.shape}"
print(f"Shape check passed: {tuple(_out.shape)}")
del _test_model, _X, _y, _out
 
# ============================================================
# CORE FUNCTIONS
# ============================================================
def evaluate(model, device, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
 
 
def time_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    start = time.time()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return time.time() - start
 
 
def run_trial(device, epochs, trial_num, criterion):
    torch.manual_seed(trial_num)  # same seed for a given trial# across CPU/GPU -> fair pairing
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracker = EmissionsTracker(
        project_name=f"{MODEL_NAME}_{device}_trial{trial_num}",
        save_to_file=True,
        output_file=f"emissions_{MODEL_NAME}.csv",
        log_level="error",
    )
    tracker.start()
 
    log = []
    for epoch in range(epochs):
        tracker.start_task(f"epoch_{epoch}")
        epoch_start = time.time()
        model.train()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        epoch_duration = time.time() - epoch_start
        epoch_emissions = tracker.stop_task()
 
        val_acc = evaluate(model, device, val_loader)
        log.append({
            "model": MODEL_NAME, "device": device, "trial": trial_num, "epoch": epoch,
            "val_acc": val_acc,
            "energy_kwh": epoch_emissions.energy_consumed,
            "duration_sec": epoch_duration,
        })
 
    tracker.stop()
    return log
 
 
# ============================================================
# PHASE 1 — SANITY CHECK
# ============================================================
if RUN_SANITY_CHECK:
    print("\n" + "=" * 60)
    print(f"PHASE 1: SANITY CHECK — {MODEL_NAME} model, 20 epochs")
    print("=" * 60)
    check_device = device_gpu or device_cpu
    criterion = nn.CrossEntropyLoss()
    sanity_log = run_trial(check_device, epochs=20, trial_num=0, criterion=criterion)
    for row in sanity_log:
        print(f"  epoch {row['epoch']:>3}  val_acc={row['val_acc']:.3f}  energy={row['energy_kwh']:.6f} kWh")
    final_acc = sanity_log[-1]["val_acc"]
    print(f"\nFinal validation accuracy: {final_acc:.1%}")
    if final_acc < 0.35:
        print("WARNING: accuracy is near chance level (~25%). Check data/labels/learning rate before proceeding.")
    else:
        print("Sanity check passed — pipeline appears to be learning correctly.")
 
# ============================================================
# PHASE 2 — TIMING CHECK
# ============================================================
if RUN_TIMING_CHECK:
    print("\n" + "=" * 60)
    print(f"PHASE 2: TIMING CHECK — {MODEL_NAME} model, 1 epoch per device")
    print("=" * 60)
    criterion = nn.CrossEntropyLoss()
    devices_to_test = [d for d in [device_cpu, device_gpu] if d is not None]
    for dev in devices_to_test:
        model = SmallCNN().to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sec = time_one_epoch(model, dev, trainloader, optimizer, criterion)
        planned = EPOCHS * TRIALS[dev]
        est_hours = sec * planned / 3600
        print(f"  {dev:>4}: {sec:6.2f} sec/epoch  ->  {planned} epoch-units planned  ->  ~{est_hours:.2f} hours total")
    print("\nUse these numbers to adjust EPOCHS / TRIALS above before running the full grid.")
 
# ============================================================
# PHASE 3 — FULL GRID (this model only)
# ============================================================
if RUN_FULL_GRID:
    print("\n" + "=" * 60)
    print(f"PHASE 3: FULL GRID — {MODEL_NAME} model, CPU and GPU")
    print("=" * 60)
    criterion = nn.CrossEntropyLoss()
    all_logs = []
    for dev in [device_cpu, device_gpu]:
        if dev is None:
            continue
        n_trials = TRIALS[dev]
        print(f"\n--- {MODEL_NAME} on {dev}: {n_trials} trials x {EPOCHS} epochs ---")
        for trial in range(n_trials):
            t0 = time.time()
            trial_log = run_trial(dev, EPOCHS, trial, criterion)
            all_logs.extend(trial_log)
            print(f"  trial {trial + 1}/{n_trials} done in {time.time() - t0:.1f}s "
                  f"(final val_acc={trial_log[-1]['val_acc']:.3f})")
 
    results_df = pd.DataFrame(all_logs)
    results_df.to_csv(f"full_results_{MODEL_NAME}.csv", index=False)
    print(f"\nSaved {len(results_df)} epoch-level rows to full_results_{MODEL_NAME}.csv")
 
    trial_totals = results_df.groupby(["model", "device", "trial"]).agg(
        total_energy_kwh=("energy_kwh", "sum"),
        total_time_sec=("duration_sec", "sum"),
        final_val_acc=("val_acc", "last"),
    ).reset_index()
 
    summary = trial_totals.groupby(["model", "device"]).agg(
        mean_energy_kwh=("total_energy_kwh", "mean"),
        std_energy_kwh=("total_energy_kwh", "std"),
        mean_time_sec=("total_time_sec", "mean"),
        mean_final_val_acc=("final_val_acc", "mean"),
        n_trials=("trial", "count"),
    ).reset_index()
    summary["est_cost_thb"] = summary["mean_energy_kwh"] * ELECTRICITY_RATE_THB_PER_KWH
 
    print("\nSummary (mean +/- std per trial, per device):")
    print(summary.to_string(index=False))
    summary.to_csv(f"summary_results_{MODEL_NAME}.csv", index=False)
    print(f"\nSaved summary_results_{MODEL_NAME}.csv")
 
print("\nDone.")