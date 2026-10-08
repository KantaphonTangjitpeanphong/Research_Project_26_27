# Claude Code Prompt — CPU vs GPU Training Scripts

Oct 8, 2026 · @Kasira Krisrayaphach

## How to use this doc

Copy the whole prompt in the next section into Claude Code, opened in your project folder on the Windows PC. It will build four standalone scripts that train start to finish with no manual "go" steps, announce when they finish, and save the trained model.

1. Open a terminal in the project folder and activate your virtual environment.
2. Start Claude Code and paste the master prompt.
3. Let it build and self-test (it runs a 1-trial, 2-epoch smoke test of each script).
4. Run the real experiments one at a time, in this order: `run_small_gpu.py` → `run_large_gpu.py` → `run_small_cpu.py` → `run_large_cpu.py` (shortest first, so problems show up early).
5. Close other heavy apps and keep the PC plugged in while each one runs.

Your list had "small model GPU" twice; this doc assumes the fourth script is the **large** model on GPU, so the four cells are Small/Large × CPU/GPU.

## Master prompt (paste into Claude Code)

```text
You are helping me with an A-Level Computer Science research project.

RESEARCH QUESTION
"When training an AI CNN model, is a GPU or a CPU better value for money once
accounting for time, accuracy, electricity, and hardware, and does the answer
change as the model gets bigger?"

TASK
Create FOUR standalone, fully automatic training scripts in this folder:
  1. run_small_cpu.py  - Small CNN on CPU
  2. run_large_cpu.py  - Large CNN on CPU
  3. run_small_gpu.py  - Small CNN on GPU
  4. run_large_gpu.py  - Large CNN on GPU
plus one shared module, common.py, that holds everything the four have in common
(data loading, models, training loop, logging, saving). Each run_*.py should be
short: just its settings + a call into common.py. Before writing anything, look
at the existing files in this folder (older run_*.py / train_*.py scripts,
analyze_results.py, experiment-plan-final.md) and reuse their model definitions,
file names and CSV columns where they exist, so the new results stay compatible.
Tell me about anything you change.

MY SETUP
- Windows 11 desktop: Ryzen 5 7600 (6 cores / 12 threads), RTX 4070 Super 12GB,
  32GB RAM. Python virtual environment already exists.
- PyTorch cu128 build (CUDA 13.1 driver). Do NOT install or suggest cu130.
- Libraries: torch, torchvision, optuna, codecarbon, pandas, numpy, scikit-learn,
  pillow, tqdm. Install anything missing into the venv only.

DATASET
- Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar): 4 classes (glioma,
  meningioma, notumor, pituitary), ~7,000 images, Training/ and Testing/ folders.
- Ask me for the dataset path once and store it in a DATA_DIR setting at the
  top of common.py.
- Preprocess: grayscale, resize to 64x64, normalise. Light augmentation on the
  training set only (random horizontal flip, small rotation).
- Split the official Training folder into train/validation (85/15, stratified,
  seed 42). Save the split to splits.json on first run and REUSE it, so all four
  scripts train and validate on exactly the same images. The Testing folder is
  used only once, for the final test of the best model.
- To keep CPU vs GPU fair, preload the resized images into memory once
  (they are small) so disk speed does not affect timing.

MODELS (both use global average pooling, no big fully-connected layers)
- SmallCNN: 3 conv blocks (16, 32, 64 channels), each Conv3x3-BatchNorm-ReLU-
  MaxPool, then GAP -> Dropout -> Linear(4).
- LargeCNN: 5 conv blocks (32, 64, 128, 256, 512 channels), each block has two
  Conv3x3-BatchNorm-ReLU layers then MaxPool, then GAP -> Dropout -> Linear(4).
- Print the parameter count of each model at startup and save it in the results.
- If the existing scripts already define these models differently, keep the
  existing definitions and tell me.

EXPERIMENT DESIGN (settings at the top of each run_*.py)
- N_TRIALS: 50 for small_cpu, small_gpu, large_gpu; 10 for large_cpu
  (feasibility limit, the CPU large run is very slow).
- EPOCHS per trial: 100 for all four.
- Optuna tunes: learning rate (1e-4 to 1e-2, log), batch size (32, 64, 128),
  weight decay (1e-6 to 1e-3, log), dropout (0.1 to 0.5).
- Use optuna.samplers.RandomSampler(seed=42) so trial N gets the SAME
  hyperparameters on CPU and GPU. (A TPE sampler adapts to results, so CPU and
  GPU would test different settings and the comparison would not be fair.)
  The 10 large_cpu trials must therefore match the first 10 large_gpu trials.
- Optimiser: AdamW. Loss: CrossEntropy. Objective: best validation accuracy.
- Fixed seeds (python, numpy, torch) per trial = 42 + trial_number.
- No early stopping and no pruning: every trial runs all epochs, so time and
  energy are comparable.

DEVICE RULES
- GPU scripts: require CUDA. If torch.cuda.is_available() is False, stop
  immediately with a clear message (likely wrong PyTorch build or wrong Python).
  Print the GPU name and CUDA version at startup. Use pin_memory and
  non_blocking transfers. Do not use mixed precision (keep FP32 on both devices
  so accuracy is comparable).
- CPU scripts: set os.environ["CUDA_VISIBLE_DEVICES"] = "" before importing
  torch, force device = cpu, and torch.set_num_threads(6). Print the thread count.
- Both: num_workers=0 for the DataLoader (data is already in memory; avoids
  Windows multiprocessing problems). Wrap the entry point in
  if __name__ == "__main__":.

MEASUREMENTS (per trial AND per epoch)
- Wall-clock training time (time.perf_counter). For GPU, call
  torch.cuda.synchronize() before reading the clock.
- Energy with CodeCarbon: one EmissionsTracker per trial, measure_power_secs=5,
  log_level="error", output saved in results/<run_name>/codecarbon/. Record
  energy_consumed (kWh), cpu/gpu/ram energy split, and emissions.
- GPU scripts only: a background thread that samples nvidia-smi power.draw every
  5 seconds and integrates it to kWh as a cross-check column.
- Electricity cost = kWh x ELECTRICITY_PRICE (setting, default 4.18 THB/kWh;
  tell me to check my bill).
- Train loss, train accuracy, validation loss, validation accuracy each epoch.
- Peak RAM (psutil) and, for GPU, peak VRAM (torch.cuda.max_memory_allocated).

SAFETY / RESUME (long runs, power cuts are possible)
- Save each finished trial to the CSVs immediately.
- Use an Optuna SQLite storage (results/<run_name>/optuna.db, load_if_exists)
  so re-running the same script continues from the next unfinished trial.
- Save a checkpoint (model + optimizer + scheduler state + epoch + RNG states)
  at the end of every epoch to results/<run_name>/checkpoint.pt. If the script
  is restarted mid-trial, resume that trial from the last epoch and mark it
  resumed=True in the CSV (its time/energy will be split across sessions).
- Prevent Windows from sleeping while training (ctypes
  SetThreadExecutionState with ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and restore
  normal behaviour on exit.
- Ctrl+C should save a checkpoint and exit cleanly.

PROGRESS DISPLAY
- At startup print a summary: model, device, trials, epochs, parameter count.
- After the first epoch of trial 1, print an estimated total runtime
  (hours) and estimated finish time. Do NOT wait for input; keep going.
- One tqdm bar for trials and a one-line update per epoch
  (epoch, loss, val acc, epoch time, elapsed, ETA).

WHEN FINISHED
- Retrain is NOT needed: during the study keep the weights of the best
  trial (highest validation accuracy) as they are produced.
- Evaluate that best model once on the untouched Testing folder: test
  accuracy, per-class precision/recall/F1, and confusion matrix.
- Save outputs to results/<run_name>/ (run_name = small_cpu, large_cpu,
  small_gpu or large_gpu):
    models/<Size> CNN <DEVICE> Trained <N>t_100E.pt  (state_dict + model
        class name + hyperparameters + class names, loadable by test_model.py)
    summary_results_<run_name>.csv  (one row: means and totals over all trials)
    full_results_<run_name>.csv     (one row per epoch per trial)
    trials_<run_name>.csv           (one row per trial: hyperparameters, time,
                                     kWh, cost, best val acc, resumed flag)
    test_report_<run_name>.txt      (test accuracy, per-class metrics,
                                     confusion matrix)
    run_info.json   (hardware, library versions, settings, start/end time)
    optuna.db, checkpoint.pt, codecarbon/
- Print a big, clear "TRAINING COMPLETE" banner with: total time, total kWh,
  total cost (THB), best validation accuracy, test accuracy, and the path of
  the saved model.
- Play a sound (winsound.Beep, 3 beeps) and show a Windows toast notification
  if the plyer package is available (fall back silently if not).
- If the script crashes, also beep and print the error, and write it to
  results/<run_name>/error_log.txt.

ALSO CREATE
- test_model.py: loads any saved .pt model, opens a file picker, and prints the
  predicted class + confidence for a chosen MRI image.
- README_RUNS.md: how to run each script and what each output file contains.

HOW TO WORK
1. First read the existing files and show me a short plan. Flag any problem or
   limitation you see BEFORE writing code (I prefer to know early).
2. Write the code.
3. Run python -m py_compile on every file after each batch of edits.
4. Run a smoke test of each script with --smoke (1 trial, 2 epochs, 5% of data,
   writes to results/smoke/ so it never mixes with real results) and show me
   the output. Add the --smoke flag to all four scripts.
5. Do NOT start the full runs yourself. Tell me the exact commands to run.
```

## The four scripts at a glance

All four share the same data split, seeds, hyperparameter trials and epoch count; only the model and the device change.

| Script | Model | Device | Trials × epochs | Saved model name |
| --- | --- | --- | --- | --- |
| run\_small\_cpu.py | SmallCNN (3 blocks) | CPU, 6 threads | 50 × 100 | Small CNN CPU Trained 50t\_100E.pt |
| run\_large\_cpu.py | LargeCNN (5 blocks) | CPU, 6 threads | 10 × 100 | Large CNN CPU Trained 10t\_100E.pt |
| run\_small\_gpu.py | SmallCNN (3 blocks) | RTX 4070 Super | 50 × 100 | Small CNN GPU Trained 50t\_100E.pt |
| run\_large\_gpu.py | LargeCNN (5 blocks) | RTX 4070 Super | 50 × 100 | Large CNN GPU Trained 50t\_100E.pt |

Large CPU stays at 10 trials as a feasibility limit. Because the sampler is seeded, those 10 trials use the same hyperparameters as the first 10 Large GPU trials, so you can still compare them like-for-like.

## What you get when a run finishes

Each script writes into its own folder, so the four runs never overwrite each other. The same list is inside the prompt, so Claude Code builds exactly this.

```text
results/
  small_gpu/                     (same layout for small_cpu, large_cpu, large_gpu)
    models/Small CNN GPU Trained 50t_100E.pt   <- the trained model
    summary_results_small_gpu.csv              <- one row for your results table
    full_results_small_gpu.csv                 <- per-epoch curves for graphs
    trials_small_gpu.csv                       <- one row per trial
    test_report_small_gpu.txt                  <- final test accuracy + confusion matrix
    run_info.json                              <- hardware, versions, settings
    optuna.db, checkpoint.pt                   <- used for resuming
    codecarbon/                                <- raw energy logs
  smoke/                                       <- quick tests only, ignore for results
```

The summary and full-results CSV names match your earlier scripts, so analyze\_results.py should read them with little or no change.

## Before you start the real runs

- [ ] All four smoke tests passed and Claude Code showed you their output.
- [ ] The GPU scripts printed "RTX 4070 Super" at startup; the CPU scripts printed 6 threads.
- [ ] splits.json exists and was created only once (all four runs reuse it).
- [ ] ELECTRICITY\_PRICE matches your latest electricity bill (4.18 THB/kWh is only a placeholder).
- [ ] Note the wall power meter reading before and after each run, for your cross-check against CodeCarbon.
- [ ] Check the estimated runtime printed after the first epoch. Large CPU is likely the longest by far; if it looks like days, plan when the PC can stay on.
- [ ] Close games, browsers and background updates so they don't distort time and energy.
- [ ] If a run stops (crash, power cut, Ctrl+C), just run the same command again: it resumes from the last saved epoch.

Two limitations to mention in your write-up: on an AMD CPU under Windows, CodeCarbon usually estimates CPU power from the chip's rated wattage rather than measuring it, so the wall meter is your more reliable CPU energy figure. And any trial marked resumed=True has its time and energy split across sessions, so flag it in your analysis.
