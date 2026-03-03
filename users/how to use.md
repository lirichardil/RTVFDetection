## Phase 1 — Environment Setup Complete

### Files Created

File

Purpose

[environment.yml](vscode-webview://1qd99teotqvugosnh7jarnr0f4p7ft2v7den3sjh8o6q3s2bno9b/environment.yml)

Conda env spec — primary install path

[setup.py](vscode-webview://1qd99teotqvugosnh7jarnr0f4p7ft2v7den3sjh8o6q3s2bno9b/setup.py)

Installable package definition (`pip install -e .`)

[requirements.txt](vscode-webview://1qd99teotqvugosnh7jarnr0f4p7ft2v7den3sjh8o6q3s2bno9b/requirements.txt)

Pip fallback for non-conda users

[tests/test_phase1_environment.py](vscode-webview://1qd99teotqvugosnh7jarnr0f4p7ft2v7den3sjh8o6q3s2bno9b/tests/test_phase1_environment.py)

6 pytest tests with PASS/FAIL output

[run_tests.sh](vscode-webview://1qd99teotqvugosnh7jarnr0f4p7ft2v7den3sjh8o6q3s2bno9b/run_tests.sh)

Shell script — auto-activates env and runs pytest

---

### How to Use

**Step 1 — Create the environment:**

```bash
conda env create -f environment.yml
```

**Step 2 — Run the tests:**

```bash
./run_tests.sh
```

Or manually:

```bash
conda activate rtvf-detectionpytest tests/test_phase1_environment.py -v -s
```

---

### What the 6 Tests Check

#

Test

What it validates

1

DeepLabCut import + version

`dlc.__version__ == "2.3.6"` (paper-exact)

2

TensorFlow GPU detection

`tf.config.list_physical_devices("GPU")` is non-empty

3

OpenCV import

`import cv2` succeeds

4

EfficientNetB0 ImageNet weights

Loads the exact paper backbone, counts parameters

5

PyTorch CUDA

`torch.cuda.is_available() == True`

6

GPU VRAM >= 6 GB

Queries via `pynvml`, falls back to TensorFlow

Each test prints an inline coloured `[PASS]` / `[FAIL]` line, and a consolidated summary table prints at the end of the run.