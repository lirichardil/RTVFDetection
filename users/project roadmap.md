# 🔬 Project Roadmap: Replication of Real-Time Vocal Fold Tracking

### Based on: *"Feasibility of Real-Time Automated Vocal Fold Motion Tracking for In-Office Laryngoscopy"* — Koivu et al., Laryngoscope 2026

---

> **How to use this document:**

> Each phase contains a description, test cases, and a ready-to-paste prompt for Claude Code (`claude` CLI).

> Work through phases sequentially. Each phase depends on the previous one passing its test cases.

---

## 📋 Project Overview

| Item | Detail |

|------|--------|

| **Goal** | Replicate real-time laryngeal keypoint tracking at 30fps |

| **Model** | DeepLabCut 2.3.6 + EfficientNet-B0 |

| **Keypoints** | 39 anatomical landmarks |

| **Target mKA** | ≥80% on validation set |

| **Target FPS** | ≥25 fps on laptop GPU |

| **Timeline** | ~10–14 weeks |

| **Language** | Python 3.9+ |

---

## 🗂️ Master Phase Summary

| Phase | Name | Duration | Key Output |

|-------|------|----------|------------|

| 1 | Environment Setup | Week 1 | Working conda env + verified installs |

| 2 | Data Collection | Week 2–3 | Raw video dataset assembled |

| 3 | Frame Extraction | Week 3 | PNG frames extracted and organized |

| 4 | Keypoint Annotation | Week 4–5 | 1,400+ annotated frames in DLC format |

| 5 | Model Training | Week 6–8 | Trained model checkpoints |

| 6 | Checkpoint Selection | Week 8 | Best checkpoint identified |

| 7 | Evaluation Metrics | Week 9 | mKA, Precision, Recall, mTC scores |

| 8 | Real-Time Pipeline | Week 10–11 | Live inference at 30fps |

| 9 | Validation & Testing | Week 12 | Full results table vs paper |

| 10 | Documentation | Week 13–14 | Final report and code cleanup |

---

## PHASE 1: Environment Setup

### 🎯 Objective

Create a fully reproducible Python environment with all dependencies matching the paper's stack exactly.

### 📁 Expected Output

```
/project/├── environment.yml├── requirements.txt├── setup.py└── tests/    └── test_phase1_environment.py
```

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T1.1 | Import deeplabcut | No ImportError | `import deeplabcut` succeeds |

| T1.2 | DLC version check | Exactly 2.3.6 | `deeplabcut.__version__ == "2.3.6"` |

| T1.3 | TensorFlow GPU | GPU detected | `tf.config.list_physical_devices('GPU')` returns ≥1 device |

| T1.4 | OpenCV capture | Camera/file opens | `cv2.VideoCapture(0)` or test file opens |

| T1.5 | EfficientNet-B0 load | Model loads | Keras EfficientNetB0 loads with ImageNet weights |

| T1.6 | CUDA availability | CUDA available | `torch.cuda.is_available()` returns True |

| T1.7 | Memory check | Sufficient VRAM | GPU VRAM ≥ 6GB detected |

### 💬 Claude Code Prompt — Phase 1

```
I am replicating a medical computer vision paper on real-time vocal fold tracking.Help me set up the complete project environment.  Tasks:1. Create a conda environment file (environment.yml) with:   - Python 3.9   - deeplabcut==2.3.6   - tensorflow-gpu==2.10   - opencv-python   - numpy, pandas, matplotlib, scikit-learn, scipy   - torch==2.0 (for optional inference optimization)   - Pillow, tqdm, PyYAML  2. Create a setup.py for the project package  3. Create a requirements.txt as fallback for pip users  4. Create tests/test_phase1_environment.py with pytest test cases that verify:   - deeplabcut imports and version is 2.3.6   - TensorFlow detects GPU   - OpenCV is importable   - EfficientNetB0 loads with ImageNet weights via keras   - torch.cuda.is_available() returns True   - System has at least 6GB GPU VRAM  5. Create a run_tests.sh script that activates the conda env and runs pytest  6. Print a clear PASS/FAIL summary for each test  Make all paths relative. Use /project as root directory.
```

---

## PHASE 2: Data Collection

### 🎯 Objective

Assemble a publicly-sourced video dataset of laryngoscopy footage that approximates the paper's Table 1 composition. **MEE clinical data and NCT05216770 LD test data are not publicly available**, so all videos must be sourced from YouTube (fair use) or other open repositories. Target video counts match the paper (57 train/val + 10 held-out test = 67 total).

### 📁 Expected Output

```
/project/data/├── raw_videos/│   └── youtube/        # all 67 videos downloaded here├── video_manifest.csv└── tests/    └── test_phase2_data.py
```

### Video Target Composition (Table 1 Replication)

| Source | Count | Resolution | FPS | Avg Duration |

|--------|-------|------------|-----|--------------|

| YouTube train/val (fair use) | 57 | 360p–720p variable | 30 | ~120–240s |

| YouTube held-out test | 10 | 360p–720p variable | 30 | ~120–240s |

### Condition Distribution (Training Set)

| Condition | Target Count | Notes |

|-----------|-------------|-------|

| Normal / healthy vocal folds | ~25 | search: "normal laryngoscopy vocal folds" |

| Vocal fold immobility (VFI) | ~20 | search: "vocal fold paralysis laryngoscopy" |

| Other pathologies (mixed) | ~12 | search: "dysphonia laryngoscopy", "MTD" |

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T2.1 | Manifest file exists | CSV created | `video_manifest.csv` exists with correct columns |

| T2.2 | Minimum video count | ≥20 videos | At least 20 videos in raw_videos/ |

| T2.3 | Video readability | All videos open | 100% of listed videos open with cv2 |

| T2.4 | FPS check | All 30fps | All videos report 29–31 fps |

| T2.5 | Duration check | Reasonable lengths | All videos ≥5s duration |

| T2.6 | Resolution variety | Multiple resolutions | At least 2 distinct resolutions present |

| T2.7 | Manifest completeness | All fields filled | No null values in required columns |

### 💬 Claude Code Prompt — Phase 2

```
I am building a dataset pipeline for vocal fold tracking replication.  Tasks:1. Create src/data_collection/video_validator.py that:   - Scans /project/data/raw_videos/ recursively for .mp4, .avi, .mov files   - For each video: extracts fps, resolution, duration, frame count   - Writes results to /project/data/video_manifest.csv with columns:     [filename, path, source, fps, width, height, duration_s, frame_count, condition, split]   - Flags any video that fails to open  2. Create src/data_collection/youtube_downloader.py that:   - Takes a list of YouTube URLs as input   - Downloads videos using yt-dlp at best available quality ≤1280p   - Saves to /project/data/raw_videos/youtube/   - Logs each download with success/failure status   - Include these search-ready terms as comments:     "flexible laryngoscopy normal", "videolaryngoscopy tutorial",     "laryngoscopy vocal folds", "laryngoscopy examination demonstration"  3. Create src/data_collection/assign_splits.py that:   - Reads video_manifest.csv   - Assigns 90% to training, 10% to validation (stratified by condition)   - Marks independent test videos separately   - Updates the 'split' column in manifest  4. Create tests/test_phase2_data.py with pytest tests for all T2.x test cases above  5. Create a data_collection_report.py that prints a summary table of the dataset  Use pandas, cv2, pathlib. Handle all errors gracefully.
```

---

## PHASE 3: Frame Extraction

### 🎯 Objective

Extract representative keyframes from all videos, replicating the paper's approach of selecting frames that capture essential laryngeal motions.

### 📁 Expected Output

```
/project/data/├── extracted_frames/│   ├── training/          # ~1,254 PNG frames│   ├── validation/        # ~140 PNG frames│   └── test/              # ~50 PNG frames├── frame_manifest.csv└── tests/    └── test_phase3_frames.py
```

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T3.1 | Frame count — training | ~1,254 frames | Between 1,100 and 1,400 PNGs in training/ |

| T3.2 | Frame count — validation | ~140 frames | Between 120 and 160 PNGs in validation/ |

| T3.3 | Frame count — test | ~50 frames | Between 40 and 60 PNGs in test/ |

| T3.4 | Image readability | All PNGs valid | 100% of frames open with cv2.imread |

| T3.5 | No blank frames | Frames have content | Mean pixel value >10 for all frames |

| T3.6 | Frame uniqueness | No duplicates | All frame hashes are unique |

| T3.7 | Naming convention | Consistent naming | All files match `{video_id}_f{frame_idx}.png` |

| T3.8 | Manifest accuracy | Manifest matches disk | Row count in frame_manifest == actual file count |

### 💬 Claude Code Prompt — Phase 3

```
I need to extract representative keyframes from laryngoscopy videos for annotation.  Tasks:1. Create src/frame_extraction/extractor.py with class FrameExtractor that:   - Takes video path, output directory, target_frames_per_video (default 24)   - Uses uniform temporal sampling across video duration   - Skips first 5% and last 5% of video (often camera setup/teardown)   - Saves as PNG with naming: {video_stem}_f{frame_index:06d}.png   - Computes and logs mean pixel value to detect blank frames (skip if <10)   - Handles exceptions per-video without crashing the pipeline  2. Create src/frame_extraction/run_extraction.py that:   - Reads /project/data/video_manifest.csv   - Runs FrameExtractor for each video   - Routes output to training/, validation/, or test/ based on split column   - Creates /project/data/frame_manifest.csv with columns:     [frame_path, video_source, split, width, height, mean_pixel_value, frame_hash]   - Prints progress bar using tqdm  3. Create src/frame_extraction/quality_filter.py that:   - Loads all extracted frames   - Flags frames with mean_pixel < 10 (too dark)   - Flags frames with mean_pixel > 245 (overexposed)   - Flags duplicate frames via MD5 hash   - Outputs a quality_report.csv and removes flagged frames  4. Create tests/test_phase3_frames.py with pytest tests for all T3.x test cases  5. Add a visualization tool src/frame_extraction/visualize_sample.py that:   - Displays a random 4x4 grid of extracted frames using matplotlib   - Saves as /project/data/frame_sample_grid.png  Use cv2, pathlib, hashlib, tqdm, pandas, matplotlib.
```

---

## PHASE 4: Keypoint Annotation

### 🎯 Objective

Annotate all extracted frames with 39 anatomical keypoints using DeepLabCut's annotation interface, following the paper's keypoint schema exactly.

### 📁 Expected Output

```
/project/data/dlc_project/├── config.yaml├── labeled-data/│   └── {video_name}/│       ├── CollectedData_{annotator}.csv│       └── CollectedData_{annotator}.h5└── videos/
```

### 39-Keypoint Schema (Paper Figure 1A)

| Group | Keypoints | IDs |

|-------|-----------|-----|

| Left true VF + anterior commissure | LC1–LC6, AC | 1–7 |

| Right true VF | RC1–RC6 | 8–13 |

| Midline supraglottis | PET | 14 |

| False left VF | LFC1–LFC6 | 15–20 |

| False right VF | RFC1–RFC6 | 21–26 |

| Midline points | ML1–ML3 | 27–29 |

| Left aryepiglottic fold | LAE1–LAE2 | 30–31 |

| Left cuneiform/corniculate | LCF1, LMD, LCN1 | 32–34 |

| Right aryepiglottic fold | RAE1–RAE2 | 35–36 |

| Right cuneiform/corniculate | RCF1, RMD, RCN1 | 37–39 |

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T4.1 | DLC project created | config.yaml exists | Project initializes without error |

| T4.2 | Keypoint count | Exactly 39 | Config lists 39 bodyparts |

| T4.3 | Keypoint names | Match schema | All 39 names match paper exactly |

| T4.4 | Annotation file format | Valid H5/CSV | pandas can read CollectedData files |

| T4.5 | Annotation coverage | Sufficient frames | ≥1,000 frames annotated |

| T4.6 | No out-of-bounds | Valid coordinates | All x,y coords within image dimensions |

| T4.7 | Annotation completeness | All keypoints per frame | Each frame has all 39 points or marked occluded |

| T4.8 | Train/val split ratio | 90/10 split | 90% training, 10% validation |

### 💬 Claude Code Prompt — Phase 4

```
I need to set up a DeepLabCut project for annotating 39 laryngeal keypoints.  Tasks:1. Create src/annotation/setup_dlc_project.py that:   - Creates a DeepLabCut project at /project/data/dlc_project/   - Sets exactly these 39 bodyparts in order:     LC1, LC2, LC3, LC4, LC5, LC6, AC,     RC1, RC2, RC3, RC4, RC5, RC6,     PET,     LFC1, LFC2, LFC3, LFC4, LFC5, LFC6,     RFC1, RFC2, RFC3, RFC4, RFC5, RFC6,     ML1, ML2, ML3,     LAE1, LAE2,     LCF1, LMD, LCN1,     RAE1, RAE2,     RCF1, RMD, RCN1   - Sets skeleton connections between adjacent keypoints in same structure   - Adds extracted training frames to the project   - Outputs the config.yaml path  2. Create src/annotation/annotation_guide.py that:   - Generates a visual annotation reference image showing:     - Diagram of larynx with all 39 numbered keypoints     - Written description of each keypoint location     - Annotation rules (equidistant spacing, perpendicular for false cords)   - Save to /project/docs/annotation_guide.png  3. Create src/annotation/validate_annotations.py that:   - Reads all CollectedData CSV/H5 files from dlc_project   - Checks: 39 keypoints per frame, coordinates within image bounds   - Checks: no NaN except for legitimately occluded points   - Generates validation_report.csv with per-frame quality scores   - Prints summary statistics  4. Create src/annotation/split_dataset.py that:   - Reads annotated frame list   - Creates 90/10 train/val split stratified by video source   - Creates mirrored copies of training dataset (paper specifies this)   - Outputs split_manifest.csv  5. Create tests/test_phase4_annotations.py with pytest tests for all T4.x test cases  6. Create src/annotation/launch_annotator.py that:   - Launches the DeepLabCut GUI for manual annotation   - Prints clear instructions before launching  Use deeplabcut, pandas, numpy, matplotlib, pathlib.
```

---

## PHASE 5: Model Training

### 🎯 Objective

Train the EfficientNet-B0 based keypoint detection model for 5,000,000 iterations, saving checkpoints every 50,000 iterations, matching Table 2 parameters exactly.

### 📁 Expected Output

```
/project/models/├── dlc_project/│   └── dlc-models/iteration-0/│       └── VF_Tracking{date}/train/│           ├── snapshot-50000.*│           ├── snapshot-100000.*│           └── snapshot-5000000.*├── training_logs/│   ├── loss_curve.csv│   └── loss_curve.png└── tests/    └── test_phase5_training.py
```

### Training Hyperparameters (Table 2 — Exact)

| Parameter | Value |

|-----------|-------|

| Max iterations | 5,000,000 |

| Save every | 50,000 |

| Batch size | 8 |

| Optimizer | Adam |

| Learning rate | 0.0005 |

| Cosine decay steps | 30,000 |

| Cosine decay αR | 0.02 |

| Cosine decay rate | 0.0001 |

| Stride | 8 |

| Locref loss weight | 0.05 |

| Locref SD | 7.2801 |

| Global scale | 0.8 |

| Rotation degrees | 180° |

| Rotation probability | 0.4 |

| Crop probability | 0.4 |

| CLAHE probability | 0.1 |

| Histogram eq probability | 0.1 |

| Jitter bounds | (0.5, 1.25) |

| Jitter probability | 0.4 |

| Emboss probability | 0.1 |

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T5.1 | Training dataset created | DLC dataset exists | `create_training_dataset` completes |

| T5.2 | Config parameters | Match Table 2 | All hyperparams match paper values exactly |

| T5.3 | Training starts | No crash at iter 1 | First 1,000 iterations complete |

| T5.4 | Checkpoint saving | Files created | snapshot-50000 files exist after 50k iters |

| T5.5 | Loss decreasing | Learning occurring | Loss at iter 100k < loss at iter 1k |

| T5.6 | Augmentation active | Augmentation applied | Augmented samples differ from originals |

| T5.7 | Mirror dataset | Mirrored data exists | Mirror copies in training dataset |

| T5.8 | GPU utilization | Training on GPU | nvidia-smi shows >50% GPU util during training |

### 💬 Claude Code Prompt — Phase 5

```
I need to configure and launch DeepLabCut model training for vocal fold keypoint detection.  IMPORTANT: Do NOT download raw COCO images or re-run COCO pre-training.Instead, download the pre-trained EfficientNet-B0 ImageNet weights file(yolov8n.pt or equivalent ~6MB file) from Ultralytics/keras directly.  Tasks:1. Create src/training/configure_training.py that:   - Reads the DLC config at /project/data/dlc_project/config.yaml   - Updates the pose_cfg.yaml with ALL parameters from Table 2:     net_type: efficientnet-b0     init_weights: imagenet     batch_size: 8     optimizer: adam     lr: 0.0005     cosine_decay_steps: 30000     cosine_decay_alpha_r: 0.02     cosine_decay_rate: 0.0001     stride: 8     locref_loss_weight: 0.05     locref_stdev: 7.2801     global_scale: 0.8     All augmentation parameters as in Table 2   - Validates that all parameters were written correctly   - Prints a diff of before/after config  2. Create src/training/download_pretrained_weights.py that:   - Downloads EfficientNet-B0 ImageNet pretrained weights (~6MB)   - Uses keras.applications.EfficientNetB0(weights='imagenet')     or downloads from a stable URL   - Saves to /project/models/pretrained/efficientnet_b0_imagenet.h5   - Verifies download integrity with MD5 check  3. Create src/training/create_training_dataset.py that:   - Calls deeplabcut.create_training_dataset() with augmenter_type="imgaug"   - Creates mirrored copies of all training data   - Points to downloaded pretrained weights   - Verifies dataset creation was successful   - Reports total sample count  4. Create src/training/train.py that:   - Calls deeplabcut.train_network() with:     maxiters=5000000, saveiters=50000     max_snapshots_to_keep=100   - Supports --resume flag to continue from last checkpoint   - Logs training progress to /project/models/training_logs/loss_curve.csv  5. Create src/training/monitor_training.py that:   - Reads loss_curve.csv in real-time   - Plots loss curve with matplotlib (updates every 60 seconds)   - Flags if loss stops decreasing for >500k iterations   - Shows estimated time to completion  6. Create tests/test_phase5_training.py with pytest tests for T5.x cases  Use deeplabcut, yaml, pandas, matplotlib, keras, pathlib, argparse.
```

---

## PHASE 6: Checkpoint Selection

### 🎯 Objective

Identify the optimal checkpoint using the paper's dual-criteria method: lowest validation RMSE AND best temporal consistency (mTC). Paper found checkpoint 4,500,000 optimal.

### 📁 Expected Output

```
/project/models/├── checkpoint_evaluation/│   ├── rmse_per_checkpoint.csv│   ├── mtc_per_checkpoint.csv│   ├── combined_scores.csv│   ├── checkpoint_selection_plot.png│   └── selected_checkpoint.txt└── tests/    └── test_phase6_checkpoint.py
```

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T6.1 | Consistency video created | Video exists | 1920-frame sliding video generated |

| T6.2 | RMSE evaluation runs | CSV produced | rmse_per_checkpoint.csv has ≥10 rows |

| T6.3 | mTC evaluation runs | CSV produced | mtc_per_checkpoint.csv has ≥10 rows |

| T6.4 | RMSE trend | Decreasing then rising | RMSE plot shows U-shape (overfitting) |

| T6.5 | mTC trend | Low then rising | mTC rises after optimal checkpoint |

| T6.6 | Selected checkpoint | Reasonable range | Selected between 3M and 5M iterations |

| T6.7 | Paper alignment | Near 4.5M | Optimal checkpoint within 500k of 4,500,000 |

### 💬 Claude Code Prompt — Phase 6

```
I need to evaluate all saved training checkpoints to select the optimal one.  Tasks:1. Create src/evaluation/create_consistency_video.py that:   - Takes a representative laryngoscopy frame as input   - Creates a 1920-frame video where the frame slides 1 pixel right per frame   - Ensures all 39 keypoints are visible in the source frame   - Saves to /project/data/consistency_test_video.mp4   - Replicates the paper's mTC evaluation method exactly  2. Create src/evaluation/evaluate_checkpoints.py with class CheckpointEvaluator that:   - Iterates over all snapshots in /project/models/dlc_project/   - For each checkpoint:     a) Computes validation RMSE using deeplabcut.evaluate_network()     b) Runs inference on consistency video and computes mTC:        mTC = (1/N) * sum of Euclidean distances between consecutive frames        where N = valid transitions above prob_threshold=0.5   - Saves results to checkpoint_evaluation/ CSV files  3. Create src/evaluation/select_checkpoint.py that:   - Reads both CSV files   - Normalizes RMSE and mTC to [0,1] scale   - Identifies checkpoint minimizing combined score   - Creates visualization matching paper Figure 2B style   - Writes selected checkpoint to selected_checkpoint.txt  4. Create src/evaluation/plot_checkpoint_curves.py that:   - Plots Figure 2A style: training loss over iterations   - Plots Figure 2B style: mTC over checkpoints   - Marks selected checkpoint in green   - Saves to /project/models/checkpoint_evaluation/  5. Create tests/test_phase6_checkpoint.py with pytest tests for T6.x cases  Target values: checkpoint 4,500,000 with RMSE=1.19, mTC=5.85Use deeplabcut, numpy, pandas, matplotlib, pathlib, cv2.
```

---

## PHASE 7: Evaluation Metrics

### 🎯 Objective

Implement all paper metrics with bootstrapped confidence intervals, and evaluate on all three datasets matching Table 3.

### 📁 Expected Output

```
/project/results/├── metrics/│   ├── training_metrics.csv│   ├── validation_metrics.csv│   ├── test_metrics.csv│   └── full_results_table.csv    # replication of Table 3└── figures/    ├── figure2c_trajectories.png    └── figure2d_spatial.png
```

### Target Metrics (Paper Table 3)

| Dataset | mKA | Recall | Precision |

|---------|-----|--------|-----------|

| Training | 92% | 93% | 85% |

| Validation | 85% | 86% | 86% |

| Test (LD) | 75% | 76% | 97% |

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T7.1 | mKA implementation | Correct formula | Unit test: known input returns expected mKA |

| T7.2 | Distance threshold | 40px used | Threshold set to 40 (< 4% of 1080px) |

| T7.3 | Prob threshold | 0.5 used | Optimal threshold = 0.5 as per paper |

| T7.4 | CI computation | 95% CI computed | Bootstrapped CIs present in output |

| T7.5 | Validation mKA | Near paper value | Validation mKA ≥ 75% |

| T7.6 | Per-structure metrics | 10 structure groups | Results broken down by laryngeal structure |

| T7.7 | mTC score | Near paper value | mTC between 4.0 and 8.0 pixels |

| T7.8 | Results table format | Matches Table 3 | Output CSV matches paper column structure |

### 💬 Claude Code Prompt — Phase 7

```
I need to implement all evaluation metrics from the vocal fold tracking paper.  Tasks:1. Create src/metrics/keypoint_metrics.py with class KeypointEvaluator:     compute_mKA(predictions, ground_truth,               distance_threshold=40, prob_threshold=0.5)   - Returns: mKA, TP, FP, FN, precision, recall   - Formula: mKA = TP / (TP + FP + FN)   - distance_threshold=40 pixels (< 4% of 1080p as stated in paper)     compute_bootstrapped_ci(metric_values, n_bootstrap=1000, alpha=0.05)   - Returns: (lower_ci, upper_ci)     evaluate_by_structure(predictions, ground_truth, structure_map)   - Returns dict of metrics per laryngeal structure group   - Structure groups match paper Table 3 exactly:     Left true VF + AC (keypoints 1-7)     Right true VF (8-13)     Midline supraglottis (14)     False left VF (15-20)     False right VF (21-26)     Midline points (27-29)     Left aryepiglottic fold (30-31)     Left cuneiform/corniculate (32-34)     Right aryepiglottic fold (35-36)     Right cuneiform/corniculate (37-39)  2. Create src/metrics/temporal_consistency.py with:   - compute_mTC(video_predictions, prob_threshold=0.5)     Formula: mTC = (1/N) * sum(D_i^b) over valid transitions   - visualize_trajectories(predictions, output_path) -> Figure 2C   - visualize_spatial(predictions, output_path) -> Figure 2D  3. Create src/metrics/run_full_evaluation.py that:   - Loads selected checkpoint from selected_checkpoint.txt   - Runs inference on training, validation, test datasets   - Computes all metrics with 95% bootstrapped CIs   - Generates full_results_table.csv matching Table 3 format   - Prints PASS/FAIL comparison against paper targets  4. Create src/metrics/generate_figures.py replicating Figure 2C and 2D  5. Create tests/test_phase7_metrics.py with:   - Unit tests for mKA with known synthetic inputs   - Unit tests for mTC with synthetic trajectory data   - Integration test running full evaluation pipeline  Use numpy, pandas, matplotlib, scipy, sklearn, deeplabcut, tqdm.
```

---

## PHASE 8: Real-Time Inference Pipeline

### 🎯 Objective

Build the live inference system that captures video, runs model inference, and overlays keypoints at ≥25fps with minimal latency.

### 📁 Expected Output

```
/project/src/realtime/├── capture.py├── inference_engine.py├── visualizer.py├── tracker_app.py└── benchmark.py
```

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T8.1 | Camera opens | Device opens | cv2.VideoCapture succeeds |

| T8.2 | Single frame inference | Keypoints returned | 39 keypoints returned per frame |

| T8.3 | Inference speed | Fast enough | Single frame inference < 33ms |

| T8.4 | FPS sustained | Continuous 30fps | 100-frame average FPS ≥ 25 |

| T8.5 | Keypoint overlay | Visible display | Colored dots appear on output frame |

| T8.6 | Prob threshold | Filtering works | Points below 0.5 prob not shown |

| T8.7 | Latency check | Minimal delay | End-to-end latency < 100ms |

| T8.8 | No memory leak | Stable memory | Memory usage stable over 1000 frames |

### 💬 Claude Code Prompt — Phase 8

```
I need to build a real-time vocal fold keypoint tracking application at 30fps.  Tasks:1. Create src/realtime/capture.py with class VideoCapture that:   - Supports webcam (device index), video file, and capture card input   - Sets resolution to 640x480 and FPS to 30 (matching paper)   - Runs capture in separate thread to avoid blocking inference   - Has get_frame(), is_opened(), release() methods  2. Create src/realtime/inference_engine.py with class InferenceEngine that:   - Loads the selected DLC checkpoint   - Implements process_frame(frame) -> (coords, probs, inference_time_ms)   - Preprocesses: resize to 224x224, normalize with ImageNet stats   - Runs TensorFlow inference session   - Post-processes: scale coordinates back to original frame dimensions   - Maintains rolling FPS counter (last 30 frames)  3. Create src/realtime/visualizer.py with class KeypointVisualizer that:   - One distinct color per structural group (10 groups)   - draw_keypoints(frame, coords, probs, prob_threshold=0.5)   - draw_skeleton(frame, coords, probs)   - draw_overlay(frame, fps, inference_time_ms)   - Returns annotated frame  4. Create src/realtime/tracker_app.py with class RealTimeTracker that:   - Integrates all three above classes   - Keyboard controls: Q=quit, S=save frame, R=record, +/-=threshold   - Saves session log to /project/results/realtime_log.csv  5. Create src/realtime/benchmark.py that:   - Runs 500 frames through pipeline on a test video   - Reports: mean FPS, std, min, max, mean inference time   - Reports: memory usage start/end   - PASS if mean FPS >= 25   - Saves to benchmark_results.json  6. Create tests/test_phase8_realtime.py (use test video, not live capture)  Use cv2, tensorflow, numpy, threading, time, psutil, pathlib, argparse.
```

---

## PHASE 9: Validation & Full System Test

### 🎯 Objective

Run complete end-to-end validation and confirm replication is successful within acceptable tolerance.

### Final Acceptance Criteria

| Metric | Paper Value | Minimum Acceptable |

|--------|-------------|-------------------|

| Validation mKA | 85% | 75% |

| Test mKA | 75% | 65% |

| Validation Precision | 86% | 76% |

| Validation Recall | 86% | 76% |

| mTC score | 5.85 px | < 10.0 px |

| Inference FPS | 30 | ≥ 25 |

| Model parameters | 5.3M | 5.0–5.6M |

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T9.1 | End-to-end pipeline | Runs without error | Full pipeline completes on test video |

| T9.2 | Validation mKA | ≥75% | mKA on validation set ≥ 75% |

| T9.3 | Test mKA | ≥65% | mKA on independent test set ≥ 65% |

| T9.4 | Real-time FPS | ≥25 FPS | Benchmark shows mean FPS ≥ 25 |

| T9.5 | Model size | ~5.3M params | Parameter count within 5% of paper |

| T9.6 | mTC score | <10.0 px | Temporal consistency below threshold |

| T9.7 | Figure reproduction | Plots match paper | Qualitative match to paper figures |

| T9.8 | Per-structure results | All groups computed | 10 structure breakdowns present |

### 💬 Claude Code Prompt — Phase 9

```
I need a complete end-to-end validation suite comparing my results to the paper.  Tasks:1. Create src/validation/end_to_end_test.py that:   - Runs the complete pipeline on a held-out test video   - Reports total execution time   - Verifies no crashes or memory errors   - Saves annotated output video to /project/results/test_output.mp4  2. Create src/validation/compare_to_paper.py that:   - Loads /project/results/metrics/full_results_table.csv   - Defines ALL paper target values (Table 3)   - For each metric: computes % difference from paper value   - Outputs /project/results/final_comparison_table.csv with:     [metric, paper_value, our_value, difference_pct, within_tolerance, tolerance_used]   - Prints color-coded PASS/FAIL table   - Tolerance: ±10 percentage points for accuracy, ±5px for mTC  3. Create src/validation/count_parameters.py that:   - Loads EfficientNet-B0 model   - Counts total trainable parameters (target: 5.3M)   - Reports FLOPs per forward pass (target: 1.55 billion)   - Reports FLOPs per second at 30fps (target: 46.4 billion)  4. Create src/validation/reproduce_figures.py replicating:   - Figure 2A: Training loss over 5M iterations   - Figure 2B: mTC over checkpoints, selected checkpoint in green   - Figure 2C: X/Y trajectories during consistency video   - Figure 2D: Spatial X/Y coordinate scatter   - Saves to /project/results/figure_reproductions/  5. Create src/validation/generate_replication_report.py that outputs   /project/results/replication_report.md with:   - Environment details, dataset stats, training config   - Full results table vs paper   - Pass/fail for all acceptance criteria   - Inline figure paths   - Notes on deviations from paper  6. Create tests/test_phase9_validation.py for all T9.x cases  Use numpy, pandas, matplotlib, tensorflow, cv2, pathlib.
```

---

## PHASE 10: Documentation & Code Cleanup

### 🎯 Objective

Produce clean, well-documented, reproducible code with a complete README and usage guide.

### ✅ Test Cases

| ID | Test | Expected Result | Pass Criteria |

|----|------|-----------------|---------------|

| T10.1 | README exists | File present | README.md > 500 words |

| T10.2 | Install instructions | Steps work | Following README installs env successfully |

| T10.3 | Full pipeline script | Runs end-to-end | run_full_pipeline.sh exits code 0 |

| T10.4 | All tests pass | Test suite green | pytest runs with 0 failures |

| T10.5 | Code docstrings | All documented | Every public function has docstring |

| T10.6 | Replication notes | Deviations noted | REPLICATION_NOTES.md exists, non-empty |

### 💬 Claude Code Prompt — Phase 10

```
Finalize documentation and cleanup for my vocal fold tracking replication project.  Tasks:1. Create README.md at /project/ with sections:   - Project Overview (paper being replicated)   - Hardware Requirements   - Installation (conda env setup, step by step)   - Dataset Setup   - Running Training   - Running Evaluation   - Running Real-Time Demo   - Expected Results (comparison table to paper)   - Project Structure (directory tree)   - Citation  2. Create REPLICATION_NOTES.md documenting:   - Parameters not specified in paper that were inferred   - Any deviations from methodology and why   - Dataset differences (if clinical data unavailable)   - Hardware differences (H100 vs consumer GPU)  3. Create scripts/run_full_pipeline.sh that:   - Activates conda environment   - Runs frame extraction -> validation -> training -> evaluation   - Generates results report   - Exits with code 0 on success  4. Create scripts/run_tests.sh that:   - Runs pytest on all test phases   - Generates HTML test report   - Prints final pass/fail count  5. Add Google-style docstrings to all Python functions across all phases  6. Create a Makefile with targets:   setup, extract, train, evaluate, demo, test, clean  7. Create tests/test_phase10_docs.py verifying:   - README.md exists with required sections   - All .py files have module-level docstrings   - All public functions have docstrings  Use pathlib, subprocess. Make all scripts executable.
```

---

## 📊 Full Test Summary Matrix

| Phase | Test Count | Key Pass Criteria |

|-------|-----------|-------------------|

| P1: Environment | 7 | DLC 2.3.6 installed, GPU available |

| P2: Data | 7 | ≥20 videos, all readable, manifest complete |

| P3: Frames | 8 | ~1,444 total frames, no blanks, no duplicates |

| P4: Annotation | 8 | 39 keypoints, ≥1,000 annotated frames |

| P5: Training | 8 | Loss decreasing, checkpoints saving every 50k |

| P6: Checkpoint | 7 | Optimal checkpoint selected near 4.5M iterations |

| P7: Metrics | 8 | Validation mKA ≥75%, mTC <10px |

| P8: Real-Time | 8 | ≥25 FPS sustained, <100ms latency |

| P9: Validation | 8 | Results within 10% of paper values |

| P10: Docs | 6 | README complete, all tests pass |

| **Total** | **75 tests** | |

---

## ⚠️ Risk Register

| Risk | Probability | Impact | Mitigation |

|------|-------------|--------|------------|

| Condition imbalance (no clinical data) | Medium | Medium | Collect as many VFI/pathology YouTube videos as healthy ones; accept distribution mismatch |

| DLC 2.3.6 install conflicts | Medium | High | Use exact conda env file; isolated environment |

| Consumer GPU slower than H100 | Medium | Medium | Reduce batch size to 4; expect longer training |

| Annotation accuracy without clinician | High | High | Study paper Figure 1A carefully; cross-check with an ENT clinician if possible; use DLC confidence scores to flag uncertain frames |

| 5M iterations too slow | Medium | Medium | Evaluate intermediate checkpoint at 2.5M |

| Real-time FPS below 30 | Low | Medium | TensorRT optimization; reduce input resolution |

---

## 🔗 Key Resources

| Resource | Link |

|----------|------|

| Paper DOI | [https://doi.org/10.1002/lary.70104](https://doi.org/10.1002/lary.70104) |

| Author contact | [akoivu@meei.harvard.edu](mailto:akoivu@meei.harvard.edu) |

| DeepLabCut 2.3.6 | [https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.3.6](https://github.com/DeepLabCut/DeepLabCut/releases/tag/v2.3.6) |

| Open VF Dataset (Zenodo) | [https://zenodo.org/records/3603185](https://zenodo.org/records/3603185) |

| EfficientNet paper | [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946) |

| ImageNet pretrained weights | Available via keras.applications |

---

*Roadmap v1.0 — VF Tracking Replication Project — Koivu et al. 2026*