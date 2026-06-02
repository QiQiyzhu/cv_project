# Video Super-Resolution Paper Release

This repository is a curated release of the course project code that is directly relevant to the paper.

It keeps only:

- `Part 1`: classical baselines and SRCNN
- `Part 2`: benchmark and wild-video inference for the paper-used advanced models
- `Part 3`: the two final paper methods
  - `Gated Residual Fusion`
  - `Motion-Compensated Gated Residual Fusion`

It intentionally excludes older exploratory branches such as EDVR, SwinIR-based fusion experiments, heuristic fusion variants, and server-specific launch helpers.

## What Is Included

```text
README.md
requirements.txt
assets/
  visual_results/
scripts/
src/
```

### Source modules

- `src/vsr_part1/`: Part 1 datasets, baselines, SRCNN, metrics, and I/O helpers
- `src/vsr_part2/`: in-repo runtimes for `BasicVSR`, `BasicVSR++`, `Real-ESRGAN`, and `HAT`
- `src/vsr_part3/`: Part 3 datasets, fusion models, and fusion inference
- `src/vsr_benchmarks.py`: GT-ready benchmark registry used by the all-metrics evaluator

### Release scripts

- Part 1
  - `scripts/train_srcnn.py`
  - `scripts/run_srcnn_inference_dataset.py`
  - `scripts/run_part1_baselines_dataset.py`
  - `scripts/evaluate_part1_dataset.py`
- Part 2
  - `scripts/run_part2_inference_dataset.py`
  - `scripts/run_part2_basicvsr_mandatory.py`
  - `scripts/run_part2_basicvsrpp_mandatory.py`
  - `scripts/run_part2_realesrgan_mandatory.py`
  - `scripts/run_part2_hat_l_mandatory.py`
- Part 3
  - `scripts/train_part3_gated_residual_fusion.py`
  - `scripts/train_part3_motion_compensated_gated_residual_fusion.py`
  - `scripts/run_part3_fusion_dataset.py`
  - `scripts/run_part3_mandatory_fusion.py`
- Evaluation and benchmark prep
  - `scripts/evaluate_all_metrics_dataset.py`
  - `scripts/make_lr_dataset_from_sequence_list.py`
  - `scripts/prepare_wild_video_benchmark.py`
  - `scripts/list_gt_ready_benchmarks.py`

## Dependencies

The release was built around Python 3.10+ with PyTorch.

Install the main packages with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notes

- `torch`, `torchvision`: required everywhere
- `opencv-python`: used by mandatory video scripts and MP4 export
- `scikit-image`: used for PSNR/SSIM
- `lpips`: required for `LPIPS` and `tLPIPS`
- `mmcv`, `mmengine`: required for `BasicVSR++`
- `HAT` in this release uses the in-repo architecture files and does not require `timm`

## Recommended Data Layout

This release assumes the original project layout convention:

```text
project/
  data/
    official/
      reds_sample/
        lr_sequences/
        gt_sequences/
      vimeo_lr/
        lr_sequences/
      wild_video_benchmark_112x64/
        lr_sequences/
        gt_sequences/
    benchmarks/
      vimeo90k/
        original/
          sequences/
          sep_trainlist.txt
          sep_testlist.txt
  wild_video/
```

For nested benchmark roots, each sequence directory contains frames such as `00000000.png` or `im1.png`.

## Part 1

### Train SRCNN

Example on the official Vimeo-90K split:

```bash
PYTHONPATH=src python scripts/train_srcnn.py \
  --train-hr-dir /path/to/project/data/benchmarks/vimeo90k/original \
  --train-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_trainlist.txt \
  --val-hr-dir /path/to/project/data/benchmarks/vimeo90k/original \
  --val-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_testlist.txt \
  --output-dir /path/to/outputs/part1/srcnn_vimeo90k_y \
  --scale 4 \
  --patch-size 32 \
  --samples-per-image 1 \
  --optimizer adam \
  --color-space y
```

### Run classical baselines

```bash
PYTHONPATH=src python scripts/run_part1_baselines_dataset.py \
  --input-root /path/to/project/data/official/reds_sample/lr_sequences \
  --output-root /path/to/outputs/part1/reds_sample_baselines \
  --scale 4 \
  --save-sharpened
```

### Run SRCNN inference

```bash
PYTHONPATH=src python scripts/run_srcnn_inference_dataset.py \
  --input-root /path/to/project/results/eval_assets/vimeo90k_full_test_lr \
  --output-root /path/to/outputs/part1/vimeo90k_full_test_srcnn \
  --checkpoint /path/to/srcnn/best.pt \
  --scale 4
```

### Evaluate PSNR and SSIM

```bash
PYTHONPATH=src python scripts/evaluate_part1_dataset.py \
  --pred-root /path/to/outputs/part1/vimeo90k_full_test_srcnn \
  --gt-root /path/to/project/data/benchmarks/vimeo90k/original/sequences \
  --channel y \
  --output-json /path/to/outputs/metrics/srcnn_part1.json
```

## Part 2

### Paper-used models

- `BasicVSR`
- `BasicVSR++`
- `Real-ESRGAN`
- `HAT` / `HAT-L`

`BasicVSR`, `BasicVSR++`, and `Real-ESRGAN` are the paper’s Part 2 baselines.
`HAT-L` is kept because it is the detail branch used by the final Part 3 models.

### Generic dataset inference

```bash
PYTHONPATH=src python scripts/run_part2_inference_dataset.py \
  --input-root /path/to/project/results/eval_assets/vimeo90k_full_test_lr \
  --output-root /path/to/outputs/part2/vimeo90k_full_test_basicvsrpp \
  --model-name basicvsr++ \
  --checkpoint /path/to/basicvsrpp.pth \
  --device cuda:0
```

Example for the Part 3 detail branch:

```bash
PYTHONPATH=src python scripts/run_part2_inference_dataset.py \
  --input-root /path/to/project/results/eval_assets/vimeo90k_full_test_lr \
  --output-root /path/to/outputs/part2/vimeo90k_full_test_hat_l \
  --model-name hat_l \
  --checkpoint /path/to/HAT-L_SRx4_ImageNet-pretrain.pth \
  --device cuda:0
```

### Mandatory benchmark and wild-video runs

BasicVSR:

```bash
PYTHONPATH=src python scripts/run_part2_basicvsr_mandatory.py \
  --reds-root /path/to/project/data/official/reds_sample/lr_sequences \
  --vimeo-root /path/to/project/data/official/vimeo_lr/lr_sequences \
  --wild-root /path/to/project/wild_video \
  --output-root /path/to/outputs/part2/basicvsr_mandatory
```

BasicVSR++:

```bash
PYTHONPATH=src python scripts/run_part2_basicvsrpp_mandatory.py \
  --reds-root /path/to/project/data/official/reds_sample/lr_sequences \
  --vimeo-root /path/to/project/data/official/vimeo_lr/lr_sequences \
  --wild-root /path/to/project/wild_video \
  --output-root /path/to/outputs/part2/basicvsrpp_mandatory
```

Real-ESRGAN:

```bash
PYTHONPATH=src python scripts/run_part2_realesrgan_mandatory.py \
  --reds-root /path/to/project/data/official/reds_sample/lr_sequences \
  --vimeo-root /path/to/project/data/official/vimeo_lr/lr_sequences \
  --wild-root /path/to/project/wild_video \
  --output-root /path/to/outputs/part2/realesrgan_mandatory
```

HAT-L:

```bash
PYTHONPATH=src python scripts/run_part2_hat_l_mandatory.py \
  --reds-root /path/to/project/data/official/reds_sample/lr_sequences \
  --vimeo-root /path/to/project/data/official/vimeo_lr/lr_sequences \
  --wild-root /path/to/project/wild_video \
  --output-root /path/to/outputs/part2/hat_l_mandatory \
  --checkpoint /path/to/HAT-L_SRx4_ImageNet-pretrain.pth
```

## Part 3

The paper keeps exactly two final methods:

- `Gated Residual Fusion`
- `Motion-Compensated Gated Residual Fusion`

Both are trained with:

- base branch: `BasicVSR++`
- detail branch: `HAT-L`

### 1. Prepare the branch outputs

Run `BasicVSR++` and `HAT-L` first on the same train and validation sequence sets.

### 2. Train Gated Residual Fusion

```bash
PYTHONPATH=src python scripts/train_part3_gated_residual_fusion.py \
  --train-gt-root /path/to/project/data/benchmarks/vimeo90k/original \
  --train-base-root /path/to/outputs/part2/basicvsrpp_train \
  --train-gen-root /path/to/outputs/part2/hat_l_train \
  --train-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_trainlist.txt \
  --val-gt-root /path/to/project/data/benchmarks/vimeo90k/original \
  --val-base-root /path/to/outputs/part2/basicvsrpp_val \
  --val-gen-root /path/to/outputs/part2/hat_l_val \
  --val-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_testlist.txt \
  --output-dir /path/to/outputs/part3/gated_residual_fusion \
  --batch-size 4 \
  --epochs 20 \
  --gated-residual \
  --use-detail-delta-features \
  --use-dual-reference-features \
  --fusion-mode full_blend
```

### 3. Train Motion-Compensated Gated Residual Fusion

```bash
PYTHONPATH=src python scripts/train_part3_motion_compensated_gated_residual_fusion.py \
  --train-gt-root /path/to/project/data/benchmarks/vimeo90k/original \
  --train-base-root /path/to/outputs/part2/basicvsrpp_train \
  --train-gen-root /path/to/outputs/part2/hat_l_train \
  --train-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_trainlist.txt \
  --val-gt-root /path/to/project/data/benchmarks/vimeo90k/original \
  --val-base-root /path/to/outputs/part2/basicvsrpp_val \
  --val-gen-root /path/to/outputs/part2/hat_l_val \
  --val-list-file /path/to/project/data/benchmarks/vimeo90k/original/sep_testlist.txt \
  --output-dir /path/to/outputs/part3/motion_compensated_grf \
  --batch-size 4 \
  --epochs 5 \
  --fusion-mode detail_residual
```

### 4. Run Part 3 fusion

```bash
PYTHONPATH=src python scripts/run_part3_fusion_dataset.py \
  --base-root /path/to/outputs/part2/vimeo90k_full_test_basicvsrpp \
  --gen-root /path/to/outputs/part2/vimeo90k_full_test_hat_l \
  --output-root /path/to/outputs/part3/vimeo90k_full_test_grf \
  --checkpoint /path/to/outputs/part3/gated_residual_fusion/best.pt \
  --device cuda:0
```

Mandatory benchmark export:

```bash
PYTHONPATH=src python scripts/run_part3_mandatory_fusion.py \
  --base-root /path/to/outputs/part2/basicvsrpp_mandatory \
  --gen-root /path/to/outputs/part2/hat_l_mandatory \
  --output-root /path/to/outputs/part3/grf_mandatory \
  --checkpoint /path/to/outputs/part3/gated_residual_fusion/best.pt
```

## Full Metrics Evaluation

This release includes a unified evaluator for:

- `PSNR`
- `SSIM`
- `LPIPS`
- `FID`
- `tLPIPS`

List registered GT-ready datasets:

```bash
PYTHONPATH=src python scripts/list_gt_ready_benchmarks.py
```

Example on a GT-ready benchmark:

```bash
PYTHONPATH=src python scripts/evaluate_all_metrics_dataset.py \
  --pred-root /path/to/outputs/part3/vimeo90k_full_test_grf \
  --dataset-name vimeo90k_full_test \
  --device cuda:0 \
  --output-json /path/to/outputs/metrics/vimeo90k_full_test_grf_all_metrics.json
```

Example with explicit GT:

```bash
PYTHONPATH=src python scripts/evaluate_all_metrics_dataset.py \
  --pred-root /path/to/outputs/part3/reds_sample_grf \
  --gt-root /path/to/project/data/official/reds_sample/gt_sequences \
  --device cuda:0 \
  --output-json /path/to/outputs/metrics/reds_sample_grf_all_metrics.json
```

## Wild Benchmark Preparation

The release also keeps the script used to convert extracted wild frames into a paired benchmark-style dataset:

```bash
PYTHONPATH=src python scripts/prepare_wild_video_benchmark.py \
  --frames-root /path/to/extracted_wild_frames \
  --output-root /path/to/project/data/official/wild_video_benchmark_112x64
```

This creates:

- `gt_sequences/` at `448x256`
- `lr_sequences/` at `112x64`

## Weights

### Official pretrained weights used by the release

These can be supplied manually, or in most cases the runtime will auto-download them:

- `BasicVSR`: official OpenMMLab Vimeo-90K BI checkpoint
- `BasicVSR++`: official OpenMMLab Vimeo-90K BI checkpoint
- `Real-ESRGAN`: official `RealESRGAN_x4plus.pth`
- `HAT-L`: `checkpoints/hat/HAT-L_SRx4_ImageNet-pretrain.pth`

### Project checkpoints used in the paper

These were trained and stored on the project server:

- Part 1 `SRCNN`
  - `/data/chenhui/cv_project/checkpoints/part1/srcnn_vimeo90k_subset2000_y_adam/best.pt`
- Part 3 `Gated Residual Fusion`
  - `/data/chenhui/cv_project/results/part3/vimeo_10000_1000_learned_fusion_basicvsrpp_hat_l_yuqi/best.pt`
- Part 3 `Motion-Compensated Gated Residual Fusion`
  - `/data/chenhui/cv_project/results/part3/vimeo_10000_1000_motion_compensated_basicvsrpp_hat_l_v1_20260514/best.pt`

## Quantitative Results

Below are the archived all-metrics results that correspond to the final paper methods and baselines.

### Vimeo-90K full test

| Model | PSNR | SSIM | LPIPS | FID | tLPIPS gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| SRCNN | 28.489470 | 0.825394 | 0.326254 | 42.014677 | 0.015973 |
| BasicVSR | see note | 0.935189 | 0.107006 | 3.718550 | 0.004823 |
| BasicVSR++ | 35.535247 | 0.940956 | 0.097496 | 3.144611 | 0.004961 |
| Real-ESRGAN | 27.329255 | 0.810751 | 0.166907 | 11.214392 | 0.011406 |
| Gated Residual Fusion | 35.491048 | 0.940178 | 0.099660 | 3.272742 | 0.004970 |
| Motion-Compensated Gated Residual Fusion | 35.482253 | 0.940104 | 0.099613 | 3.268478 | 0.004973 |

Note: the archived BasicVSR full-test JSON contains some infinite-PSNR sequences, so that run is best read from the server-side summary table using finite-only aggregation.

### Mandatory datasets

| Dataset | Model | PSNR | SSIM | LPIPS | FID | tLPIPS gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| REDS sample | BasicVSR | 31.223329 | 0.888275 | 0.208600 | 6.692776 | 0.012615 |
| REDS sample | BasicVSR++ | 31.560979 | 0.893653 | 0.199891 | 6.187013 | 0.011948 |
| REDS sample | Real-ESRGAN | 25.128972 | 0.727981 | 0.198394 | 35.931201 | 0.007099 |
| REDS sample | Gated Residual Fusion | 31.264622 | 0.888334 | 0.179038 | 6.700076 | 0.010983 |
| REDS sample | Motion-Compensated Gated Residual Fusion | 31.511506 | 0.892172 | 0.199125 | 6.405150 | 0.012179 |
| Vimeo-LR mandatory | BasicVSR | 33.889255 | 0.927890 | 0.114803 | 23.895205 | 0.004667 |
| Vimeo-LR mandatory | BasicVSR++ | 34.503091 | 0.934343 | 0.104722 | 20.802051 | 0.004787 |
| Vimeo-LR mandatory | Real-ESRGAN | 26.452849 | 0.795343 | 0.173291 | 61.061178 | 0.011544 |
| Vimeo-LR mandatory | Gated Residual Fusion | 34.078802 | 0.930054 | 0.100076 | 21.817375 | 0.004760 |
| Vimeo-LR mandatory | Motion-Compensated Gated Residual Fusion | 34.357736 | 0.932535 | 0.105766 | 21.326284 | 0.004943 |
| Wild benchmark 112x64 | BasicVSR++ | 23.260570 | 0.737467 | 0.277294 | 94.445002 | 0.012741 |
| Wild benchmark 112x64 | Real-ESRGAN | 25.359723 | 0.760362 | 0.187503 | 69.795045 | 0.013486 |
| Wild benchmark 112x64 | Gated Residual Fusion | 24.387285 | 0.765882 | 0.253547 | 86.037693 | 0.010631 |
| Wild benchmark 112x64 | Motion-Compensated Gated Residual Fusion | 23.879241 | 0.752948 | 0.263511 | 89.609514 | 0.011417 |

## Archived Result Paths

### Metric JSONs

- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part1_srcnn_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part2_basicvsr_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part2_basicvsrpp_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part2_realesrgan_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part3_basicvsrpp_fusion_final_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/vimeo90k_full_test_part3_motion_compensated_all_metrics.json`
- `/data/chenhui/cv_project/results/metrics/completed_all_metrics_summary_20260515_post_alias.md`

### Visual result directories on the project server

- `Gated Residual Fusion`
  - `/data/chenhui/cv_project/results/part3/vimeo90k_full_test_basicvsrpp_fusion_final`
- `Motion-Compensated Gated Residual Fusion`
  - `/data/chenhui/cv_project/results/part3/motion_compensated_four_benchmarks_20260514/preds/vimeo90k_full_test`
- Wild fused videos
  - `/data/chenhui/cv_project/results/part3/motion_compensated_four_benchmarks_20260514/preds/wild_video_videos`

## Visual Results

The release includes one representative full-test example under:

- `assets/visual_results/vimeo90k_full_test_00001_0266/`

Files:

- `lr.png`
- `gt.png`
- `basicvsr.png`
- `basicvsrpp.png`
- `realesrgan.png`
- `hat_l.png`
- `gated_residual_fusion.png`
- `motion_compensated_gated_residual_fusion.png`

Representative comparison:

| LR | GT |
| --- | --- |
| ![](assets/visual_results/vimeo90k_full_test_00001_0266/lr.png) | ![](assets/visual_results/vimeo90k_full_test_00001_0266/gt.png) |

| BasicVSR | BasicVSR++ |
| --- | --- |
| ![](assets/visual_results/vimeo90k_full_test_00001_0266/basicvsr.png) | ![](assets/visual_results/vimeo90k_full_test_00001_0266/basicvsrpp.png) |

| Real-ESRGAN | HAT-L |
| --- | --- |
| ![](assets/visual_results/vimeo90k_full_test_00001_0266/realesrgan.png) | ![](assets/visual_results/vimeo90k_full_test_00001_0266/hat_l.png) |

| Gated Residual Fusion | Motion-Compensated Gated Residual Fusion |
| --- | --- |
| ![](assets/visual_results/vimeo90k_full_test_00001_0266/gated_residual_fusion.png) | ![](assets/visual_results/vimeo90k_full_test_00001_0266/motion_compensated_gated_residual_fusion.png) |

## Notes

- This release is a paper-oriented subset, not the full experiment sandbox.
- The full project repository may contain additional exploratory code and older result aliases not included here.
- The release keeps `HAT-L` because it is part of the final Part 3 branch pair, even though it is not a standalone paper section.
