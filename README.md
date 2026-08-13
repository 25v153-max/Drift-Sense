# DRIFT-SENSE

### Scale and Rotation-Aware Target Localization for DRAM Images

## Overview

DRIFT-SENSE is a computer-vision based localization system designed to detect and accurately localize target patterns in DRAM images under scale variation, rotation, noise, and blur.

## Problem

Small target patterns in semiconductor/DRAM images can shift in position and appearance due to scale variation, rotation, noise, and image degradation. Conventional template matching may fail under these conditions.

## Solution

DRIFT-SENSE uses multi-scale and rotation-aware template matching to locate the target and estimate its position, scale, and rotation.

**Pipeline:**

Image Generation → Preprocessing → Multi-Scale Search → Rotation Search → Template Matching → Localization → Error Evaluation

## V4 Approach

V4 extends the localization search range to:

- Scale: 85–120 pixels
- Rotation: -8° to +8°
- Rotation step: 0.5°

This improves robustness compared with V3, which used a smaller search range.

## Results

### V3 vs V4 — 40 Case Evaluation

| Metric | V3 | V4 |
|---|---:|---:|
| Success Rate | 45.00% | 100.00% |
| Mean Error | 24.98 px | 0.36 px |
| Maximum Error | 131.10 px | 1.41 px |
| Mean Confidence | 0.5105 | 0.9126 |

V4 successfully localized all 40 test cases, including difficult combinations of scale, rotation, noise, and blur.

## Hard-Case Validation

- Target Size: 120 × 120
- Rotation: +8°
- Noise: 0
- Blur: 5 × 5
- Localization Error: **0.00 pixels**
- Confidence: **0.9367**

## Technologies Used

- Python
- OpenCV
- NumPy
- CSV/JSON
- Template Matching
- Multi-Scale Search
- Rotation-Aware Search

## Project Structure

```text
DRIFT-SENSE/
├── data/
├── results/
├── generate_dataset.py
├── localize.py
├── localize_rotation.py
├── localize_scale_rotation.py
├── localize_v4.py
├── evaluate.py
├── stress_test.py
├── compare_v3_v4.py
└── test_v4_hardcase.py