# Data and Model License Notes

This repository contains code, tests, and documentation for a strawberry YOLO/KGCV pipeline. It does not redistribute raw images, raw JSON labels, trained model weights, KGCV model files, or YOLO checkpoint files.

## Project Code

The code in this repository is provided under the MIT License in `LICENSE`.

## AIHub-Style Strawberry Data

The converter is designed for AIHub-style strawberry JSON labels and images. The actual data files are not included in this repository.

Before publishing derived datasets, model weights, figures, or benchmark results, check the license and redistribution terms of the original dataset provider.

## KGCV_Strawberry

The roadmap references `DigitalAgUMN/KGCV_Strawberry` and its Zenodo dataset/model release. This repository does not bundle those files.

If you use KGCV code, pretrained models, or Zenodo data, cite the original authors and follow the license terms of those upstream resources.

## Ultralytics YOLO26

The train/eval wrappers call Ultralytics through the `ultralytics` Python package when actual training or evaluation is run. This repository does not redistribute `yolo26*.pt` weights.

Review the Ultralytics license terms before public deployment, commercial use, or redistribution of trained weights.

## Generated Outputs

The following should remain local unless you have checked data and model redistribution rights:

- converted YOLO image/label datasets
- visual sanity-check images
- trained `.pt`, `.pth`, `.onnx`, or other model files
- crop images exported from YOLO predictions
- prediction CSVs derived from private or restricted datasets
