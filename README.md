# ECG-TransUNet

This repository contains the training code for **ECG-TransUNet**, our ECG heartbeat annotation and correction network.

The open-source package is intentionally lightweight and keeps only the files required for model training and fine-tuning.

## Overview

The codebase is organized into two training stages:

1. **Base training**
   Train the initial ECG segmentation model from the public training dataset bundle.
2. **Correction fine-tuning**
   Fine-tune the model on the curated high-quality dataset used for correction and refinement.

## Dataset Setup

### Base training dataset

To run base training:

1. Download the dataset bundle that contains `ECGModelTrainData`.
2. Update the dataset path in [config.ini](D:\ecgmodeltrain\heartfenxi\ablistimunet\config.ini).
3. Run one of the training entry scripts listed below.

### Correction / fine-tuning dataset

This repository also includes the curated correction dataset:

- [excellent_data.hdf](D:\ecgmodeltrain\heartfenxi\ablistimunet\excellent_data.hdf)

The paired ScienceDB DOI for the **January 2025** dataset is:

- `10.57760/sciencedb.35878`

## Training Entry Scripts

### Base training

- [train_base_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_base_model.py): base ECG segmentation training
- [train_gan_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_gan_model.py): GAN-assisted training
- [train_gan_base_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_gan_base_model.py): GAN-base training variant

### Correction fine-tuning

- [finetune_gan_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_gan_correction.py): GAN correction fine-tuning
- [finetune_base_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_base_correction.py): base correction fine-tuning
- [finetune_gan_base_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_gan_base_correction.py): GAN-base correction fine-tuning

## Core Components

- [ECGDataset.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ECGDataset.py): base dataset loader
- [ExcellentDataset.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ExcellentDataset.py): curated fine-tuning dataset loader
- [ECGDataAugmentation.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ECGDataAugmentation.py): augmentation and label-processing utilities
- [model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\model.py): main network definition
- [models.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\models.py): supporting model components
- [transmodel.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\transmodel.py): transformer-related modules
- [losses.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\losses.py): loss definitions
- [utils](D:\ecgmodeltrain\heartfenxi\ablistimunet\utils): training utilities and feature helpers

## Related Repositories

- [SwineSync-OpenSource](https://github.com/zengzhengcheng/SwineSync-OpenSource): broader data-processing, annotation, and workflow repository
- [OpenCalori-Swine](https://github.com/zengzhengcheng/OpenCalori-Swine): heat-production calculation repository used in the broader data pipeline

## Scope

This repository is the **model-specific training repository** for ECG-TransUNet.

It is narrower in scope than the broader processing repository and is intended for:

- model training
- correction fine-tuning
- model-related utilities directly required by training

## Notes

- Core dataset and model module names such as `ECGDataset.py` remain unchanged.
- Training entry scripts were renamed to make the workflow easier to understand for public users.
- The large HDF dataset file is tracked with Git LFS.
