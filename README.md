# ECG TransUNet Open-Source Package

This repository contains the ECG TransUNet training code used in our project.

## Project structure

The project has two main parts:

1. Base model training on the public training dataset.
2. Correction / fine-tuning on the curated excellent-data dataset.

## Dataset usage

For base-model training, users should:

1. Download the dataset package that contains `ECGModelTrainData`.
2. Update the dataset path in [config.ini](D:\ecgmodeltrain\heartfenxi\ablistimunet\config.ini).
3. Run one of the base-training entry scripts to start training.

The dataset can be traced through the published DOI records in ScienceDB.

For the January 2025 correction dataset, the paired DOI recorded in our knowledge base is:

- `10.57760/sciencedb.35878`

## Main files

### Base training

- [train_base_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_base_model.py): base segmentation training
- [train_gan_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_gan_model.py): GAN-assisted training
- [train_gan_base_model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\train_gan_base_model.py): GAN-base variant training
- [ECGDataset.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ECGDataset.py): base dataset loader
- [ECGDataAugmentation.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ECGDataAugmentation.py): augmentation utilities

### Correction / fine-tuning

- [excellent_data.hdf](D:\ecgmodeltrain\heartfenxi\ablistimunet\excellent_data.hdf): curated correction dataset
- [ExcellentDataset.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\ExcellentDataset.py): excellent-data loader
- [finetune_gan_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_gan_correction.py): GAN correction model training
- [finetune_base_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_base_correction.py): base correction training
- [finetune_gan_base_correction.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\finetune_gan_base_correction.py): GAN-base correction training

### Models and utilities

- [model.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\model.py)
- [models.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\models.py)
- [transmodel.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\transmodel.py)
- [losses.py](D:\ecgmodeltrain\heartfenxi\ablistimunet\losses.py)
- [utils](D:\ecgmodeltrain\heartfenxi\ablistimunet\utils)

## Repository scope

This open-source package keeps only the training-related code and the files directly required to run training.

## Notes

- Core dataset and model module names such as `ECGDataset.py` remain unchanged.
- The training entry scripts were renamed to make the open-source workflow easier to understand.
