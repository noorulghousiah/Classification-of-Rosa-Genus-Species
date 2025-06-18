# Multi-Species-Plant-Identification


# 🌿 Family Classification of Plants using EfficientNet (PlantCLEF2024 Subset)
[PlantCLEF2024](https://www.kaggle.com/competitions/plantclef-2025)

This project performs fine-grained classification of **5 families of plants : Asteraceae, Brassicaceae, Fabaceae, Poaceae, Rosaceae**.

## 📁 Dataset

**Source**: `PlantCLEF2024singleplanttrainingdata.csv`  from [Download Link](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/)

**Preprocessing**:
1) Dataset filtered to only includes labeled images from Family: {'Asteraceae': 0, 'Brassicaceae': 1, 'Fabaceae': 2, 'Poaceae': 3, 'Rosaceae': 4}.
2) The dataset images is downloaded (from url links) and cached to the notebook environments.
3) The dataset is split for training and validation in 8:2 ratio.
4) Each image is transformed into tensor, and then resize to 400 x 400.
5) After that, **quadrat images** are created by combining 4 plant images in a grid form for each quadrat image.
6) The quadrat images is resized to 96 x 96, transformed into tensor, and normalised.
7) Training and validation dataset consist of preprocessed quadrat images.

---

## 🧠 Pretrained Model
- **Base Model**: [`efficientnet_b0`](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- **Framework**: PyTorch
- **Model Input Size**: 96 x 96
- **Model Output Class**: 5 Families
- **Modifications**:
  - Replaced classification head with `nn.Linear(in_features, 5)`
  - Frozen all other layers to speed up training on CPU

```python
model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=NUM_CLASSES)
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
