# Multi-Species-Plant-Identification


# 🌿 Plant Family Classification using EfficientNet (PlantCLEF2024 Subset)

This project performs fine-grained classification of **5 family of plant** using images from the [PlantCLEF2024](https://www.kaggle.com/competitions/plantclef-2025) dataset. The model is based on **EfficientNet-B0**, trained under limited compute resources with resized images.

---

## 📁 Dataset

**Source**: `PlantCLEF2024singleplanttrainingdata.csv`  
**Filtered**: Only includes labeled images from Family: {'Asteraceae': 0, 'Brassicaceae': 1, 'Fabaceae': 2, 'Poaceae': 3, 'Rosaceae': 4}
**Image Size**: Resized to **96×96**  
**Classes**: 5 Rosa species


---

## 🧠 Pretrained Model

- **Base Model**: [`efficientnet_b0`](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- **Framework**: PyTorch
- **Modifications**:
  - Replaced classification head with `nn.Linear(in_features, 10)`
  - Frozen all other layers to speed up training on CPU

```python
model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=NUM_CLASSES)
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
