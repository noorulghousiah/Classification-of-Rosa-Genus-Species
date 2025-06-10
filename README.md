# Classification-of-Rosa-Genus-Species

# 🌿 Rosa Species Classification using EfficientNet (PlantCLEF2024 Subset)

This project performs fine-grained classification of **39 species under the _Rosa_ genus** using images from the [PlantCLEF2024](https://www.kaggle.com/competitions/plantclef-2025) dataset. The model is based on **EfficientNet-B0**, trained under limited compute resources with resized images.

---

## 📁 Dataset

**Source**: `PlantCLEF2024singleplanttrainingdata.csv`  
**Filtered**: Only includes labeled images from the **Rosa genus**  
**Image Size**: Resized to **224×224**  
**Classes**: 39 Rosa species


---

## 🧠 Pretrained Model

- **Base Model**: [`efficientnet_b0`](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- **Framework**: PyTorch
- **Modifications**:
  - Replaced classification head with `nn.Linear(in_features, 39)`
  - Frozen all other layers to speed up training on CPU

```python
model_ft = models.efficientnet_b0(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier[1] = nn.Linear(model_ft.classifier[1].in_features, 39)
