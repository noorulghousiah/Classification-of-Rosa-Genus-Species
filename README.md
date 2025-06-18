
# 🌿 Family Classification of Plants using EfficientNet (PlantCLEF2024 Subset) [PlantCLEF2024](https://www.kaggle.com/competitions/plantclef-2025)

This project performs fine-grained classification of **5 families of plants : Asteraceae, Brassicaceae, Fabaceae, Poaceae, Rosaceae**.

---
# Multi-Family-Plant-Identification
- kaggle link: https://www.kaggle.com/code/ghousiah/multi-family-plant-classification
- kaggle dataset used: https://www.kaggle.com/datasets/ghousiah/plantclef/data
- [![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-blue?logo=kaggle)]([https://www.kaggle.com/your-username/your-notebook-name](https://www.kaggle.com/code/ghousiah/multi-family-plant-classification))

---
## 📁 Dataset

**Source**: `PlantCLEF2024singleplanttrainingdata.csv` size: 750M from [Download Link](https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/)

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
```

---  
## **Time Taken**
1) Download and Cache Dataset Images = 2 hour
Downloading 10000 images to 'plant_cache'...
100%|██████████| 10000/10000 [2:08:28<00:00,  1.30it/s]
   
2) Training Time for 5 epochs=

Epoch 1/5 | Time: 72.76s | Train Loss: 1.6177 | Val Loss: 1.8418 | Val F1: 0.5251
✅ Best model saved as 'best_model_epoch_1.pth'
Epoch 2/5 | Time: 72.43s | Train Loss: 1.6452 | Val Loss: 1.7294 | Val F1: 0.5173
Epoch 3/5 | Time: 72.38s | Train Loss: 1.5792 | Val Loss: 1.7595 | Val F1: 0.5547
✅ Best model saved as 'best_model_epoch_3.pth'
Epoch 4/5 | Time: 72.81s | Train Loss: 1.5572 | Val Loss: 1.7007 | Val F1: 0.5564
✅ Best model saved as 'best_model_epoch_4.pth'
Epoch 5/5 | Time: 72.15s | Train Loss: 1.5704 | Val Loss: 1.8604 | Val F1: 0.5596
✅ Best model saved as 'best_model_epoch_5.pth'
✅ Final model checkpoint saved as 'final_checkpoint.pth'
⏱️ Total training time: 6.05 minutes


---  
## **Evaluation**
📊 Evaluation Metrics:
✅ F1 Score   : 0.5635
✅ Precision  : 0.5958
✅ Recall     : 0.5847
