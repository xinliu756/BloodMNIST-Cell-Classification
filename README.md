# Blood Cell Classification with CNN (BloodMNIST)

A deep learning project for classifying 8 types of blood cells using ResNet18 on the BloodMNIST dataset. This project also demonstrates the impact of transfer learning (pretrained weights) vs training from scratch.

---

## Dataset

| Item | Detail |
|------|--------|
| Dataset | BloodMNIST (MedMNIST v2) |
| Source | [MedMNIST](https://medmnist.com/) |
| Total Images | 17,092 |
| Train / Val / Test | 11,959 / 1,712 / 3,421 |
| Image Size | 28×28 pixels, RGB |
| Classes | 8 types of blood cells |

### 8 Blood Cell Types

| Label | Cell Type | Chinese |
|-------|-----------|---------|
| 0 | Basophil | 嗜碱性粒细胞 |
| 1 | Eosinophil | 嗜酸性粒细胞 |
| 2 | Erythroblast | 幼红细胞 |
| 3 | IG (Immature Granulocyte) | 未成熟粒细胞 |
| 4 | Lymphocyte | 淋巴细胞 |
| 5 | Monocyte | 单核细胞 |
| 6 | Neutrophil | 中性粒细胞 |
| 7 | Platelet | 血小板 |

---

## Key Results

### Ablation Study: Pretrained vs Random Initialization

| Method | Test Accuracy | AUC | Val Stability |
|--------|--------------|-----|---------------|
| Random Init | ~90% | - | Unstable (min 27%) |
| **Pretrained ResNet18** | **93.51%** | **0.9958** | **Stable** |

> Transfer learning significantly improves both performance and training stability.

### Per-class Performance (Pretrained Model)

| Cell Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Platelet | 1.00 | 1.00 | **1.00** |
| Eosinophil | 0.99 | 0.99 | **0.99** |
| Neutrophil | 0.96 | 0.97 | **0.97** |
| Erythroblast | 0.93 | 0.96 | **0.95** |
| Lymphocyte | 0.96 | 0.91 | 0.93 |
| IG | 0.84 | 0.91 | 0.87 |
| Basophil | 0.97 | 0.80 | 0.87 |
| Monocyte | 0.83 | 0.80 | 0.81 |

**Biological Insight:** Monocyte and IG are the hardest to classify — consistent with pathology practice where these cell types are known to share morphological similarities.

---

## Visualizations

### Blood Cell Samples
![samples](results/blood_cell_samples.png)

### Training Curves Comparison
**Random Initialization** — unstable validation accuracy:

![random](results/training_curves_random.png)

**Pretrained ResNet18** — stable and consistent improvement:

![pretrained](results/training_curves_pretrained.png)

### Confusion Matrix
![confusion](results/confusion_matrix.png)

---

## Model Architecture

```
ResNet18 (Pretrained on ImageNet)
    ↓
Replace final FC layer: 1000 → 8 classes
    ↓
Fine-tune on BloodMNIST
```

**Why ResNet18?**
- Lightweight but powerful (11M parameters)
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Transfer learning enables fast convergence on medical images

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | ResNet18 |
| Pretrained | ImageNet weights |
| Optimizer | Adam (lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 10 |
| LR Scheduler | StepLR (step=5, gamma=0.5) |
| Device | CPU |

---

## Pipeline

```
BloodMNIST Data (.npz)
        ↓
01: Data Loading & Exploration
        ↓
02: Preprocessing (Normalize + Transpose + DataLoader)
        ↓
03: Model Setup (ResNet18 + Transfer Learning)
        ↓
04: Training (Random Init) ← Ablation
        ↓
05: Training (Pretrained)  ← Main experiment
        ↓
06: Evaluation (Accuracy + AUC + Confusion Matrix)
```

---

## Requirements

```python
torch
torchvision
numpy
matplotlib
scikit-learn
```

---

## Biological Interpretation

The confusion matrix reveals clinically meaningful patterns:

- **Platelet** achieves perfect classification — its unique small size and lack of nucleus make it easily distinguishable
- **Monocyte vs IG** confusion (19% misclassification) reflects genuine morphological overlap — a known challenge even for experienced pathologists
- **Basophil → Monocyte** confusion aligns with their similar nuclear staining patterns

This suggests the model has learned biologically meaningful features rather than superficial patterns.

---

## Author

**Xin Liu**
Molecular Biology Researcher | Bioinformatics | Medical AI
GitHub: [xinliu756](https://github.com/xinliu756)

---

## Reference

Yang, J., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*, 2023.
