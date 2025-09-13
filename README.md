# Diabetic Retinopathy Detection

This repository provides two workflows for training a deep learning model to classify diabetic retinopathy using the Kaggle [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) dataset.

---

## 📂 Repository Structure

diabetic-retinopathy/
│
├── sample.zip
├──trainLabels.csv
├── sample.ipynb # quick test using Kaggle's sample.zip
├── full_size.ipynb # full workflow using train/test images from Kaggle
├── README.md

---

## 🚀 Workflows

### 1. Sample Workflow (for testing)
- Notebook: **`sample.ipynb`**
- Requirements:
  - `sample.zip` (provided by Kaggle as part of the competition data)
  - `trainLabels.csv`
- Process:
  1. Extracts images from `sample.zip`.
  2. Filters `trainLabels.csv` to include only sample images.
  3. Trains a model on the small dataset for quick verification.

### 2. Full Workflow (for training on the complete dataset)
- Notebook: **`full_size.ipynb`**
- Requirements:
  - Kaggle competition data: `train.zip`, `test.zip`, `trainLabels.csv`
  - A valid [Kaggle API key](https://www.kaggle.com/docs/api) for automated download
- Process:
  1. Downloads and extracts the complete dataset.
  2. Preprocesses images using:
     - Green channel extraction
     - Contrast enhancement (CLAHE)
     - Resizing and normalization
  3. Trains a **ResNet50V2** model with **Focal Loss** to address class imbalance.
  4. Evaluates with **Accuracy, AUC, and Recall**.
  5. Saves the trained model as `diabetic_retinopathy_model.h5`.

---

## ⚠️ Notes
- The **sample workflow** is for quick debugging and testing only.
- The **full dataset** is ~90 GB and should be obtained directly from Kaggle.
- Do **not** commit the full dataset to this repository. Only keep small example files like `sample.zip`.

---

## 📚 References
- Kaggle Competition: [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)  
- He K. et al., *Deep Residual Learning for Image Recognition (ResNet)*, 2015  
- Lin T.-Y. et al., *Focal Loss for Dense Object Detection*, 2017  
