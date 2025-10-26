# Multi-class Chest X‑Ray Classification (Normal · Pneumonia · Tuberculosis)

A classical machine-learning baseline for classifying chest X‑ray images into three classes: normal, pneumonia, and tuberculosis. The pipeline uses HOG features with standardized training and evaluates on a held-out test set.

## Overview

- Task: 3-class image classification from chest X‑rays.
- Approach: Histogram of Oriented Gradients (HOG) features + classical classifiers (baseline results shown for Logistic Regression).
- Implementation: Jupyter notebook (`model.ipynb`).
- Data layout: class-per-folder convention under `train/` and `test/`.

## Dataset

- Classes: `normal`, `pneumonia`, `tuberculosis`
- Format: RGB `.jpg` images, labels inferred from folder names.
- Directory structure:
  - `train/normal`, `train/pneumonia`, `train/tuberculosis`
  - `test/normal`, `test/pneumonia`, `test/tuberculosis`
- Test set counts (from evaluation support):
  - normal: 763
  - pneumonia: 545
  - tuberculosis: 943
  - Total: 2251

## Methods

- Preprocessing:
  - Load images, enforce RGB
  - Grayscale conversion for feature extraction
  - Resize to a fixed square size
  - Standardize features via z-score using train-set statistics
- Features: HOG (edge/orientation structure)
- Models: Classical baselines (e.g., Logistic Regression, KNN, SVM-RBF, Naive Bayes, Decision Tree, Random Forest, AdaBoost, MLP). Reported metrics below are for Logistic Regression.
- Evaluation: Accuracy, per-class Precision/Recall/F1, macro and weighted averages on the held-out test set.

## Getting Started

### Requirements

- Python 3.9+ (Windows)
- Recommended packages: numpy, pandas, scikit-learn, scikit-image, matplotlib, seaborn, jupyter
- Optional (if enabled in the notebook): xgboost, pgmpy

### Setup (Windows PowerShell)

````powershell
# From the repository root
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn jupyter
# Optional extras if your notebook uses them:
# pip install xgboost pgmpy

# Launch VS Code or Jupyter
code .
# or
jupyter notebook
