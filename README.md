# hitl-cnn-app

A Human-in-the-Loop Convolutional Neural Network (HITL-CNN) system for crop
disease diagnosis in precision agriculture. Combines an EfficientNet-B4
backbone with Monte Carlo Dropout uncertainty quantification, Grad-CAM
explainability, and a confidence-threshold-based expert referral mechanism.

Live demo: hitl-crop-diagnosis.streamlit.app
Trained model: huggingface.co/Initreasure/hitl-cnn-model

## Requirements

- Python 3.10+
- pip

## Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd hitl-cnn-app
pip install -r requirements.txt
```

requirements.txt:streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scipy>=1.9.0


## Dataset

This project uses the PlantVillage dataset (Hughes and Salathé, 2015),
licensed under CC BY-NC-SA 4.0. Download it from Kaggle:

```bash
# kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

## Usage

### 1. Split the dataset

```bash
python split_plantvillage.py --src /path/to/plantvillage/color --dst ./plantvillage_split
```

Produces `train/`, `val/`, `test/` folders (70/15/15 stratified split) and
a `split_manifest.csv` recording every image's assigned split.

### 2. Train the main model

```bash
python train_hitl_cnn.py --data_dir ./plantvillage_split --epochs 30 --device cuda --batch_size 32
```

Two-stage fine-tuning: Stage 1 trains the classification head only (2
epochs); Stage 2 fine-tunes the full network with early stopping.

### 3. Run inference with HITL referral

```bash
python hitl_inference.py --checkpoint best_model.pt --image path/to/leaf.jpg
```

### 4. Reproduce the baseline comparison (optional)

```bash
python train_baseline.py --arch resnet50 --data_dir ./plantvillage_split --out_dir ./baseline_resnet50
```

Note: a VGG-16 baseline was also attempted (`--arch vgg16`) but found
computationally infeasible on standard consumer GPU hardware (approximately
3 hours/epoch); this is documented as a limitation in the dissertation
(Chapter 4, Section 4.3.3), not something this repository claims to deliver.

### 5. Run the statistical comparison

```bash
python statistical_comparison.py --main main_predictions.csv --baseline baseline_predictions.csv --baseline_name "ResNet-50"
```

## Project structure

hitl-cnn-app/
├── split_plantvillage.py # Dataset splitting
├── train_hitl_cnn.py # Main model training
├── hitl_inference.py # HITL referral inference
├── feature_engineering.py # 9-channel pilot feature engineering
├── train_hitl_cnn_9ch_onthefly.py # Full-scale 9-channel training
├── train_baseline.py # ResNet-50 / VGG-16 baselines
├── statistical_comparison.py # McNemar's test, bootstrap CIs
├── benchmark_inference.py # Inference latency benchmarking
├── app.py # Streamlit deployment app
├── requirements.txt
├── LICENSE
└── README.md


## Results

Full results and methodology are documented in the accompanying MRes
dissertation (University of Greater Manchester, MRES7015). Headline result:
99.69% test accuracy, 88.0% error-capture rate at a 2.6% referral rate,
ECE 0.0025 post-calibration.

## Licence and citation

Code: MIT License. Dataset: CC BY-NC-SA 4.0 (Hughes and Salathé, 2015).
If you use this work, please cite the accompanying dissertation.

