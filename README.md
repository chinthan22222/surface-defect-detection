# Surface Defect Detection System

AI-powered industrial quality control for screw manufacturing using EfficientNet-B3 architecture.

## Performance
- **Accuracy**: 93.75% on test dataset
- **Speed**: <0.3 seconds per image
- **Classes**: 6 defect types

## Quick Start

```bash
git clone https://github.com/chinthan22222/surface-defect-detection.git
cd surface-defect-detection
pip install -r requirements.txt
python demo.py
```

## Usage

```bash
# Demo with sample images
python demo.py

# Single image detection
python main.py test_images/screw_good_001.png

# Batch processing
python main.py test_images/

# Full validation
python comprehensive_validation.py
```

## Requirements
- Python 3.8+
- PyTorch
- timm
- PIL
- NumPy

## Architecture
- **Model**: EfficientNet-B3
- **Framework**: PyTorch
- **Dataset**: MVTec Anomaly Detection (Screws)

## Classes
- good
- scratch_head
- scratch_neck
- thread_top
- thread_side
- manipulated_front