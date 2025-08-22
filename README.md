# 🤖 Surface Defect Detection System

**AI-Powered Industrial Quality Control for Screw Manufacturing**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.75%25-green.svg)](#performance)

## 🎯 Overview

A production-ready deep learning system for automated defect detection in screw manufacturing, achieving **93.75% accuracy** using EfficientNet-B3 architecture. Perfect for industrial quality control and technical demonstrations.

## ✨ Key Features

- **🔍 High Accuracy:** 93.75% validated performance on 160 test images
- **⚡ Fast Inference:** Sub-second processing for real-time quality control
- **🧠 Advanced AI:** EfficientNet-B3 with transfer learning
- **🎯 Multi-Class:** Detects 6 types of defects automatically
- **🖥️ Console Interface:** Clean, production-ready command-line tool
- **📊 Comprehensive Validation:** Full metrics and confusion matrix analysis

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/chinthan22222/surface-defect-detection.git
cd surface-defect-detection
pip install -r requirements.txt
```

### 30-Second Demo
```bash
python demo.py
```

### Test Individual Images
```bash
python main.py test_images/screw_good_001.png
python main.py test_images/screw_defect_002.png
```

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 93.75% |
| **Processing Speed** | <0.3s per image |
| **Classes Supported** | 6 defect types |
| **Test Dataset** | 160 images |

### Per-Class Performance
- **good**: 95.1% accuracy
- **scratch_neck**: 100% accuracy  
- **manipulated_front**: 100% accuracy
- **scratch_head**: 95.8% accuracy
- **thread_top**: 87.0% accuracy
- **thread_side**: 82.6% accuracy

## 🛠️ Technical Stack

- **Framework:** PyTorch + timm
- **Architecture:** EfficientNet-B3
- **Dataset:** MVTec Anomaly Detection (Screws)
- **Training:** Transfer learning with fine-tuning
- **Deployment:** Console-based Python application

## 📁 Project Structure

```
surface_defect_detection/
├── demo.py              # Quick demonstration
├── main.py              # Main detection system  
├── train.py             # Training pipeline
├── models/              # Trained model weights
├── dataset/             # Training & test data
└── test_images/         # Sample images for demo
```

## 💡 Use Cases

- **Manufacturing Quality Control:** Automated defect detection in production lines
- **Technical Interviews:** Showcase AI/ML expertise with working demonstration
- **Research & Development:** Foundation for industrial computer vision projects

## 🎤 Interview Ready

```bash
# Professional demo in 30 seconds
python demo.py

# Test with custom images  
python main.py your_image.jpg

# Show technical validation
python comprehensive_validation.py
```

## 📈 Results

The system successfully identifies:
- ✅ Good quality screws
- ❌ Scratch defects (head/neck)
- ❌ Thread manufacturing issues
- ❌ Manipulated/damaged parts

With industry-grade accuracy suitable for production deployment.

## 👨‍💻 Author

**Chinthan** - AI/ML Engineer specializing in computer vision and industrial automation.

---

⭐ **Star this repository if it helped you!** ⭐