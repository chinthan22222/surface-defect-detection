
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from collections import defaultdict, Counter
import logging
import timm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model from main.py
sys.path.append('.')
from main import EfficientNetScrewClassifier

class ComprehensiveTestDataset(Dataset):
    """Dataset for comprehensive testing"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Class mapping
        self.class_to_idx = {
            'good': 0,
            'thread_top': 1,
            'scratch_neck': 2,
            'manipulated_front': 3,
            'scratch_head': 4,
            'thread_side': 5
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all test samples"""
        test_dir = self.data_dir / 'test'
        
        if not test_dir.exists():
            raise ValueError(f"Test directory {test_dir} does not exist")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), class_idx, class_name))
        
        print(f"📊 Loaded {len(self.samples)} test samples")
        
        # Print class distribution
        class_counts = Counter([sample[1] for sample in self.samples])
        print("📈 Test Dataset Distribution:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = self.idx_to_class[class_idx]
            print(f"   {class_name:20}: {count:3d} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx, img_path

def comprehensive_validation():
    """Run comprehensive validation on entire test set"""
    
    print("🎯 COMPREHENSIVE MODEL VALIDATION")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create model
    print("\n1. 🧠 Loading Model...")
    model = EfficientNetScrewClassifier(num_classes=6)
    
    # Load trained weights
    model_path = Path('models/elite_model.pth')
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None
    
    # Create dataset and dataloader
    print("\n2. 📁 Loading Test Dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ComprehensiveTestDataset('dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Run evaluation
    print("\n3. 🔍 Running Comprehensive Evaluation...")
    
    all_predictions = []
    all_targets = []
    all_paths = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for batch_idx, (images, targets, paths) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_paths.extend(paths)
            
            # Per-class accuracy
            for i in range(len(targets)):
                class_idx = targets[i].item()
                class_total[class_idx] += 1
                if predicted[i] == targets[i]:
                    class_correct[class_idx] += 1
            
            if batch_idx % 10 == 0:
                print(f"   Processed {(batch_idx + 1) * 32}/{len(dataset)} images...")
    
    print(f"   ✅ Evaluation completed on {len(all_predictions)} images")
    
    # Calculate metrics
    print("\n4. 📊 Computing Final Metrics...")
    
    # Overall accuracy
    overall_accuracy = accuracy_score(all_targets, all_predictions)
    
    # Per-class accuracy
    class_names = ['good', 'thread_top', 'scratch_neck', 'manipulated_front', 'scratch_head', 'thread_side']
    per_class_acc = {}
    
    for class_idx in range(6):
        if class_total[class_idx] > 0:
            acc = class_correct[class_idx] / class_total[class_idx]
            per_class_acc[class_names[class_idx]] = acc
        else:
            per_class_acc[class_names[class_idx]] = 0.0
    
    # Classification report
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=class_names, 
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("🏆 FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"📈 Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print()
    
    print("📊 Per-Class Accuracy:")
    for class_name, acc in per_class_acc.items():
        total_samples = class_total[class_names.index(class_name)]
        correct_samples = class_correct[class_names.index(class_name)]
        print(f"   {class_name:20}: {acc:.4f} ({acc*100:.1f}%) - {correct_samples}/{total_samples}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    print("🔢 Confusion Matrix:")
    print("Predicted →")
    print("Actual ↓  ", end="")
    for name in class_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name[:8]:>8}  ", end="")
        for j in range(6):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    # Save results
    results = {
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_predictions),
        'class_distribution': {class_names[i]: int(class_total[i]) for i in range(6)}
    }
    
    with open('comprehensive_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: comprehensive_validation_results.json")
    
    return results

if __name__ == "__main__":
    results = comprehensive_validation()
    
    if results:
        print(f"\n🎉 Validation completed successfully!")
        print(f"🎯 Final Project Accuracy: {results['overall_accuracy']*100:.2f}%")
    else:
        print("\n❌ Validation failed!")
