

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import time
import timm
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

class ScrewDataset(Dataset):
    """Dataset class for screw defect detection"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Define class mapping for screw dataset
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
        """Load all samples from the dataset directory"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Loaded {len(self.samples)} {self.split} samples")
        
        # Print class distribution
        class_counts = defaultdict(int)
        for _, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            class_counts[class_name] += 1
        
        logger.info(f"{self.split.title()} class distribution:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx

class EfficientNetScrewClassifier(nn.Module):
    """
    EfficientNet-based screw defect classifier
    Matches the architecture used for 97.5% accuracy model
    """
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetScrewClassifier, self).__init__()
        
        # Use timm for exact architecture match
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        
        # Custom classifier to match trained model structure
        self.classifier = nn.Linear(1536, num_classes)  # EfficientNet B3 feature size: 1536
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_transforms(split='train'):
    """Get data transforms for training and validation"""
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_targets

def save_results(results, output_dir, class_names):
    """Save training results and visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(results['train_accuracies'], label='Train Acc')
    plt.plot(results['val_accuracies'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(results['final_targets'], results['final_predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(results['final_targets'], results['final_predictions'], 
                                 target_names=class_names, output_dict=True)
    
    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Surface Defect Detection Model')
    parser.add_argument('--data_dir', type=str, default='dataset', 
                       help='Dataset directory (default: dataset)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output_dir', type=str, default='training_output', 
                       help='Output directory (default: training_output)')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = ScrewDataset(args.data_dir, transform=train_transform, split='train')
    val_dataset = ScrewDataset(args.data_dir, transform=val_transform, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Create model
    num_classes = len(train_dataset.class_to_idx)
    class_names = list(train_dataset.class_to_idx.keys())
    
    model = EfficientNetScrewClassifier(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    logger.info(f"Model created with {num_classes} classes: {class_names}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    results = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'final_predictions': [],
        'final_targets': []
    }
    
    best_val_acc = 0.0
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Save results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['train_accuracies'].append(train_acc)
        results['val_accuracies'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            results['final_predictions'] = val_preds
            results['final_targets'] = val_targets
            
            # Save model
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                   f"Time: {epoch_time:.2f}s")
    
    # Final evaluation
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Save results
    save_results(results, args.output_dir, class_names)
    
    # Print final classification report
    if results['final_predictions']:
        report = classification_report(results['final_targets'], results['final_predictions'], 
                                     target_names=class_names)
        logger.info(f"\nFinal Classification Report:\n{report}")

if __name__ == "__main__":
    main()
