
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientNetScrewClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetScrewClassifier, self).__init__()
        try:
            import timm
            self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
            self.classifier = nn.Linear(1536, num_classes)
        except ImportError:
            logger.error("timm not available. Installing fallback architecture...")
            import torchvision.models as models
            backbone = models.efficientnet_b3(pretrained=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.backbone.add_module('adaptive_pool', nn.AdaptiveAvgPool2d(1))
            self.backbone.add_module('flatten', nn.Flatten())
            self.classifier = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

class SurfaceDefectDetector:

    def __init__(self, model_path="models/elite_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Define class names
        self.classes = [
            'good',
            'thread_top', 
            'scratch_neck',
            'manipulated_front',
            'scratch_head',
            'thread_side'
        ]
        
        # Initialize model
        self.model = EfficientNetScrewClassifier(num_classes=len(self.classes))
        self._load_model(model_path)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):

        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Using untrained model - predictions may be unreliable")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def predict(self, image_path):

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            # Prepare results
            results = {
                'image_path': image_path,
                'predicted_class': self.classes[predicted_class_idx],
                'confidence': confidence,
                'is_defective': predicted_class_idx != 0,  # 'good' is class 0
                'all_probabilities': {
                    self.classes[i]: probabilities[0][i].item() 
                    for i in range(len(self.classes))
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def batch_predict(self, image_dir):
        """
        Process all images in a directory
        
        Args:
            image_dir (str): Directory containing images
            
        Returns:
            list: List of prediction results
        """
        results = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        image_files = [
            f for f in os.listdir(image_dir) 
            if os.path.splitext(f.lower())[1] in supported_extensions
        ]
        
        logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        for filename in sorted(image_files):
            image_path = os.path.join(image_dir, filename)
            result = self.predict(image_path)
            if result:
                results.append(result)
                
                # Print results
                status = "🔴 DEFECTIVE" if result['is_defective'] else "🟢 GOOD"
                print(f"{filename:25} | {result['predicted_class']:15} | {result['confidence']:.3f} | {status}")
        
        return results
    
    def save_results(self, results, output_file="results.json"):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def print_statistics(results):
    """Print summary statistics"""
    if not results:
        return
    
    total = len(results)
    defective = sum(1 for r in results if r['is_defective'])
    good = total - defective
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images Processed: {total}")
    print(f"Good Parts:             {good} ({100*good/total:.1f}%)")
    print(f"Defective Parts:        {defective} ({100*defective/total:.1f}%)")
    print(f"Average Confidence:     {avg_confidence:.3f}")
    
    # Per-class breakdown
    class_counts = {}
    for result in results:
        cls = result['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"\nPer-Class Detection:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:15}: {count:3d} ({100*count/total:.1f}%)")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Surface Defect Detection System - Console Interface"
    )
    parser.add_argument(
        'input', 
        help='Input image file or directory containing images'
    )
    parser.add_argument(
        '--model', 
        default='models/elite_model.pth',
        help='Path to the trained model file (default: models/elite_model.pth)'
    )
    parser.add_argument(
        '--output', 
        default='results.json',
        help='Output file for results (default: results.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize detector
    print("Initializing Surface Defect Detection System...")
    detector = SurfaceDefectDetector(model_path=args.model)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print(f"\nProcessing single image: {args.input}")
        print(f"{'Image':25} | {'Predicted Class':15} | {'Confidence':10} | {'Status':12}")
        print("-" * 70)
        
        result = detector.predict(args.input)
        if result:
            status = "🔴 DEFECTIVE" if result['is_defective'] else "🟢 GOOD"
            filename = os.path.basename(result['image_path'])
            print(f"{filename:25} | {result['predicted_class']:15} | {result['confidence']:.3f} | {status}")
            
            # Save single result
            detector.save_results([result], args.output)
        
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"\nProcessing directory: {args.input}")
        print(f"{'Image':25} | {'Predicted Class':15} | {'Confidence':10} | {'Status':12}")
        print("-" * 70)
        
        results = detector.batch_predict(args.input)
        
        if results:
            print_statistics(results)
            detector.save_results(results, args.output)
        else:
            print("No images found or processed.")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
