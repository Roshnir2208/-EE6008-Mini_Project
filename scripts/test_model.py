"""
Model Testing and Validation Script
Test the trained TFLite model before deployment to OpenMV
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class TFLiteModelTester:
    def __init__(self, tflite_model_path, metadata_path, dataset_dir):
        self.tflite_model_path = Path(tflite_model_path)
        self.metadata_path = Path(metadata_path)
        self.dataset_dir = Path(dataset_dir)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_size = self.metadata['input_size']
        self.classes = self.metadata['classes']
        self.reverse_classes = self.metadata['reverse_classes']
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path):
        """Run inference on single image"""
        # Preprocess
        image = self.preprocess_image(image_path)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output[0]
        
        # Get class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = self.reverse_classes[str(predicted_class_idx)]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions.tolist(),
            'class_scores': {self.reverse_classes[str(i)]: float(predictions[i]) 
                           for i in range(len(predictions))}
        }
    
    def load_test_data(self, test_split_dir):
        """Load test images from directory"""
        images = []
        csv_file = test_split_dir / '_annotations.csv'
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Annotations not found: {csv_file}")
        
        # Read annotations
        import csv
        image_classes = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                class_name = row['class']
                
                if filename not in image_classes:
                    image_classes[filename] = class_name
        
        # Build image list
        for filename, class_name in image_classes.items():
            image_path = test_split_dir / filename
            if image_path.exists():
                images.append((image_path, class_name))
        
        return images
    
    def evaluate_on_test_set(self, test_split_dir):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("Evaluating on Test Set")
        print("="*70)
        
        # Load test data
        test_images = self.load_test_data(test_split_dir)
        print(f"Found {len(test_images)} test images")
        
        if len(test_images) == 0:
            print("No test images found!")
            return None
        
        # Run predictions
        predictions = []
        ground_truth = []
        confidences = []
        
        for image_path, true_class in tqdm(test_images, desc="Evaluating"):
            try:
                result = self.predict(image_path)
                predictions.append(result['class'])
                ground_truth.append(true_class)
                confidences.append(result['confidence'])
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        accuracy = np.mean(predictions == ground_truth)
        mean_confidence = np.mean(confidences)
        
        print(f"\n✓ Accuracy: {accuracy:.2%}")
        print(f"✓ Mean Confidence: {mean_confidence:.2%}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(ground_truth, predictions, labels=list(self.classes.keys())))
        
        # Confusion matrix
        unique_classes = sorted(set(list(ground_truth) + list(predictions)))
        class_indices = {cls: i for i, cls in enumerate(unique_classes)}
        
        y_true = [class_indices[cls] for cls in ground_truth]
        y_pred = [class_indices[cls] for cls in predictions]
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'confusion_matrix': cm,
            'class_labels': unique_classes,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidences': confidences
        }
    
    def plot_confusion_matrix(self, eval_results, output_path):
        """Plot and save confusion matrix"""
        cm = eval_results['confusion_matrix']
        class_labels = eval_results['class_labels']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        plt.title('Confusion Matrix - Garbage Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"✓ Confusion matrix saved to {output_path}")
    
    def test_single_image(self, image_path):
        """Test on single image"""
        print(f"\nTesting image: {image_path}")
        result = self.predict(image_path)
        
        print(f"Predicted: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll predictions:")
        for cls, score in sorted(result['class_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 50)
            print(f"  {cls:15} {score:.4f} {bar}")
        
        return result
    
    def benchmark_speed(self, test_split_dir, num_images=100):
        """Benchmark inference speed"""
        import time
        
        print("\n" + "="*70)
        print("Benchmarking Inference Speed")
        print("="*70)
        
        # Load test images
        test_images = self.load_test_data(test_split_dir)
        test_images = test_images[:num_images]
        
        if len(test_images) == 0:
            print("No test images found!")
            return
        
        times = []
        
        for image_path, _ in tqdm(test_images, desc="Benchmarking"):
            start = time.time()
            try:
                self.predict(image_path)
                elapsed = time.time() - start
                times.append(elapsed)
            except:
                continue
        
        times = np.array(times)
        
        print(f"\nInference Speed (on {len(times)} images):")
        print(f"  Min: {times.min()*1000:.2f} ms")
        print(f"  Max: {times.max()*1000:.2f} ms")
        print(f"  Mean: {times.mean()*1000:.2f} ms")
        print(f"  Median: {np.median(times)*1000:.2f} ms")
        print(f"  Std Dev: {times.std()*1000:.2f} ms")
        print(f"\nFPS: {1/times.mean():.1f} FPS (estimated)")

def main():
    # Paths
    tflite_model = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models\garbage_classifier.tflite"
    metadata = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models\model_metadata.json"
    dataset_root = r"C:\Users\shall\Mini_Proj\Garbage Classifier.v27i.tensorflow"
    test_dir = Path(dataset_root) / "test"
    
    print("="*70)
    print("TFLite Model Testing & Validation")
    print("="*70)
    
    # Create tester
    tester = TFLiteModelTester(tflite_model, metadata, dataset_root)
    
    # Menu
    print("\nOptions:")
    print("[1] Evaluate on full test set")
    print("[2] Test single image")
    print("[3] Benchmark inference speed")
    print("[4] All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        eval_results = tester.evaluate_on_test_set(test_dir)
        if eval_results:
            output_path = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\confusion_matrix.png"
            tester.plot_confusion_matrix(eval_results, output_path)
    
    if choice in ['2', '4']:
        test_image = input("Enter path to test image: ").strip()
        if Path(test_image).exists():
            tester.test_single_image(test_image)
        else:
            print("Image not found!")
    
    if choice in ['3', '4']:
        tester.benchmark_speed(test_dir, num_images=50)
    
    print("\n" + "="*70)
    print("✓ Testing Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
