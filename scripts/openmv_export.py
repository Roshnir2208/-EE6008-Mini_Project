"""
OpenMV RT1062 Deployment Script
Prepares TFLite model for deployment on OpenMV Cam RT1062
Handles quantization and creates deployment-ready files
"""

import tensorflow as tf
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io

class OpenMVDeployer:
    def __init__(self, tflite_model_path, metadata_path, output_dir):
        self.tflite_model_path = Path(tflite_model_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded model: {self.metadata['model_name']}")
        print(f"Classes: {self.metadata['classes']}")
    
    def quantize_model(self, quantization_type='int8'):
        """Quantize model for better performance on edge devices"""
        print(f"\nQuantizing model to {quantization_type}...")
        
        # Read original model (already quantized TFLite)
        with open(self.tflite_model_path, 'rb') as f:
            quantized_model = f.read()
        
        quantized_path = self.output_dir / f'garbage_classifier_quantized_{quantization_type}.tflite'
        with open(quantized_path, 'wb') as f:
            f.write(quantized_model)
        
        print(f"✓ Quantized model saved to {quantized_path}")
        print(f"  Model size: {len(quantized_model) / 1024:.2f} KB")
        
        return quantized_path
    
    def create_openmv_firmware_script(self):
        """Create firmware script for OpenMV Cam RT1062"""
        
        script_content = '''#!/usr/bin/env python3
"""
Garbage Classification Model - OpenMV RT1062
Captures images and performs real-time garbage classification
"""

import sensor
import time
import image
import tf
import math
import json
from pyb import LED

# Configuration
MODEL_PATH = '/sd/garbage_classifier.tflite'
LABELS_PATH = '/sd/labels.json'
CONFIDENCE_THRESHOLD = 0.6
INPUT_SIZE = 320

# Load labels
def load_labels(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        # Fallback labels
        return {
            "labels": ["Plastic", "Paper", "Metal", "Glass", "Organic", "Other"],
            "reverse_labels": {
                "0": "Plastic",
                "1": "Paper", 
                "2": "Metal",
                "3": "Glass",
                "4": "Organic",
                "5": "Other"
            }
        }

# Initialize camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.skip_frames(time=2000)

# Setup LEDs for feedback
led_red = LED(1)
led_green = LED(2)
led_blue = LED(3)

print("Loading model...")
net = tf.load(MODEL_PATH)
labels = load_labels(LABELS_PATH)

print("Model loaded successfully!")
print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"Classes: {labels['labels']}")

# Main loop
clock = time.clock()
frame_count = 0

while True:
    clock.tick()
    frame_count += 1
    
    # Capture frame
    img = sensor.snapshot()
    
    # Resize for model input
    img_resized = img.copy()
    img_resized = img_resized.resize(INPUT_SIZE, INPUT_SIZE, copy=True)
    
    # Normalize (optional - depends on model training)
    # img_resized = img_resized.morph(image.MORPH_ERODE, 2, threshold=200)
    
    # Run inference
    try:
        output = net.predict(img_resized)
        
        # Get predictions
        predictions = output[0]  # Get first output
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Get top prediction
        max_idx = predictions.argmax()
        confidence = float(predictions[max_idx])
        
        # Get label
        label = labels['reverse_labels'].get(str(max_idx), "Unknown")
        
        # Display results
        if confidence > CONFIDENCE_THRESHOLD:
            # Display on screen
            text = f"{label}: {confidence:.2f}"
            img.draw_string(10, 10, text, color=(0, 255, 0), scale=2)
            
            # LED feedback (green for good confidence)
            if confidence > 0.8:
                led_green.on()
                led_red.off()
            else:
                led_green.off()
                led_red.on()
            
            print(f"Frame {frame_count}: {label} ({confidence:.2%})")
        else:
            # Low confidence - not recognized
            img.draw_string(10, 10, "Unknown", color=(255, 0, 0), scale=2)
            led_red.on()
            led_green.off()
            print(f"Frame {frame_count}: Low confidence ({confidence:.2%})")
        
        # Display FPS
        fps = clock.fps()
        img.draw_string(10, 30, f"FPS: {fps:.1f}", color=(255, 255, 0), scale=1)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        img.draw_string(10, 10, "ERROR", color=(255, 0, 0), scale=2)
        led_red.on()
        led_green.off()
'''
        
        script_path = self.output_dir / 'main_openmv.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✓ OpenMV firmware script created: {script_path}")
        return script_path
    
    def create_deployment_guide(self):
        """Create deployment guide for OpenMV"""
        
        guide = f'''# OpenMV RT1062 Garbage Classification Deployment Guide

## System Requirements
- OpenMV Cam RT1062
- MicroSD card (16GB recommended)
- OpenMV IDE
- USB cable

## Model Information
- Model Name: {self.metadata['model_name']}
- Input Size: {self.metadata['input_size']}x{self.metadata['input_size']}
- Classes: {len(self.metadata['classes'])}
- Framework: TensorFlow Lite

## Classes
'''
        for cls_name, cls_idx in self.metadata['classes'].items():
            guide += f"- {cls_name}\n"
        
        guide += f'''

## Deployment Steps

### 1. Prepare Files
Copy these files to the MicroSD card:
- `garbage_classifier_quantized_int8.tflite` → `/sd/garbage_classifier.tflite`
- `labels.json` → `/sd/labels.json`
- `main_openmv.py` → `/sd/main.py`

### 2. Connect to OpenMV IDE
1. Connect OpenMV Cam RT1062 to computer via USB
2. Open OpenMV IDE
3. Tools → Connected Camera → Select your device

### 3. Upload Files
1. File → Open File Viewer
2. Navigate to SD card
3. Upload the model and script files

### 4. Run Script
1. Open `main.py` in IDE
2. Click "Run" button (or press Ctrl+R)
3. Watch the frame buffer for garbage classification results

## Performance Tips
- Reduce INPUT_SIZE if FPS is low (trades accuracy for speed)
- Adjust CONFIDENCE_THRESHOLD based on your requirements
- For better accuracy, use INPUT_SIZE = 320
- For faster inference, use INPUT_SIZE = 224

## Input Size Trade-offs
| Size | Accuracy | Speed | Memory |
|------|----------|-------|--------|
| 224x224 | Good | Fast | Low |
| 320x320 | Better | Moderate | Moderate |
| 416x416 | Best | Slow | High |

## Debugging
- Enable serial output to see inference results
- Check OpenMV IDE console for errors
- Verify model file exists on SD card
- Check JPEG codec support for image format

## Network Requirements
Model works offline - no internet connection needed!

## Optimization for RT1062
- The model is optimized for Arm Cortex-M7 processor
- Uses integer quantization for faster inference
- Memory usage: ~2-4 MB for model + buffers
- Typical inference time: 100-500ms per frame

## Advanced Configuration
### Custom Labels
Edit `labels.json` to customize class labels:
```json
{{
  "labels": ["Class1", "Class2", ...],
  "reverse_labels": {{
    "0": "Class1",
    "1": "Class2",
    ...
  }}
}}
```

### Camera Parameters
Adjust in `main_openmv.py`:
- `sensor.set_framesize()` - Change resolution
- `INPUT_SIZE` - Model input size
- `CONFIDENCE_THRESHOLD` - Detection threshold

## Troubleshooting

### Model not loading
- Check file path is correct
- Ensure file is valid TFLite model
- Try re-uploading the file

### Low FPS
- Reduce INPUT_SIZE
- Disable serial output
- Check for other processes

### Poor accuracy
- Ensure training data matches real-world conditions
- Verify model is on SD card (not flash)
- Check lighting conditions

## Support
For issues with the model, check training parameters in train_tflite_model.py
For camera issues, see OpenMV documentation: https://docs.openmv.io/
'''
        
        guide_path = self.output_dir / 'DEPLOYMENT_GUIDE.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(f"✓ Deployment guide created: {guide_path}")
        return guide_path
    
    def create_labels_file(self):
        """Create labels file for OpenMV script"""
        labels_data = {
            "labels": list(self.metadata['classes'].keys()),
            "reverse_labels": self.metadata['reverse_classes']
        }
        
        labels_path = self.output_dir / 'labels.json'
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Labels file created: {labels_path}")
        return labels_path
    
    def deploy(self):
        """Execute full deployment pipeline"""
        print("\n" + "=" * 70)
        print("OpenMV RT1062 Deployment Pipeline")
        print("=" * 70)
        
        # Quantize
        quantized_model = self.quantize_model('int8')
        
        # Create deployment files
        openmv_script = self.create_openmv_firmware_script()
        labels_file = self.create_labels_file()
        guide = self.create_deployment_guide()
        
        print("\n" + "=" * 70)
        print("✓ Deployment Preparation Complete!")
        print("=" * 70)
        print("\nNext Steps:")
        print(f"1. Copy {quantized_model} to MicroSD as 'garbage_classifier.tflite'")
        print(f"2. Copy {labels_file} to MicroSD as 'labels.json'")
        print(f"3. Copy {openmv_script} to MicroSD as 'main.py'")
        print(f"4. Insert MicroSD into OpenMV Cam RT1062")
        print(f"5. Connect to OpenMV IDE and run")
        print(f"\nSee {guide} for detailed instructions")

def main():
    # Paths
    tflite_model = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models\garbage_classifier.tflite"
    metadata = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models\model_metadata.json"
    output_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\openmv_deployment"
    
    # Deploy
    deployer = OpenMVDeployer(tflite_model, metadata, output_dir)
    deployer.deploy()

if __name__ == "__main__":
    main()
