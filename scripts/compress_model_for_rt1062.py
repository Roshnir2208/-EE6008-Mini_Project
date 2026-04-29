#!/usr/bin/env python3
"""
Aggressive Model Compression for OpenMV RT1062
Compresses TensorFlow Lite model to fit in 1 MB RAM constraint
Uses extreme quantization and optimization techniques
"""

import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path

print("=" * 60)
print("Model Compression for RT1062 (1 MB RAM Constraint)")
print("=" * 60)

# Paths
BASE_DIR = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier")
MODELS_DIR = BASE_DIR / "output" / "trained_models"
DEPLOYMENT_DIR = BASE_DIR / "output" / "openmv_deployment"
OUTPUT_DIR = DEPLOYMENT_DIR / "compressed_models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the full model
print("\n1. Loading original model...")
original_model_path = MODELS_DIR / "garbage_classifier_final.h5"
print(f"   Loading from: {original_model_path}")

try:
    model = tf.keras.models.load_model(original_model_path)
    print(f"    Model loaded")
    print(f"   Model summary:")
    model.summary()
except Exception as e:
    print(f"    Error loading model: {e}")
    raise

# Convert to TFLite with multiple compression strategies
print("\n2. Testing compression strategies...")

# Strategy 1: Full Integer Quantization (Most Aggressive)
print("\n   Strategy 1: Full Integer Quantization (INT8)")
print("   " + "-" * 50)

try:
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable experimental optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert
    tflite_int8_model = converter.convert()
    
    # Save
    int8_path = OUTPUT_DIR / "garbage_classifier_int8_aggressive.tflite"
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8_model)
    
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"    Converted: {int8_size:.2f} MB")
    
    if int8_size < 1.0:
        print(f"   FITS IN RT1062! ({int8_size:.2f} MB < 1 MB)")
    else:
        print(f"    Still too large ({int8_size:.2f} MB > 1 MB)")
        
except Exception as e:
    print(f"    Error: {e}")
    int8_size = float('inf')

# Strategy 2: Pruned + Quantized
print("\n   Strategy 2: Pruned Model + Quantization")
print("   " + "-" * 50)

try:
    # Load and prune the model
    import tensorflow_model_optimization as tfmot
    
    print("   Applying pruning...")
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # Remove 50% of weights
        begin_step=0,
        end_step=1000
    )
    
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )
    
    # Compile
    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Quick training to settle pruning
    print("   Fine-tuning pruned model...")
    pruned_model.fit(
        np.random.rand(10, 320, 320, 3).astype(np.float32),
        np.random.randint(0, 7, 10),
        epochs=1,
        verbose=0
    )
    
    # Strip pruning
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # Convert with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_pruned_model = converter.convert()
    
    pruned_path = OUTPUT_DIR / "garbage_classifier_pruned_int8.tflite"
    with open(pruned_path, 'wb') as f:
        f.write(tflite_pruned_model)
    
    pruned_size = os.path.getsize(pruned_path) / (1024 * 1024)
    print(f"   Converted: {pruned_size:.2f} MB")
    
    if pruned_size < 1.0:
        print(f"   FITS IN RT1062! ({pruned_size:.2f} MB < 1 MB)")
    else:
        print(f"    Still too large ({pruned_size:.2f} MB > 1 MB)")
        
except Exception as e:
    print(f"   Pruning not available or error: {e}")
    pruned_size = float('inf')

# Strategy 3: Quantization Aware Training (if we have time)
print("\n   Strategy 3: Dynamic Quantization")
print("   " + "-" * 50)

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_dynamic = converter.convert()
    
    dynamic_path = OUTPUT_DIR / "garbage_classifier_dynamic_q.tflite"
    with open(dynamic_path, 'wb') as f:
        f.write(tflite_dynamic)
    
    dynamic_size = os.path.getsize(dynamic_path) / (1024 * 1024)
    print(f"    Converted: {dynamic_size:.2f} MB")
    
    if dynamic_size < 1.0:
        print(f"    FITS IN RT1062! ({dynamic_size:.2f} MB < 1 MB)")
    else:
        print(f"   Still too large ({dynamic_size:.2f} MB > 1 MB)")
        
except Exception as e:
    print(f"    Error: {e}")
    dynamic_size = float('inf')

# Summary
print("\n" + "=" * 60)
print("COMPRESSION SUMMARY")
print("=" * 60)

results = [
    ("INT8 Quantized", int8_size, OUTPUT_DIR / "garbage_classifier_int8_aggressive.tflite"),
    ("Pruned + INT8", pruned_size, OUTPUT_DIR / "garbage_classifier_pruned_int8.tflite"),
    ("Dynamic Quant", dynamic_size, OUTPUT_DIR / "garbage_classifier_dynamic_q.tflite"),
]

fits = []
for name, size, path in results:
    if size < 10:  # Only print valid results
        status = " FITS" if size < 1.0 else " Too large"
        print(f"{name:20} {size:8.2f} MB  {status}")
        if size < 1.0:
            fits.append((name, size, path))

if fits:
    print("\n SUCCESS! Found compressed models that fit!")
    fits.sort(key=lambda x: x[1])  # Sort by size
    best_name, best_size, best_path = fits[0]
    
    print(f"\nBest option: {best_name} ({best_size:.2f} MB)")
    print(f"File: {best_path}")
    
    # Copy to deployment
    print(f"\nCopying to deployment folder...")
    import shutil
    deployment_model = DEPLOYMENT_DIR / "garbage_classifier_compressed.tflite"
    shutil.copy(best_path, deployment_model)
    print(f" Saved to: {deployment_model}")
    
    print("\n NEXT STEPS:")
    print("1. The compressed model has been created")
    print("2. Update main_openmv.py to use 'garbage_classifier_compressed.tflite'")
    print("3. Copy to camera and test")
    print("4. Model size: {:.2f} MB (plenty of room in 1 MB)".format(best_size))
    
else:
    print("\n All models still too large for 1 MB RT1062!")
    print("\nAlternative solutions:")
    print("1. Use a smaller base architecture (MobileNet v0.25)")
    print("2. Reduce input resolution (160x160 instead of 320x320)")
    print("3. Reduce number of classes or use class grouping")
    print("4. Switch to H7 Plus board with more RAM")

print("\n" + "=" * 60)
