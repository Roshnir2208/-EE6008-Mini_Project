# ============================================================================
# TensorFlow GPU Environment Setup Script for Garbage Classifier
# ============================================================================
# This script sets up the complete environment for training a TensorFlow Lite
# model for garbage classification with GPU support

Write-Host "========================================" -ForegroundColor Green
Write-Host "TF Lite GPU Setup for Garbage Classifier" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Activate virtual environment
$venv_path = "C:\Users\shall\Mini_Proj\tfod_env"
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& "$venv_path\Scripts\Activate.ps1"

# Step 1: Upgrade pip
Write-Host "`n[Step 1] Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Step 2: Install NVIDIA GPU support libraries
Write-Host "`n[Step 2] Installing NVIDIA CUDA/cuDNN support packages..." -ForegroundColor Yellow
pip install nvidia-cudnn-cu11==8.6.0
pip install nvidia-cuda-runtime-cu11==11.8.89
pip install nvidia-cuda-toolkit==11.8

# Step 3: Install TensorFlow with GPU support
Write-Host "`n[Step 3] Installing TensorFlow with GPU support (this may take a few minutes)..." -ForegroundColor Yellow
pip install tensorflow[and-cuda]==2.13.0

# Step 4: Install additional dependencies
Write-Host "`n[Step 4] Installing additional dependencies..." -ForegroundColor Yellow
pip install -r "$PSScriptRoot\..\garbage_classifier\requirements.txt"

# Step 5: Verify GPU setup
Write-Host "`n[Step 5] Verifying GPU setup..." -ForegroundColor Yellow
python -c "
import tensorflow as tf
print('TensorFlow Version:', tf.__version__)
print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)
print('GPUs detected:', len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.list_physical_devices('GPU'):
    print('  -', gpu)
"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: python prepare_dataset.py" -ForegroundColor White
Write-Host "2. Run: python train_tflite_model.py" -ForegroundColor White
Write-Host "3. Deploy to OpenMV using: openmv_export.py" -ForegroundColor White
