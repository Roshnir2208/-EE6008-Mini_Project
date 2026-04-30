import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Silence the oneDNN and INFO logs 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Setup Model
model = MobileNetV2(weights='imagenet')

# 2. Mapping Categories - Added common Roboflow/Waste keywords
CATEGORY_MAP = {
    'Plastic': ['plastic', 'bottle', 'poly', 'pet', 'pvc'],
    'Paper': ['paper', 'newspaper', 'envelope', 'notebook'],
    'Cardboard': ['cardboard', 'carton', 'box', 'board'],
    'Glass': ['glass', 'bottle', 'jar', 'cup', 'beaker'],
    'Metal': ['metal', 'tin', 'can', 'aluminium', 'aluminum', 'iron', 'steel'],
    'Trash': ['trash', 'garbage', 'waste', 'refuse']
}

def get_category_from_filename(filename):
    """Trust the existing filename label (like 'metal265_jpg.rf...')"""
    fn_lower = filename.lower()
    for cat in CATEGORY_MAP.keys():
        # Check if the filename starts with or contains the category name
        if cat.lower() in fn_lower:
            return cat
    return None

def get_category_from_ai(imagenet_label):
    """Fallback to AI if the filename is generic (e.g., 'IMG_001.jpg')"""
    label_lower = imagenet_label.lower()
    for cat, keywords in CATEGORY_MAP.items():
        if any(key in label_lower for key in keywords):
            return cat
    return "Trash"

def organize_dataset(src_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    # Create subfolders
    for cat in CATEGORY_MAP.keys():
        os.makedirs(os.path.join(dest_path, cat), exist_ok=True)

    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    counters = {cat: 1 for cat in CATEGORY_MAP.keys()}

    print(f"Sorting {len(files)} images. Prioritising existing labels...")

    for filename in files:
        file_path = os.path.join(src_path, filename)
        
        # STEP 1: Trust your manual labels first
        category = get_category_from_filename(filename)
        source_type = "Filename"

        # STEP 2: Use AI only if the filename provides no clue
        if category is None:
            try:
                img = image.load_img(file_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = model.predict(x, verbose=0)
                _, label, _ = decode_predictions(preds, top=1)[0][0]
                category = get_category_from_ai(label)
                source_type = f"AI ({label})"
            except Exception:
                category = "Trash"
                source_type = "Default"

        # Copy to the new subfolder
        new_filename = f"{category}_{counters[category]}.jpg"
        final_dest = os.path.join(dest_path, category, new_filename)

        shutil.copy2(file_path, final_dest)
        print(f"[{source_type}] -> {category}/{new_filename}")
        counters[category] += 1

# --- PATHS ---
source_dir = r'D:\OneDrive Data\Masters Study\Spring\EE6008 - DEEP LEARNING AT THE EDGE\-EE6008-Mini_Project\Garbage Classifier.v27i.voc\train2'
destination_dir = r'D:\OneDrive Data\Masters Study\Spring\EE6008 - DEEP LEARNING AT THE EDGE\-EE6008-Mini_Project\Sorted_Garbage_Dataset'

organize_dataset(source_dir, destination_dir)
