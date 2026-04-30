import os
import hashlib
import random
from collections import Counter

# --- CONFIGURATION ---
destination_dir = r'D:\OneDrive Data\Masters Study\Spring\EE6008 - DEEP LEARNING AT THE EDGE\-EE6008-Mini_Project\Sorted_Garbage_Dataset'

# Set this to the image count of the SMALLEST class to achieve perfect balance
TARGET_COUNT = 150 

def get_file_hash(file_path):
    """Identifies identical images to prevent 'leakage' between train/val sets."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def process_dataset():
    if not os.path.exists(destination_dir):
        print(f"Error: Path {destination_dir} not found. Check your drive connection.")
        return

    # Identify class folders (Cardboard, Glass, etc.)
    classes = [d for d in os.listdir(destination_dir) if os.path.isdir(os.path.join(destination_dir, d))]
    
    print("--- Phase 1: Removing Exact Duplicates ---")
    for cls in classes:
        path = os.path.join(destination_dir, cls)
        hashes = set()
        removed = 0
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            h = get_file_hash(img_path)
            if h in hashes:
                os.remove(img_path)
                removed += 1
            else:
                hashes.add(h)
        if removed > 0:
            print(f"[{cls}]: Deleted {removed} duplicate images.")

    print("\n--- Phase 2: Balancing to Target Count ---")
    for cls in classes:
        path = os.path.join(destination_dir, cls)
        files = os.listdir(path)
        current_total = len(files)
        
        if current_total > TARGET_COUNT:
            to_remove = random.sample(files, current_total - TARGET_COUNT)
            for f in to_remove:
                os.remove(os.path.join(path, f))
            print(f"[{cls}]: Pruned {len(to_remove)} images. New total: {TARGET_COUNT}")
        else:
            print(f"[{cls}]: Below or at target ({current_total} images).")

    print("\nProcessing complete. Your dataset is now balanced.")

if __name__ == "__main__":
    # Safety check before mass deletion
    confirm = input(f"This will modify files in {destination_dir}. Continue? (y/n): ")
    if confirm.lower() == 'y':
        process_dataset()
