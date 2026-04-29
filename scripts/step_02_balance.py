"""
step_02_balance.py
Organise images by dominant class for all three splits.
- TRAIN: NO undersampling — keep all images per class (diversity preserved).
         Augmentation in step_03 will bring every class to exactly 600.
- VALID/TEST: NO undersampling — keep all images per class.

This avoids the degenerate case where undersampling to the garbage class
minimum (57 train images) would throw away 95% of the training data.
"""
import os, shutil, glob, random, json, xml.etree.ElementTree as ET

random.seed(42)

RAW_DIR     = "raw_dataset"
OUT_DIR     = "balanced"
label_map   = json.load(open("label_map.json"))
CLASS_NAMES = [label_map[str(i)] for i in range(1, len(label_map))]


def dominant_class(xml_path):
    """Return the class name with the most objects in this annotation file."""
    counts = {}
    for obj in ET.parse(xml_path).findall("object"):
        n = obj.find("name").text.strip().lower()
        counts[n] = counts.get(n, 0) + 1
    return max(counts, key=counts.get) if counts else None


def find_img(split_dir, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(split_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


for split in ["train", "valid", "test"]:
    split_dir  = os.path.join(RAW_DIR, split)
    class_xmls = {c: [] for c in CLASS_NAMES}
    for xp in sorted(glob.glob(os.path.join(split_dir, "*.xml"))):
        dom = dominant_class(xp)
        if dom and dom in class_xmls:
            class_xmls[dom].append(xp)

    raw_counts = {c: len(v) for c, v in class_xmls.items()}
    print(f"\n{split}: organising (all images kept):")
    for c, n in raw_counts.items():
        print(f"  {c:12s}: {n}")

    for cls in CLASS_NAMES:
        dst_img = os.path.join(OUT_DIR, split, cls, "images")
        dst_ann = os.path.join(OUT_DIR, split, cls, "Annotations")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_ann, exist_ok=True)
        for xp in class_xmls[cls]:
            stem = os.path.splitext(os.path.basename(xp))[0]
            img  = find_img(split_dir, stem)
            if img is None:
                continue
            shutil.copy2(img, os.path.join(dst_img, os.path.basename(img)))
            shutil.copy2(xp,  os.path.join(dst_ann, os.path.basename(xp)))

print("\nOrganisation complete -> balanced/")
