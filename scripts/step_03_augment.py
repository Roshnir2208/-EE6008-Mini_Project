"""
step_03_augment.py
- Train: each class → exactly 600 images.
    Classes with >= 600 originals: random sample 600 (resize to 96x96, scale XMLs).
    Classes with <  600 originals: keep all + augment to reach 600.
- Valid/Test: resize to 96x96, scale bounding boxes — no augmentation.
"""
import os, glob, shutil, random, json
import cv2, albumentations as A, numpy as np, xml.etree.ElementTree as ET

random.seed(42)
np.random.seed(42)

IN_DIR           = "balanced"
OUT_DIR          = "augmented"
TARGET_PER_CLASS = 600
IMG_SIZE         = 96

label_map   = json.load(open("label_map.json"))
CLASS_NAMES = [label_map[str(i)] for i in range(1, len(label_map))]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(limit=0.25, p=0.7),
    A.RandomContrast(limit=0.25, p=0.7),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=12,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.RandomCrop(height=80, width=80, p=0.4),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.GaussNoise(var_limit=(5, 25), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'],
                             min_visibility=0.3))


def load_voc(xml_path, w, h):
    boxes, labels = [], []
    for obj in ET.parse(xml_path).findall("object"):
        bb  = obj.find("bndbox")
        xmn = max(0., float(bb.find("xmin").text))
        ymn = max(0., float(bb.find("ymin").text))
        xmx = min(float(w), float(bb.find("xmax").text))
        ymx = min(float(h), float(bb.find("ymax").text))
        if xmx > xmn and ymx > ymn:
            boxes.append([xmn, ymn, xmx, ymx])
            labels.append(obj.find("name").text.strip().lower())
    return boxes, labels


def save_voc(path, stem, w, h, boxes, labels):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = stem + ".jpg"
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text  = str(w)
    ET.SubElement(sz, "height").text = str(h)
    ET.SubElement(sz, "depth").text  = "3"
    for box, lbl in zip(boxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = lbl
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(int(box[0]))
        ET.SubElement(bb, "ymin").text = str(int(box[1]))
        ET.SubElement(bb, "xmax").text = str(int(box[2]))
        ET.SubElement(bb, "ymax").text = str(int(box[3]))
    ET.ElementTree(root).write(path)


def resize_and_save(src_img, src_xml, dst_img_path, dst_xml_path):
    """Resize image to 96x96 and proportionally scale bounding boxes."""
    img    = cv2.imread(src_img)
    orig_h, orig_w = img.shape[:2]
    cv2.imwrite(dst_img_path, cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    if os.path.exists(src_xml):
        boxes, labels = load_voc(src_xml, orig_w, orig_h)
        sx = IMG_SIZE / orig_w
        sy = IMG_SIZE / orig_h
        scaled = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes]
        stem = os.path.splitext(os.path.basename(dst_xml_path))[0]
        save_voc(dst_xml_path, stem, IMG_SIZE, IMG_SIZE, scaled, labels)


# ── Valid / Test: resize + scale bounding boxes, no augmentation ───────────
for split in ["valid", "test"]:
    for cls in CLASS_NAMES:
        src_img_dir = os.path.join(IN_DIR, split, cls, "images")
        src_ann_dir = os.path.join(IN_DIR, split, cls, "Annotations")
        dst_img_dir = os.path.join(OUT_DIR, split, cls, "images")
        dst_ann_dir = os.path.join(OUT_DIR, split, cls, "Annotations")
        if not os.path.exists(src_img_dir):
            continue
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)
        for src_img in glob.glob(os.path.join(src_img_dir, "*.jpg")) + \
                       glob.glob(os.path.join(src_img_dir, "*.jpeg")) + \
                       glob.glob(os.path.join(src_img_dir, "*.png")):
            stem = os.path.splitext(os.path.basename(src_img))[0]
            resize_and_save(
                src_img,
                os.path.join(src_ann_dir, stem + ".xml"),
                os.path.join(dst_img_dir, stem + ".jpg"),
                os.path.join(dst_ann_dir, stem + ".xml"),
            )

# ── Train: sample or augment each class to exactly TARGET_PER_CLASS ────────
for cls in CLASS_NAMES:
    src_img_dir = os.path.join(IN_DIR, "train", cls, "images")
    src_ann_dir = os.path.join(IN_DIR, "train", cls, "Annotations")
    all_imgs    = sorted(glob.glob(os.path.join(src_img_dir, "*.jpg")) +
                         glob.glob(os.path.join(src_img_dir, "*.jpeg")) +
                         glob.glob(os.path.join(src_img_dir, "*.png")))
    n_orig      = len(all_imgs)
    dst_img_dir = os.path.join(OUT_DIR, "train", cls, "images")
    dst_ann_dir = os.path.join(OUT_DIR, "train", cls, "Annotations")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    if n_orig >= TARGET_PER_CLASS:
        # Enough originals: randomly sample TARGET_PER_CLASS, resize + scale
        chosen = random.sample(all_imgs, TARGET_PER_CLASS)
        for src in chosen:
            stem = os.path.splitext(os.path.basename(src))[0]
            resize_and_save(
                src,
                os.path.join(src_ann_dir, stem + ".xml"),
                os.path.join(dst_img_dir, stem + "_orig.jpg"),
                os.path.join(dst_ann_dir, stem + "_orig.xml"),
            )
        print(f"{cls:12s}: {n_orig} orig → sampled {TARGET_PER_CLASS} (no augmentation needed)")
    else:
        # Not enough originals: keep all + augment to TARGET_PER_CLASS
        n_needed = TARGET_PER_CLASS - n_orig
        for src in all_imgs:
            stem = os.path.splitext(os.path.basename(src))[0]
            resize_and_save(
                src,
                os.path.join(src_ann_dir, stem + ".xml"),
                os.path.join(dst_img_dir, stem + "_orig.jpg"),
                os.path.join(dst_ann_dir, stem + "_orig.xml"),
            )
        print(f"{cls:12s}: {n_orig} orig + {n_needed} augmented")
        aug_count = 0
        while aug_count < n_needed:
            src     = random.choice(all_imgs)
            stem    = os.path.splitext(os.path.basename(src))[0]
            img     = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            src_xml = os.path.join(src_ann_dir, stem + ".xml")
            boxes, labels = load_voc(src_xml, orig_w, orig_h) if os.path.exists(src_xml) else ([], [])
            # Scale boxes to IMG_SIZE first, then augment (Albumentations expects image-scale coords)
            sx = IMG_SIZE / orig_w
            sy = IMG_SIZE / orig_h
            img_rs = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (IMG_SIZE, IMG_SIZE))
            img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
            boxes_s = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes]
            try:
                res = transform(image=img_rs, bboxes=boxes_s, class_labels=labels)
            except Exception:
                continue
            if not res["bboxes"] and boxes:
                continue
            new_stem = f"{stem}_aug{aug_count:04d}"
            cv2.imwrite(os.path.join(dst_img_dir, new_stem + ".jpg"),
                        cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR))
            save_voc(os.path.join(dst_ann_dir, new_stem + ".xml"),
                     new_stem, IMG_SIZE, IMG_SIZE, res["bboxes"], res["class_labels"])
            aug_count += 1

# Checkpoint
print("\nCHECKPOINT — verifying exactly 600 images/class in train:")
all_ok = True
for cls in CLASS_NAMES:
    n = len(glob.glob(os.path.join(OUT_DIR, "train", cls, "images", "*")))
    ok = n == TARGET_PER_CLASS
    print(f"  {cls:12s}: {n}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_ok = False
assert all_ok, "Some classes not exactly 600"

print("\nAugmentation complete -> augmented/")
