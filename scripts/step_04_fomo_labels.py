import os, glob, json, numpy as np, cv2
import xml.etree.ElementTree as ET, tensorflow as tf

IN_DIR    = "augmented"
LABEL_MAP = json.load(open("label_map.json"))

NAME_TO_IDX = {v: int(k) for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)   # 8 (background + 7 waste classes)
CLASS_NAMES = [LABEL_MAP[str(i)] for i in range(1, NUM_CLASSES)]

IMG_SIZE = 96
GRID     = 12
CELL     = IMG_SIZE // GRID   # 8 pixels per cell
BATCH    = 32
AUTOTUNE = tf.data.AUTOTUNE


def xml_to_grid(xml_path, img_w, img_h):
    grid = np.zeros((GRID, GRID), dtype=np.int32)
    if not os.path.exists(xml_path):
        return grid
    for obj in ET.parse(xml_path).findall("object"):
        name    = obj.find("name").text.strip().lower()
        cls_idx = NAME_TO_IDX.get(name, 0)
        if cls_idx == 0:
            continue
        bb  = obj.find("bndbox")
        cx  = (float(bb.find("xmin").text) + float(bb.find("xmax").text)) / 2.0
        cy  = (float(bb.find("ymin").text) + float(bb.find("ymax").text)) / 2.0
        col = min(int(cx / (img_w / GRID)), GRID - 1)
        row = min(int(cy / (img_h / GRID)), GRID - 1)
        grid[row, col] = cls_idx   # larger box wins if collision
    return grid


def load_sample(img_bytes, xml_bytes):
    img_path = img_bytes.decode()
    xml_path = xml_bytes.decode()
    img      = cv2.imread(img_path)
    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img      = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    grid     = xml_to_grid(xml_path, IMG_SIZE, IMG_SIZE)
    onehot   = np.eye(NUM_CLASSES, dtype=np.float32)[grid]  # (12, 12, 8)
    return img, onehot


def build_ds(split):
    img_paths, xml_paths = [], []
    for cls in CLASS_NAMES:
        for p in sorted(glob.glob(os.path.join(IN_DIR, split, cls, "images", "*.jpg"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            img_paths.append(p)
            xml_paths.append(os.path.join(IN_DIR, split, cls, "Annotations", stem + ".xml"))
    ds = tf.data.Dataset.from_tensor_slices((img_paths, xml_paths))
    ds = ds.map(lambda i, x: tf.numpy_function(load_sample, [i, x], [tf.float32, tf.float32]),
                num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, lbl: (
        tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
        tf.ensure_shape(lbl, [GRID, GRID, NUM_CLASSES])
    ))
    if split == "train":
        ds = ds.shuffle(2000, seed=42)
    print(f"{split}: {len(img_paths)} samples")
    return ds.batch(BATCH).prefetch(AUTOTUNE)


for split, name in [("train", "train"), ("valid", "val"), ("test", "test")]:
    os.makedirs(f"tfdata/{name}", exist_ok=True)
    tf.data.experimental.save(build_ds(split), f"tfdata/{name}")

print("Saved to tfdata/")
