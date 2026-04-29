import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf

BASE = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier")
TFRECORD_DIR = BASE / "data" / "tfrecords"
MODELS_DIR = BASE / "output" / "trained_models"

IMG_SIZE = 128
BEST_H5 = MODELS_DIR / "best_tiny_detector.h5"
OUT_TFLITE = MODELS_DIR / "tiny_detector_int8.tflite"
OUT_META = MODELS_DIR / "tiny_detector_metadata.json"
MAPPING = TFRECORD_DIR / "class_mapping.json"


def parse_record(serialized):
    features = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized, features)

    image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    labels = tf.sparse.to_dense(parsed["image/object/class/label"])
    xmin = tf.sparse.to_dense(parsed["image/object/bbox/xmin"])
    xmax = tf.sparse.to_dense(parsed["image/object/bbox/xmax"])
    ymin = tf.sparse.to_dense(parsed["image/object/bbox/ymin"])
    ymax = tf.sparse.to_dense(parsed["image/object/bbox/ymax"])

    has_obj = tf.size(labels) > 0

    def select_obj_idx():
        areas = tf.maximum((xmax - xmin) * (ymax - ymin), 1e-6)
        return tf.argmax(areas, output_type=tf.int32)

    obj_idx = tf.cond(has_obj, select_obj_idx, lambda: tf.constant(0, tf.int32))
    label = tf.cond(has_obj, lambda: tf.cast(labels[obj_idx], tf.int32), lambda: tf.constant(0, tf.int32))

    x1 = tf.cond(has_obj, lambda: xmin[obj_idx], lambda: tf.constant(0.25, tf.float32))
    x2 = tf.cond(has_obj, lambda: xmax[obj_idx], lambda: tf.constant(0.75, tf.float32))
    y1 = tf.cond(has_obj, lambda: ymin[obj_idx], lambda: tf.constant(0.25, tf.float32))
    y2 = tf.cond(has_obj, lambda: ymax[obj_idx], lambda: tf.constant(0.75, tf.float32))

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = tf.clip_by_value(x2 - x1, 0.01, 1.0)
    h = tf.clip_by_value(y2 - y1, 0.01, 1.0)
    bbox = tf.stack([cx, cy, w, h], axis=0)

    return image, {"cls": label, "bbox": bbox}


print("Loading best model:", BEST_H5)
model = tf.keras.models.load_model(BEST_H5)

rep_ds = tf.data.TFRecordDataset(str(TFRECORD_DIR / "train.tfrecord"))
rep_ds = rep_ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE).take(200)


def representative_dataset():
    for image, _ in rep_ds:
        image = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
        yield [image]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

with open(MAPPING, "r", encoding="utf-8") as f:
    mapping = json.load(f)

size_mb = len(tflite_model) / (1024 * 1024)
metadata = {
    "type": "single_object_detector",
    "input_size": IMG_SIZE,
    "num_classes": len(mapping["class_to_index"]),
    "classes": mapping["class_to_index"],
    "reverse_classes": mapping["index_to_class"],
    "output_heads": ["cls", "bbox(cx,cy,w,h)"],
    "created_at": datetime.now().isoformat(),
    "target_device": "OpenMV RT1062",
    "model_size_mb": round(size_mb, 4),
    "source_model": str(BEST_H5),
}
with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {OUT_TFLITE} ({size_mb:.2f} MB)")
print(f"Saved: {OUT_META}")
