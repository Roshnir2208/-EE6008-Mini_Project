"""
Garbage Classifier V3 - Complete Training Pipeline for OpenMV RT1062
Trains both classifier and detector with INT8 quantization.

Classifier : MobileNetV2 alpha=0.35, 96x96 -> 5-class softmax
Detector   : MobileNetV2 alpha=0.35, 96x96 -> [cls softmax, bbox sigmoid]

Target: >=88% accuracy, <700KB per model, >=30 FPS on RT1062.
GPU: RTX 4080 (CUDA). Reads pre-built TFRecords from data/tfrecords/.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TFRECORD_DIR = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\data\tfrecords")
OUTPUT_DIR   = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models")
CAM_DIR      = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\Cam")
CAMERA_DRIVE = Path(r"D:\\")  # Camera appears as D:\ on host when connected

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE      = 96      # 96x96 -> ~30 FPS on RT1062
BATCH_SIZE    = 64      # RTX 4080 has plenty of VRAM
ALPHA         = 0.35    # MobileNetV2 width multiplier -> small model
LR_WARM       = 3e-4
LR_FINETUNE   = 6e-5
EPOCHS_WARM   = 18      # Frozen backbone
EPOCHS_FINE   = 22      # Fine-tune top backbone layers
PATIENCE      = 8
REP_DS_SIZE   = 300     # Samples for INT8 calibration


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"Training on GPU: {gpus[0].name}")
    else:
        print("WARNING: No GPU found, training on CPU (slow)")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_class_mapping():
    with open(TFRECORD_DIR / "class_mapping.json", encoding="utf-8") as f:
        m = json.load(f)
    class_to_idx = m["class_to_index"]
    idx_to_class = {int(k): v for k, v in m["index_to_class"].items()}
    return class_to_idx, idx_to_class


def parse_record(serialized):
    feats = {
        "image/encoded":              tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin":     tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax":     tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin":     tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax":     tf.io.VarLenFeature(tf.float32),
        "image/object/class/label":   tf.io.VarLenFeature(tf.int64),
    }
    p = tf.io.parse_single_example(serialized, feats)

    img = tf.io.decode_jpeg(p["image/encoded"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    lbls = tf.sparse.to_dense(p["image/object/class/label"])
    xmin = tf.sparse.to_dense(p["image/object/bbox/xmin"])
    xmax = tf.sparse.to_dense(p["image/object/bbox/xmax"])
    ymin = tf.sparse.to_dense(p["image/object/bbox/ymin"])
    ymax = tf.sparse.to_dense(p["image/object/bbox/ymax"])

    has_obj = tf.size(lbls) > 0

    def pick_largest():
        areas = tf.maximum((xmax - xmin) * (ymax - ymin), 1e-6)
        return tf.argmax(areas, output_type=tf.int32)

    idx   = tf.cond(has_obj, pick_largest, lambda: tf.constant(0, tf.int32))
    label = tf.cond(has_obj, lambda: tf.cast(lbls[idx], tf.int32), lambda: tf.constant(0, tf.int32))

    x1 = tf.cond(has_obj, lambda: xmin[idx], lambda: tf.constant(0.25, tf.float32))
    x2 = tf.cond(has_obj, lambda: xmax[idx], lambda: tf.constant(0.75, tf.float32))
    y1 = tf.cond(has_obj, lambda: ymin[idx], lambda: tf.constant(0.25, tf.float32))
    y2 = tf.cond(has_obj, lambda: ymax[idx], lambda: tf.constant(0.75, tf.float32))

    cx   = (x1 + x2) * 0.5
    cy   = (y1 + y2) * 0.5
    bw   = tf.clip_by_value(x2 - x1, 0.01, 1.0)
    bh   = tf.clip_by_value(y2 - y1, 0.01, 1.0)
    bbox = tf.stack([cx, cy, bw, bh])

    return img, label, bbox


def augment(img, label, bbox):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.80, 1.20)
    img = tf.image.random_saturation(img, 0.80, 1.20)
    img = tf.image.random_hue(img, 0.05)
    img = tf.clip_by_value(img, 0.0, 1.0)
    # Random zoom-like crop: randomly scale & crop back to IMG_SIZE
    img = tf.image.resize_with_crop_or_pad(
        tf.image.resize(img, [int(IMG_SIZE * 1.10), int(IMG_SIZE * 1.10)]),
        IMG_SIZE, IMG_SIZE
    )
    return img, label, bbox


def make_classifier_ds(split, training):
    """Returns (image, label) dataset for classification."""
    ds = tf.data.TFRecordDataset(str(TFRECORD_DIR / f"{split}.tfrecord"))
    ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda img, lbl, _bbox: (img, lbl), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def make_detector_ds(split, training):
    """Returns (image, {"cls": label, "bbox": bbox}) dataset."""
    ds = tf.data.TFRecordDataset(str(TFRECORD_DIR / f"{split}.tfrecord"))
    ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda img, lbl, bbox: (img, {"cls": lbl, "bbox": bbox}),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _mobilenet_backbone():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=ALPHA,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    return base


def build_classifier(num_classes):
    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x       = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)  # [0,1]->[-1,1]
    backbone = _mobilenet_backbone()
    x       = backbone(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(128, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.30)(x)
    out     = tf.keras.layers.Dense(num_classes, activation="softmax", name="cls", dtype="float32")(x)
    return tf.keras.Model(inputs, out), backbone


def build_detector(num_classes):
    inputs   = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x        = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    backbone = _mobilenet_backbone()
    x        = backbone(x, training=False)
    x        = tf.keras.layers.GlobalAveragePooling2D()(x)
    x        = tf.keras.layers.Dense(192, activation="relu")(x)
    x        = tf.keras.layers.Dropout(0.25)(x)
    cls_out  = tf.keras.layers.Dense(num_classes, activation="softmax", name="cls", dtype="float32")(x)
    bbox_out = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox", dtype="float32")(x)
    return tf.keras.Model(inputs, [cls_out, bbox_out]), backbone


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _callbacks(ckpt_path, monitor="val_accuracy"):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor=monitor, save_best_only=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=4, min_lr=1e-6, verbose=1
        ),
    ]


def unfreeze_top(backbone, n=40):
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False


# ---------------------------------------------------------------------------
# INT8 export
# ---------------------------------------------------------------------------
def _representative_dataset():
    ds = (
        tf.data.TFRecordDataset(str(TFRECORD_DIR / "train.tfrecord"))
        .map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        .take(REP_DS_SIZE)
    )
    for img, _lbl, _bbox in ds:
        yield [tf.expand_dims(tf.cast(img, tf.float32), 0)]


def export_int8(model, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = _representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tfl)
    size_kb = len(tfl) / 1024
    print(f"  INT8 TFLite -> {out_path}  ({size_kb:.1f} KB)")
    return tfl


# ---------------------------------------------------------------------------
# Copy to camera drive
# ---------------------------------------------------------------------------
def deploy_to_camera(files_map):
    if not CAMERA_DRIVE.exists():
        print(f"\nCamera drive {CAMERA_DRIVE} not found, skipping auto-deploy.")
        print("Connect the camera and copy manually:")
        for src in files_map:
            print(f"  {src}")
        return
    print(f"\nDeploying to camera drive {CAMERA_DRIVE} ...")
    for src, dst_name in files_map.items():
        dst = CAMERA_DRIVE / dst_name
        shutil.copy2(src, dst)
        print(f"  Copied -> {dst}")
    print("Deploy complete.")


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------
def main():
    setup_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_to_idx, idx_to_class = load_class_mapping()
    num_classes = len(class_to_idx)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nClasses ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Run tag: {tag}")

    # ------------------------------------------------------------------ #
    # 1. CLASSIFIER                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 1 — Classifier (classification only, no bbox)")
    print("=" * 70)

    clf_model, clf_backbone = build_classifier(num_classes)
    clf_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARM),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    clf_model.summary(line_length=90)

    clf_ckpt = OUTPUT_DIR / "classifier_v3_best.h5"
    train_ds = make_classifier_ds("train", training=True)
    val_ds   = make_classifier_ds("valid", training=False)

    print(f"\n[Classifier] Warmup {EPOCHS_WARM} epochs (backbone frozen)...")
    clf_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARM,
                  callbacks=_callbacks(clf_ckpt), verbose=1)

    print(f"\n[Classifier] Fine-tune top-40 backbone layers for {EPOCHS_FINE} epochs...")
    unfreeze_top(clf_backbone, 40)
    clf_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    clf_model.fit(train_ds, validation_data=val_ds,
                  initial_epoch=EPOCHS_WARM, epochs=EPOCHS_WARM + EPOCHS_FINE,
                  callbacks=_callbacks(clf_ckpt), verbose=1)

    clf_model.save(str(OUTPUT_DIR / "classifier_v3_final.h5"))
    clf_tflite_path = OUTPUT_DIR / "classifier_v3_int8.tflite"
    print("\n[Classifier] Exporting INT8 TFLite...")
    export_int8(clf_model, clf_tflite_path)

    test_ds = make_classifier_ds("test", training=False)
    clf_results = clf_model.evaluate(test_ds, verbose=0)
    print(f"[Classifier] Test accuracy: {clf_results[1]*100:.2f}%")

    # ------------------------------------------------------------------ #
    # 2. DETECTOR                                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PHASE 2 — Detector (classification + bounding box)")
    print("=" * 70)

    det_model, det_backbone = build_detector(num_classes)
    det_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARM),
        loss={"cls": tf.keras.losses.SparseCategoricalCrossentropy(),
              "bbox": tf.keras.losses.Huber()},
        loss_weights={"cls": 2.5, "bbox": 1.0},
        metrics={"cls": ["accuracy"]},
    )
    det_model.summary(line_length=90)

    det_ckpt  = OUTPUT_DIR / "detector_v3_best.h5"
    train_ds  = make_detector_ds("train", training=True)
    val_ds    = make_detector_ds("valid", training=False)

    print(f"\n[Detector] Warmup {EPOCHS_WARM} epochs (backbone frozen)...")
    det_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARM,
                  callbacks=_callbacks(det_ckpt, monitor="val_cls_accuracy"), verbose=1)

    print(f"\n[Detector] Fine-tune top-40 backbone layers for {EPOCHS_FINE} epochs...")
    unfreeze_top(det_backbone, 40)
    det_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss={"cls": tf.keras.losses.SparseCategoricalCrossentropy(),
              "bbox": tf.keras.losses.Huber()},
        loss_weights={"cls": 2.5, "bbox": 1.0},
        metrics={"cls": ["accuracy"]},
    )
    det_model.fit(train_ds, validation_data=val_ds,
                  initial_epoch=EPOCHS_WARM, epochs=EPOCHS_WARM + EPOCHS_FINE,
                  callbacks=_callbacks(det_ckpt, monitor="val_cls_accuracy"), verbose=1)

    det_model.save(str(OUTPUT_DIR / "detector_v3_final.h5"))
    det_tflite_path = OUTPUT_DIR / "detector_v3_int8.tflite"
    print("\n[Detector] Exporting INT8 TFLite...")
    export_int8(det_model, det_tflite_path)

    test_ds = make_detector_ds("test", training=False)
    det_results = det_model.evaluate(test_ds, verbose=0)
    print(f"[Detector] Test cls_accuracy: {det_results[-1]*100:.2f}%")

    # ------------------------------------------------------------------ #
    # 3. Write labels.txt + metadata                                       #
    # ------------------------------------------------------------------ #
    labels_path = OUTPUT_DIR / "labels.txt"
    label_lines = [idx_to_class[i] for i in range(num_classes)]
    labels_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    print(f"\nLabels: {label_lines}")

    metadata = {
        "created_at": tag,
        "num_classes": num_classes,
        "classes": class_to_idx,
        "img_size": IMG_SIZE,
        "alpha": ALPHA,
        "classifier": {
            "tflite": str(clf_tflite_path),
            "test_accuracy": round(float(clf_results[1]), 4),
        },
        "detector": {
            "tflite": str(det_tflite_path),
            "test_cls_accuracy": round(float(det_results[-1]), 4),
        },
    }
    meta_path = OUTPUT_DIR / f"v3_metadata_{tag}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")

    # ------------------------------------------------------------------ #
    # 4. Deploy to camera (Cam/ folder + D:\ drive)                        #
    # ------------------------------------------------------------------ #
    cam_scripts = [
        CAM_DIR / "main.py",
        CAM_DIR / "main_classifier.py",
        CAM_DIR / "main_detector.py",
        CAM_DIR / "run_mode.txt",
    ]

    # Copy models + labels into Cam/ folder
    shutil.copy2(clf_tflite_path, CAM_DIR / "trained.tflite")
    shutil.copy2(det_tflite_path, CAM_DIR / "detector.tflite")
    shutil.copy2(labels_path,     CAM_DIR / "labels.txt")
    print("\nCopied models + labels to Cam/")

    # Copy everything to D:\ (camera drive)
    files_to_deploy = {
        CAM_DIR / "trained.tflite":     "trained.tflite",
        CAM_DIR / "detector.tflite":    "detector.tflite",
        CAM_DIR / "labels.txt":         "labels.txt",
        CAM_DIR / "main.py":            "main.py",
        CAM_DIR / "main_classifier.py": "main_classifier.py",
        CAM_DIR / "main_detector.py":   "main_detector.py",
        CAM_DIR / "run_mode.txt":       "run_mode.txt",
    }
    deploy_to_camera(files_to_deploy)

    print("\n" + "=" * 70)
    print("V3 Training complete!")
    print(f"  Classifier test accuracy : {clf_results[1]*100:.2f}%")
    print(f"  Detector   test accuracy : {det_results[-1]*100:.2f}%")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
