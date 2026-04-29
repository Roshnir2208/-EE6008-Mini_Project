"""
Garbage Classifier V4 - Improved accuracy training
Key changes vs V3:
  - 128x128 input (was 96) -> more texture detail
  - MobileNetV2 alpha=0.5  (was 0.35) -> more capacity
  - Class weights to correct Cardboard under-representation
  - Aggressive augmentation: noise, random crop, simulated blur
  - Longer fine-tuning, more backbone layers unfrozen
"""

import json, shutil
from datetime import datetime
from pathlib import Path
import tensorflow as tf

TFRECORD_DIR = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\data\tfrecords")
OUTPUT_DIR   = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models")
CAM_DIR      = Path(r"C:\Users\shall\Mini_Proj\garbage_classifier\Cam")
CAMERA_DRIVE = Path(r"D:\\")

IMG_SIZE    = 128   # larger -> better texture discrimination
ALPHA       = 0.5   # more capacity
BATCH_SIZE  = 64
LR_WARM     = 3e-4
LR_FINE     = 5e-5
EPOCHS_WARM = 20
EPOCHS_FINE = 30
PATIENCE    = 10
REP_SAMPLES = 300


# ---------------------------------------------------------------------------
def setup_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", tf.config.list_physical_devices("GPU"))


def load_class_mapping():
    with open(TFRECORD_DIR / "class_mapping.json", encoding="utf-8") as f:
        m = json.load(f)
    return m["class_to_index"], {int(k): v for k, v in m["index_to_class"].items()}


# ---------------------------------------------------------------------------
# Class weights (inverse frequency)
# ---------------------------------------------------------------------------
def compute_class_weights(num_classes):
    print("Computing class weights from training split...")
    counts = [0] * num_classes

    def _parse_labels(s):
        p = tf.io.parse_single_example(
            s, {"image/object/class/label": tf.io.VarLenFeature(tf.int64)})
        return tf.sparse.to_dense(p["image/object/class/label"])

    for lbls in (tf.data.TFRecordDataset(str(TFRECORD_DIR / "train.tfrecord"))
                 .map(_parse_labels)):
        for l in lbls.numpy():
            counts[int(l)] += 1

    total = sum(counts)
    weights = {i: total / (num_classes * max(c, 1)) for i, c in enumerate(counts)}
    print("  Counts:", counts)
    print("  Weights:", {i: round(v, 3) for i, v in weights.items()})
    return weights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _parse_record(serialized):
    feats = {
        "image/encoded":            tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/bbox/xmin":   tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax":   tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin":   tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax":   tf.io.VarLenFeature(tf.float32),
    }
    p = tf.io.parse_single_example(serialized, feats)
    img = tf.cast(tf.io.decode_jpeg(p["image/encoded"], channels=3), tf.float32) / 255.0
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    lbls = tf.sparse.to_dense(p["image/object/class/label"])
    xmin = tf.sparse.to_dense(p["image/object/bbox/xmin"])
    xmax = tf.sparse.to_dense(p["image/object/bbox/xmax"])
    ymin = tf.sparse.to_dense(p["image/object/bbox/ymin"])
    ymax = tf.sparse.to_dense(p["image/object/bbox/ymax"])
    has  = tf.size(lbls) > 0

    def pick():
        areas = tf.maximum((xmax-xmin)*(ymax-ymin), 1e-6)
        return tf.argmax(areas, output_type=tf.int32)

    idx   = tf.cond(has, pick, lambda: tf.constant(0, tf.int32))
    label = tf.cond(has, lambda: tf.cast(lbls[idx], tf.int32), lambda: tf.constant(0, tf.int32))
    x1 = tf.cond(has, lambda: xmin[idx], lambda: tf.constant(0.25, tf.float32))
    x2 = tf.cond(has, lambda: xmax[idx], lambda: tf.constant(0.75, tf.float32))
    y1 = tf.cond(has, lambda: ymin[idx], lambda: tf.constant(0.25, tf.float32))
    y2 = tf.cond(has, lambda: ymax[idx], lambda: tf.constant(0.75, tf.float32))
    bbox = tf.stack([(x1+x2)*0.5, (y1+y2)*0.5,
                     tf.clip_by_value(x2-x1, 0.01, 1.0),
                     tf.clip_by_value(y2-y1, 0.01, 1.0)])
    return img, label, bbox


def _augment(img, label, bbox):
    # Flip
    img = tf.image.random_flip_left_right(img)

    # Random crop: pad then crop back (simulates zoom/translation)
    pad = int(IMG_SIZE * 0.12)
    img = tf.pad(img, [[pad,pad],[pad,pad],[0,0]], mode="REFLECT")
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])

    # Colour jitter (aggressive)
    img = tf.image.random_brightness(img, 0.30)
    img = tf.image.random_contrast(img, 0.60, 1.40)
    img = tf.image.random_saturation(img, 0.55, 1.45)
    img = tf.image.random_hue(img, 0.10)

    # Gaussian noise (simulates real camera noise)
    img = img + tf.random.normal(tf.shape(img), 0.0, 0.04)

    # Simulate slight blur by downscale+upscale
    small = tf.image.resize(img, [IMG_SIZE//2, IMG_SIZE//2])
    img   = tf.cond(
        tf.random.uniform([]) > 0.5,
        lambda: tf.image.resize(small, [IMG_SIZE, IMG_SIZE]),
        lambda: img
    )

    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label, bbox


def _clf_ds(split, training):
    ds = tf.data.TFRecordDataset(str(TFRECORD_DIR / f"{split}.tfrecord"))
    ds = ds.map(_parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(3000).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda i,l,_: (i,l), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _det_ds(split, training):
    ds = tf.data.TFRecordDataset(str(TFRECORD_DIR / f"{split}.tfrecord"))
    ds = ds.map(_parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(3000).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda i,l,b: (i,{"cls":l,"bbox":b}), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _backbone():
    b = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=ALPHA, include_top=False, weights="imagenet")
    b.trainable = False
    return b


def build_classifier(n):
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x   = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inp)
    bb  = _backbone()
    x   = bb(x, training=False)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(n, activation="softmax", name="cls", dtype="float32")(x)
    return tf.keras.Model(inp, out), bb


def build_detector(n):
    inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x    = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inp)
    bb   = _backbone()
    x    = bb(x, training=False)
    x    = tf.keras.layers.GlobalAveragePooling2D()(x)
    x    = tf.keras.layers.Dense(256, activation="relu")(x)
    x    = tf.keras.layers.Dropout(0.30)(x)
    cls  = tf.keras.layers.Dense(n, activation="softmax", name="cls",  dtype="float32")(x)
    bbox = tf.keras.layers.Dense(4, activation="sigmoid",  name="bbox", dtype="float32")(x)
    return tf.keras.Model(inp, [cls, bbox]), bb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _callbacks(ckpt, monitor="val_accuracy"):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor=monitor, save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=4, min_lr=1e-6, verbose=1),
    ]


def _unfreeze(bb, n=60):
    bb.trainable = True
    for layer in bb.layers[:-n]:
        layer.trainable = False


def _representative():
    ds = (tf.data.TFRecordDataset(str(TFRECORD_DIR / "train.tfrecord"))
          .map(_parse_record).take(REP_SAMPLES))
    for img, _, _ in ds:
        yield [tf.expand_dims(tf.cast(img, tf.float32), 0)]


def export_int8(model, path):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = _representative
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type  = tf.int8
    c.inference_output_type = tf.int8
    data = c.convert()
    path.write_bytes(data)
    print(f"  INT8 -> {path}  ({len(data)/1024:.1f} KB)")
    return data


# ---------------------------------------------------------------------------
def main():
    setup_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    c2i, i2c = load_class_mapping()
    n   = len(c2i)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Classes ({n}):", list(c2i.keys()), "| tag:", tag)

    cw = compute_class_weights(n)

    # ------------------------------------------------------------------ #
    # CLASSIFIER                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("CLASSIFIER  alpha=%.2f  img=%d  epochs=%d+%d" % (ALPHA,IMG_SIZE,EPOCHS_WARM,EPOCHS_FINE))
    print("="*70)

    clf, clf_bb = build_classifier(n)
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARM),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    clf.summary(line_length=90)

    tr  = _clf_ds("train", True)
    val = _clf_ds("valid", False)
    ck  = OUTPUT_DIR / "clf_v4_best.h5"

    print(f"\nWarmup {EPOCHS_WARM} epochs (backbone frozen)...")
    clf.fit(tr, validation_data=val, epochs=EPOCHS_WARM,
            callbacks=_callbacks(ck), class_weight=cw, verbose=1)

    print(f"\nFine-tune top-60 backbone layers for {EPOCHS_FINE} epochs...")
    _unfreeze(clf_bb, 60)
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    clf.fit(tr, validation_data=val,
            initial_epoch=EPOCHS_WARM, epochs=EPOCHS_WARM+EPOCHS_FINE,
            callbacks=_callbacks(ck), class_weight=cw, verbose=1)

    clf.save(str(OUTPUT_DIR / "clf_v4_final.h5"))
    clf_tfl = OUTPUT_DIR / "classifier_v4_int8.tflite"
    print("\nExporting INT8...")
    export_int8(clf, clf_tfl)

    clf_acc = clf.evaluate(_clf_ds("test", False), verbose=0)[1]
    print(f"Classifier test accuracy: {clf_acc*100:.2f}%")

    # ------------------------------------------------------------------ #
    # DETECTOR                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("DETECTOR  alpha=%.2f  img=%d  epochs=%d+%d" % (ALPHA,IMG_SIZE,EPOCHS_WARM,EPOCHS_FINE))
    print("="*70)

    det, det_bb = build_detector(n)
    det.compile(
        optimizer=tf.keras.optimizers.Adam(LR_WARM),
        loss={"cls": tf.keras.losses.SparseCategoricalCrossentropy(),
              "bbox": tf.keras.losses.Huber()},
        loss_weights={"cls": 2.5, "bbox": 1.0},
        metrics={"cls": ["accuracy"]})
    det.summary(line_length=90)

    tr  = _det_ds("train", True)
    val = _det_ds("valid", False)
    dk  = OUTPUT_DIR / "det_v4_best.h5"

    print(f"\nWarmup {EPOCHS_WARM} epochs (backbone frozen)...")
    det.fit(tr, validation_data=val, epochs=EPOCHS_WARM,
            callbacks=_callbacks(dk, "val_cls_accuracy"), verbose=1)

    print(f"\nFine-tune top-60 backbone layers for {EPOCHS_FINE} epochs...")
    _unfreeze(det_bb, 60)
    det.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss={"cls": tf.keras.losses.SparseCategoricalCrossentropy(),
              "bbox": tf.keras.losses.Huber()},
        loss_weights={"cls": 2.5, "bbox": 1.0},
        metrics={"cls": ["accuracy"]})
    det.fit(tr, validation_data=val,
            initial_epoch=EPOCHS_WARM, epochs=EPOCHS_WARM+EPOCHS_FINE,
            callbacks=_callbacks(dk, "val_cls_accuracy"), verbose=1)

    det.save(str(OUTPUT_DIR / "det_v4_final.h5"))
    det_tfl = OUTPUT_DIR / "detector_v4_int8.tflite"
    print("\nExporting INT8...")
    export_int8(det, det_tfl)

    det_acc = det.evaluate(_det_ds("test", False), verbose=0)[-1]
    print(f"Detector test accuracy: {det_acc*100:.2f}%")

    # ------------------------------------------------------------------ #
    # Deploy                                                               #
    # ------------------------------------------------------------------ #
    labels_path = OUTPUT_DIR / "labels.txt"
    labels_path.write_text("\n".join(i2c[i] for i in range(n)) + "\n", encoding="utf-8")

    shutil.copy2(clf_tfl,    CAM_DIR / "trained.tflite")
    shutil.copy2(det_tfl,    CAM_DIR / "detector.tflite")
    shutil.copy2(labels_path, CAM_DIR / "labels.txt")
    print("\nCopied to Cam/")

    if CAMERA_DRIVE.exists():
        for src, name in {
            CAM_DIR/"trained.tflite":     "trained.tflite",
            CAM_DIR/"detector.tflite":    "detector.tflite",
            CAM_DIR/"labels.txt":         "labels.txt",
            CAM_DIR/"main.py":            "main.py",
            CAM_DIR/"main_classifier.py": "main_classifier.py",
            CAM_DIR/"main_detector.py":   "main_detector.py",
            CAM_DIR/"run_mode.txt":       "run_mode.txt",
        }.items():
            shutil.copy2(src, CAMERA_DRIVE / name)
        print("Deployed to", CAMERA_DRIVE)
    else:
        print("Camera drive not found, copy Cam/ manually.")

    print("\n" + "="*70)
    print(f"V4 done | Classifier {clf_acc*100:.1f}%  Detector {det_acc*100:.1f}%")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}  Alpha: {ALPHA}")
    print("="*70)


if __name__ == "__main__":
    main()
