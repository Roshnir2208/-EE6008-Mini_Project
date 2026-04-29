"""
Train a tiny single-object detector for OpenMV RT1062.
Outputs class probabilities + one bounding box (cx, cy, w, h) per frame.
"""

import json
from pathlib import Path
from datetime import datetime

import tensorflow as tf


class TinyRT1062DetectorTrainer:
    def __init__(self, tfrecord_dir, output_dir):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.img_size = 128
        self.batch_size = 12
        self.epochs = 30
        self.learning_rate = 2e-4
        self.use_class_weights = False

        self._load_class_mapping()
        if self.use_class_weights:
            self._compute_class_weights()

    def _load_class_mapping(self):
        with open(self.tfrecord_dir / "class_mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.class_map = mapping["class_to_index"]
        self.reverse_class_map = mapping["index_to_class"]
        self.num_classes = len(self.class_map)
        print("Classes:", self.class_map)

    def _extract_primary_label(self, serialized):
        features = {
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        }
        parsed = tf.io.parse_single_example(serialized, features)

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
        return label

    def _compute_class_weights(self):
        print("\nComputing class weights from training set...")
        counts = [1] * self.num_classes
        train_file = self.tfrecord_dir / "train.tfrecord"
        ds = tf.data.TFRecordDataset(str(train_file))

        for serialized in ds:
            label = int(self._extract_primary_label(serialized).numpy())
            if 0 <= label < self.num_classes:
                counts[label] += 1

        total = float(sum(counts))
        weights = []
        for i, c in enumerate(counts):
            w = total / (self.num_classes * float(c))
            w = max(0.6, min(3.0, w))
            weights.append(w)

        # Targeted boosts for commonly confused paper classes.
        for class_name, boost in (("Paper", 1.25), ("Cardboard", 1.20)):
            if class_name in self.class_map:
                idx = self.class_map[class_name]
                weights[idx] = min(3.0, weights[idx] * boost)

        self.class_weights = weights
        self.class_weights_tensor = tf.constant(weights, dtype=tf.float32)
        print("Class counts:", counts)
        print("Class weights:", [round(w, 3) for w in weights])

    def _with_sample_weights(self, image, targets):
        cls_id = tf.cast(targets["cls"], tf.int32)
        cls_w = tf.gather(self.class_weights_tensor, cls_id)
        sample_weights = {
            "cls": tf.cast(cls_w, tf.float32),
            "bbox": tf.constant(1.0, dtype=tf.float32),
        }
        return image, targets, sample_weights

    def _parse_record(self, serialized):
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
        image = tf.image.resize(image, [self.img_size, self.img_size])
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

        targets = {
            "cls": label,
            "bbox": bbox,
        }

        return image, targets

    def _augment(self, image, targets):
        label = targets["cls"]
        bbox = targets["bbox"]

        flip = tf.random.uniform([]) > 0.5

        def do_flip():
            flipped_img = tf.image.flip_left_right(image)
            cx, cy, w, h = tf.unstack(bbox)
            flipped_bbox = tf.stack([1.0 - cx, cy, w, h], axis=0)
            return flipped_img, {"cls": label, "bbox": flipped_bbox}

        def keep():
            return image, {"cls": label, "bbox": bbox}

        image_out, targets_out = tf.cond(flip, do_flip, keep)
        image_out = tf.image.random_brightness(image_out, max_delta=0.08)
        image_out = tf.image.random_contrast(image_out, lower=0.9, upper=1.1)
        image_out = tf.clip_by_value(image_out, 0.0, 1.0)
        return image_out, targets_out

    def _dataset(self, split_name, training):
        ds = tf.data.TFRecordDataset(str(self.tfrecord_dir / f"{split_name}.tfrecord"))
        ds = ds.map(self._parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(1200)
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            if self.use_class_weights:
                ds = ds.map(self._with_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3), name="input_1")

        x = tf.keras.layers.Rescaling(255.0)(inputs)
        x = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input, name="preprocess")(x)

        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            alpha=0.35,
            include_top=False,
            weights="imagenet",
        )
        backbone.trainable = False
        x = backbone(x, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        cls_out = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="cls")(x)
        bbox_out = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)

        model = tf.keras.Model(inputs=inputs, outputs=[cls_out, bbox_out])
        return model, backbone

    def train(self):
        print("=" * 70)
        print("Training tiny RT1062 detector")
        print("=" * 70)

        train_ds = self._dataset("train", training=True)
        val_ds = self._dataset("valid", training=False)

        model, backbone = self._build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                "cls": tf.keras.losses.SparseCategoricalCrossentropy(),
                "bbox": tf.keras.losses.Huber(),
            },
            loss_weights={"cls": 2.0, "bbox": 1.0},
            metrics={"cls": ["accuracy"]},
        )

        model.summary()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.output_dir / "best_tiny_detector.h5"),
                monitor="val_cls_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_cls_accuracy",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        warmup_epochs = max(8, self.epochs // 3)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Fine-tune top backbone layers for better class separation.
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate * 0.2),
            loss={
                "cls": tf.keras.losses.SparseCategoricalCrossentropy(),
                "bbox": tf.keras.losses.Huber(),
            },
            loss_weights={"cls": 2.0, "bbox": 1.0},
            metrics={"cls": ["accuracy"]},
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(str(self.output_dir / "tiny_detector_final.h5"))
        print("Saved Keras model:", self.output_dir / "tiny_detector_final.h5")

        return model, history

    def export_int8_tflite(self, model):
        print("\nExporting full INT8 TFLite...")
        rep_ds = tf.data.TFRecordDataset(str(self.tfrecord_dir / "train.tfrecord"))
        rep_ds = rep_ds.map(self._parse_record, num_parallel_calls=tf.data.AUTOTUNE).take(150)

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
        tflite_path = self.output_dir / "tiny_detector_int8.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"Saved TFLite: {tflite_path} ({size_mb:.2f} MB)")

        metadata = {
            "type": "single_object_detector",
            "input_size": self.img_size,
            "num_classes": self.num_classes,
            "classes": self.class_map,
            "reverse_classes": self.reverse_class_map,
            "output_heads": ["cls", "bbox(cx,cy,w,h)"],
            "created_at": datetime.now().isoformat(),
            "target_device": "OpenMV RT1062",
            "model_size_mb": round(size_mb, 4),
        }

        with open(self.output_dir / "tiny_detector_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return tflite_path


def main():
    tfrecord_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\data\tfrecords"
    output_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models"

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU devices:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    trainer = TinyRT1062DetectorTrainer(tfrecord_dir, output_dir)
    model, _ = trainer.train()
    trainer.export_int8_tflite(model)

    print("\nDone: tiny detector training + export complete")


if __name__ == "__main__":
    main()
