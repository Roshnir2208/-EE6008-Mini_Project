"""
Training Script for TensorFlow Lite Garbage Classification Model
Uses GPU-accelerated training with fine-tuning on pre-trained model
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class GarbageClassificationTrainer:
    def __init__(self, tfrecord_dir, output_dir, model_name='mobilenet_v2'):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load class mapping
        self.load_class_mapping()
        
        # Training parameters tuned for RT1062 memory limits
        self.IMG_SIZE = 160
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 25
        
    def load_class_mapping(self):
        """Load class mapping from prepared dataset"""
        mapping_file = self.tfrecord_dir / 'class_mapping.json'
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        self.class_map = mapping['class_to_index']
        self.reverse_class_map = mapping['index_to_class']
        self.num_classes = len(self.class_map)
        print(f"Classes: {self.class_map}")
    
    def load_tfrecord(self, tfrecord_path):
        """Load and parse TFRecord"""
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        
        def parse_tfrecord(serialized):
            features = {
                'image/height': tf.io.FixedLenFeature([], tf.int64),
                'image/width': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/format': tf.io.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            }
            
            parsed = tf.io.parse_single_example(serialized, features)
            
            # Decode image
            image = tf.io.decode_jpeg(parsed['image/encoded'], channels=3)
            image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE])
            image = image / 255.0  # Normalize
            
            # Parse labels - take the first label (primary class)
            labels = tf.sparse.to_dense(parsed['image/object/class/label'])
            labels = tf.cast(labels, tf.int32)
            # Get first label or 0 if empty
            if tf.size(labels) > 0:
                label = labels[0]
            else:
                label = tf.constant(0, dtype=tf.int32)
            
            return image, label
        
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        train_tfrecord = self.tfrecord_dir / 'train.tfrecord'
        valid_tfrecord = self.tfrecord_dir / 'valid.tfrecord'
        
        print(f"\nLoading training data from {train_tfrecord}...")
        train_dataset = self.load_tfrecord(train_tfrecord)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"Loading validation data from {valid_tfrecord}...")
        valid_dataset = self.load_tfrecord(valid_tfrecord)
        valid_dataset = valid_dataset.batch(self.BATCH_SIZE)
        valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, valid_dataset
    
    def build_model(self):
        """Build compact classifier for RT1062 deployment"""
        print(f"\nBuilding {self.model_name} model...")
        
        # Smaller MobileNetV2 backbone keeps model size and RAM usage low.
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
            alpha=0.35,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add compact classification head
        inputs = tf.keras.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        
        # Feature extraction
        x = base_model(inputs, training=False)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        classification = tf.keras.layers.Dense(128, activation='relu')(x)
        classification = tf.keras.layers.Dropout(0.3)(classification)
        classification_output = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='classification'
        )(classification)
        
        # Build model
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)
        
        return model, base_model
    
    def compile_model(self, model):
        """Compile model with optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel compiled successfully")
        model.summary()
    
    def train(self):
        """Train the model"""
        print("\n" + "=" * 70)
        print("Starting Training with GPU Support")
        print("=" * 70)
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # Prepare data
        train_dataset, valid_dataset = self.prepare_datasets()
        
        # Build model
        model, base_model = self.build_model()
        self.compile_model(model)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train
        history = model.fit(
            train_dataset,
            epochs=self.EPOCHS,
            validation_data=valid_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(str(self.output_dir / 'garbage_classifier_final.h5'))
        print(f"\n Final model saved to {self.output_dir / 'garbage_classifier_final.h5'}")
        
        return model, history
    
    def convert_to_tflite(self, model):
        """Convert model to TensorFlow Lite format"""
        print("\n" + "=" * 70)
        print("Converting Model to TensorFlow Lite Format")
        print("=" * 70)
        
        # Build a representative dataset for full INT8 quantization.
        rep_dataset = self.load_tfrecord(self.tfrecord_dir / 'train.tfrecord').take(200)

        def representative_dataset():
            for image, _ in rep_dataset:
                image = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
                yield [image]

        # Convert with aggressive optimization for microcontroller deployment.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = self.output_dir / 'garbage_classifier.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        return tflite_path
    
    def save_metadata(self, model, tflite_path):
        """Save model metadata for deployment"""
        metadata = {
            'model_name': 'Garbage Classifier',
            'input_size': self.IMG_SIZE,
            'num_classes': self.num_classes,
            'classes': self.class_map,
            'reverse_classes': self.reverse_class_map,
            'model_path': str(tflite_path),
            'created_at': datetime.now().isoformat(),
            'framework': 'TensorFlow Lite',
            'target_device': 'OpenMV RT1062'
        }
        
        metadata_path = self.output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Model metadata saved to {metadata_path}")

def main():
    # Paths
    tfrecord_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\data\tfrecords"
    output_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\output\trained_models"
    
    print("=" * 70)
    print("Garbage Classification Model Training")
    print("GPU-Accelerated TensorFlow Lite Training")
    print("=" * 70)
    
    # Create trainer
    trainer = GarbageClassificationTrainer(tfrecord_dir, output_dir)
    
    # Train model
    model, history = trainer.train()
    
    # Convert to TFLite
    tflite_path = trainer.convert_to_tflite(model)
    
    # Save metadata
    trainer.save_metadata(model, tflite_path)
    
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
