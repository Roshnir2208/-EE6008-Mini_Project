"""
Dataset Preparation Script
Converts Roboflow CSV annotations to TensorFlow Record format for training
"""

import os
import csv
import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
import json

class DatasetConverter:
    def __init__(self, dataset_root, output_dir):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_map = {}
        self.class_reverse_map = {}
        
    def create_int64_feature(self, value):
        """Creates an int64 TensorFlow feature"""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    def create_float_feature(self, value):
        """Creates a float TensorFlow feature"""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def create_bytes_feature(self, value):
        """Creates a bytes TensorFlow feature"""
        if isinstance(value, str):
            value = value.encode('utf-8')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def load_classes(self, csv_file):
        """Load unique classes from annotations"""
        classes = set()
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                classes.add(row['class'])
        
        # Create mapping
        for idx, cls in enumerate(sorted(classes)):
            self.class_map[cls] = idx
            self.class_reverse_map[idx] = cls
        
        print(f"Found {len(self.class_map)} classes: {list(self.class_map.keys())}")
        
        # Save class mapping
        mapping_file = self.output_dir / "class_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'class_to_index': self.class_map,
                'index_to_class': {str(k): v for k, v in self.class_reverse_map.items()}
            }, f, indent=2)
        print(f"Class mapping saved to {mapping_file}")
    
    def load_annotations(self, csv_file, image_dir):
        """Load annotations from CSV"""
        annotations = {}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                
                if filename not in annotations:
                    annotations[filename] = {
                        'width': int(row['width']),
                        'height': int(row['height']),
                        'boxes': [],
                        'classes': [],
                        'image_path': image_dir / filename
                    }
                
                # Add bounding box
                annotations[filename]['boxes'].append([
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax'])
                ])
                annotations[filename]['classes'].append(self.class_map[row['class']])
        
        return annotations
    
    def create_tf_example(self, filename, annotation):
        """Create a TensorFlow Example proto"""
        image_path = annotation['image_path']
        
        # Read image
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return None
        
        with open(image_path, 'rb') as f:
            encoded_image = f.read()
        
        # Read image to get encoding
        with Image.open(image_path) as img:
            image_format = img.format.lower() if img.format else 'jpeg'
        
        width = annotation['width']
        height = annotation['height']
        boxes = annotation['boxes']
        classes = annotation['classes']
        
        # Normalize coordinates to [0, 1]
        xmins = [box[0] / width for box in boxes]
        xmaxs = [box[2] / width for box in boxes]
        ymins = [box[1] / height for box in boxes]
        ymaxs = [box[3] / height for box in boxes]
        
        feature = {
            'image/height': self.create_int64_feature(height),
            'image/width': self.create_int64_feature(width),
            'image/filename': self.create_bytes_feature(filename),
            'image/source_id': self.create_bytes_feature(filename),
            'image/encoded': self.create_bytes_feature(encoded_image),
            'image/format': self.create_bytes_feature(image_format),
            'image/object/bbox/xmin': self.create_float_feature(xmins),
            'image/object/bbox/xmax': self.create_float_feature(xmaxs),
            'image/object/bbox/ymin': self.create_float_feature(ymins),
            'image/object/bbox/ymax': self.create_float_feature(ymaxs),
            'image/object/class/text': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[self.class_reverse_map[c].encode('utf-8') for c in classes]
                )
            ),
            'image/object/class/label': self.create_int64_feature(classes),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def convert_split(self, split_name, csv_file, image_dir, output_file):
        """Convert a dataset split to TFRecord"""
        print(f"\nProcessing {split_name} split...")
        
        annotations = self.load_annotations(csv_file, image_dir)
        
        with tf.io.TFRecordWriter(str(output_file)) as writer:
            for filename, annotation in tqdm(annotations.items(), desc=f"Writing {split_name}"):
                example = self.create_tf_example(filename, annotation)
                if example:
                    writer.write(example.SerializeToString())
        
        print(f"✓ {split_name} split saved to {output_file}")
    
    def process_dataset(self):
        """Process all splits of the dataset"""
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            split_dir = self.dataset_root / split
            csv_file = split_dir / '_annotations.csv'
            output_file = self.output_dir / f'{split}.tfrecord'
            
            if not csv_file.exists():
                print(f"Warning: {csv_file} not found, skipping {split} split")
                continue
            
            self.convert_split(split, csv_file, split_dir, output_file)

def main():
    # Paths
    dataset_root = r"C:\Users\shall\Mini_Proj\Garbage Classifier.v27i.tensorflow"
    output_dir = r"C:\Users\shall\Mini_Proj\garbage_classifier\data\tfrecords"
    
    print("=" * 60)
    print("TensorFlow Dataset Converter for Garbage Classification")
    print("=" * 60)
    
    # Create converter
    converter = DatasetConverter(dataset_root, output_dir)
    
    # First pass: load classes
    print("\n[Step 1] Loading classes...")
    csv_train = Path(dataset_root) / 'train' / '_annotations.csv'
    converter.load_classes(csv_train)
    
    # Second pass: convert all splits
    print("\n[Step 2] Converting dataset splits...")
    converter.process_dataset()
    
    print("\n" + "=" * 60)
    print("Dataset conversion complete!")
    print(f" TFRecords saved to: {output_dir}")
    print(f" Class mapping saved: {output_dir}/class_mapping.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
