# **Dataset Acquisition & Preparation for Agricultural AI Chatbot**

The following is a comprehensive guide to finding and preparing datasets for rice pest/disease detection in Meta, Colombia.

## **1. Dataset Sources for Rice Pest/Disease Detection**

### **A. Primary Public Datasets**

| **Dataset Name**            | **Size**      | **Classes**             | **Region**  | **Access**                                                                        |
| --------------------------- | ------------- | ----------------------- | ----------- | --------------------------------------------------------------------------------- |
| **PlantVillage Rice**       | ~3,500 images | 3 diseases + healthy    | Global      | [Kaggle Link](https://www.kaggle.com/datasets/emmarex/plantdisease)               |
| **Rice Leaf Diseases**      | 4,000 images  | 3 diseases              | Bangladesh  | [GitHub](https://github.com/yusufarslan0/Rice-Leaf-Disease-Dataset)               |
| **Paddy Disease Dataset**   | 10,407 images | 10 classes              | Philippines | [Kaggle](https://www.kaggle.com/datasets/akhil4wire/paddy-disease-classification) |
| **Rice Diseases Detection** | 1,200 images  | 4 diseases              | India       | [Mendeley Data](https://data.mendeley.com/datasets/)                              |
| **IRRI Rice Doctor**        | 5,000+ images | Multiple pests/diseases | Asia        | [IRRI Website](http://ricedoctor.irri.org/)                                       |

### **B. Regional/Colombian Focus**

1. **Corpoica/Fedearroz Databases** (Colombian)
   
   - Contact: Colombian Corporation for Agricultural Research
   - May require partnership agreements
   - Contains local pest/disease data specific to Colombian conditions

2. **ICA Colombia** (Agricultural Institute)
   
   - Pest surveillance reports
   - Historical outbreak data
   - Contact through academic partnerships

3. **Universidad Nacional de Colombia - Palmira Campus**
   
   - Agricultural research departments
   - Field trial data
   - Existing image collections

### **C. Data Augmentation Sources**

```
# Complementary data from similar crops
1. Corn/Cereal Diseases - Similar fungal issues
2. General Plant Pathology Databases
3. Insect Pest Databases (non-rice specific for morphology)
```

## **2. Data Preparation Pipeline**

### **A. Technical Report Structure**

```python
"""
DATA PREPARATION TECHNICAL REPORT
Project: AI/AR System for Rice Pest Management - Meta, Colombia
Date: [Current Date]
Author: [Your Name/Team]

1. EXECUTIVE SUMMARY
   - Total images collected: X,XXX
   - Classes identified: 8 (6 diseases + 2 pest types)
   - Data augmentation techniques applied
   - Final dataset structure

2. DATA ACQUISITION
   2.1 Primary Sources
   2.2 Secondary Sources
   2.3 Field Collection Protocols

3. DATA CLEANING PROCESS
   3.1 Quality Assessment
   3.2 Duplicate Removal
   3.3 Invalid Image Filtering
   3.4 Metadata Verification

4. ANNOTATION METHODOLOGY
   4.1 Labeling Standards (COCO/YOLO format)
   4.2 Annotation Tool Selection
   4.3 Quality Control Procedures

5. PREPROCESSING PIPELINE
   5.1 Image Normalization
   5.2 Augmentation Strategies
   5.3 Dataset Splitting

6. DATASET STRUCTURE
   6.1 Directory Organization
   6.2 Metadata Files
   6.3 Version Control

7. VALIDATION METRICS
   7.1 Class Balance Analysis
   7.2 Data Quality Scores
   7.3 Expert Verification

8. ETHICAL CONSIDERATIONS
   8.1 Data Privacy
   8.2 Source Attribution
   8.3 Usage Rights

APPENDICES
   A. Annotation Guidelines
   B. Code Repository
   C. Quality Control Reports
"""
```

### **B. Data Cleaning Process**

```python
import cv2
import numpy as np
from pathlib import Path
import json

class DataCleaner:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    def validate_image(self, image_path):
        """Check image quality and validity"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Cannot read image"

            # Check dimensions
            if img.shape[0] < 224 or img.shape[1] < 224:
                return False, "Image too small"

            # Check blurriness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < 100:  # Threshold for blur detection
                return False, "Image too blurry"

            return True, "Valid"
        except Exception as e:
            return False, str(e)

    def remove_duplicates(self, hash_size=8):
        """Remove duplicate images using perceptual hashing"""
        from PIL import Image
        import imagehash

        hashes = {}
        duplicates = []

        for img_path in self.data_dir.rglob('*.*'):
            if img_path.suffix.lower() in self.valid_extensions:
                try:
                    img = Image.open(img_path)
                    img_hash = imagehash.phash(img, hash_size=hash_size)

                    if img_hash in hashes:
                        duplicates.append(img_path)
                    else:
                        hashes[img_hash] = img_path
                except:
                    continue

        # Remove duplicates
        for dup in duplicates:
            dup.unlink()

        return len(duplicates)
```

### **C. Annotation Pipeline**

```python
# Annotation standards for YOLOv8 format
"""
YOLO Format:
<class_id> <x_center> <y_center> <width> <height>
- All values normalized to [0, 1]
- One line per object
"""

class AnnotationManager:
    def __init__(self):
        self.class_mapping = {
            'rice_blast': 0,
            'brown_spot': 1,
            'bacterial_blight': 2,
            'sheath_blight': 3,
            'tungro': 4,
            'stem_borer': 5,
            'leaf_folder': 6,
            'healthy': 7
        }

    def convert_coco_to_yolo(self, coco_json_path, output_dir):
        """Convert COCO format to YOLO format"""
        import json

        with open(coco_json_path) as f:
            coco_data = json.load(f)

        # Create image_id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        annotations = coco_data['annotations']

        for img_id, img_info in images.items():
            img_annots = [a for a in annotations if a['image_id'] == img_id]

            yolo_lines = []
            for ann in img_annots:
                # COCO bbox: [x, y, width, height]
                x, y, w, h = ann['bbox']

                # Convert to YOLO format
                img_w, img_h = img_info['width'], img_info['height']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h

                class_id = self.class_mapping.get(ann['category_name'], 0)
                yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Write YOLO annotation file
            txt_path = output_dir / f"{Path(img_info['file_name']).stem}.txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
```

### **D. Preprocessing & Augmentation**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataPreprocessor:
    def get_training_transforms(self):
        return A.Compose([
            A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, 
                         saturation=0.2, hue=0.1, p=0.5),
            A.RandomShadow(p=0.2),
            A.RandomRain(p=0.1),  # Simulate field conditions
            A.RandomSunFlare(p=0.1),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def get_validation_transforms(self):
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
```

## **3. Dataset Organization Structure**

```
rice_pest_dataset/
├── README.md
├── dataset.yaml
├── raw_data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── coco_format.json
│       └── original_labels/
├── processed_data/
│   ├── images_640x640/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       ├── class_distribution.csv
│       ├── image_stats.json
│       └── split_indices.json
├── augmented_data/
│   ├── augmented_images/
│   └── augmented_labels/
└── scripts/
    ├── 01_data_cleaning.py
    ├── 02_data_augmentation.py
    ├── 03_dataset_split.py
    ├── 04_annotation_conversion.py
    └── 05_quality_check.py
```

### **Dataset Configuration File (dataset.yaml)**

```yaml
# Dataset configuration for YOLO training
path: ./rice_pest_dataset/processed_data
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: rice_blast
  1: brown_spot
  2: bacterial_blight
  3: sheath_blight
  4: tungro
  5: stem_borer
  6: leaf_folder
  7: healthy

# Additional metadata
metadata:
  total_images: 8500
  train_size: 5950
  val_size: 1275
  test_size: 1275
  image_size: 640
  created_date: 2026-02-15
  location: Meta, Colombia
  license: CC BY-NC-SA 4.0
  contributors:
    - Universidad Nacional de Colombia
    - Fedearroz
    - Project Team
```

## **4. Field Data Collection Protocol**

For collecting additional local data in Meta, Colombia:

```python
class FieldDataCollector:
    def __init__(self):
        self.collection_protocol = {
            'camera_settings': {
                'resolution': '1920x1080',
                'focus_mode': 'auto',
                'flash': 'auto',
                'format': 'JPEG'
            },
            'environmental_data': {
                'required_fields': [
                    'gps_coordinates',
                    'timestamp',
                    'temperature',
                    'humidity',
                    'growth_stage',
                    'variety'
                ]
            },
            'image_requirements': {
                'min_distance': '30 cm',
                'max_distance': '1 m',
                'angles': ['top', 'side', 'close-up'],
                'lighting': 'natural with scale card'
            }
        }

    def create_field_collection_app(self):
        """Simple app for field data collection"""
        import tkinter as tk
        from tkinter import filedialog

        # This would be a mobile app in practice
        # Using Kivy or React Native for actual implementation
        pass
```

## **5. Quality Control & Validation**

```python
class QualityValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def validate_dataset(self):
        """Comprehensive dataset validation"""
        report = {
            'total_images': 0,
            'valid_images': 0,
            'class_distribution': {},
            'annotation_issues': [],
            'image_quality_issues': []
        }

        # Check all images
        for img_path in self.dataset_path.rglob('*.jpg'):
            report['total_images'] += 1

            # Validate image
            is_valid, msg = self.validate_image(img_path)
            if is_valid:
                report['valid_images'] += 1
            else:
                report['image_quality_issues'].append({
                    'image': str(img_path),
                    'issue': msg
                })

            # Check corresponding annotation
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                ann_issues = self.validate_annotation(txt_path)
                if ann_issues:
                    report['annotation_issues'].extend(ann_issues)

        return report

    def check_class_balance(self):
        """Analyze class distribution"""
        import pandas as pd

        class_counts = {}
        for label_file in self.dataset_path.rglob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # Calculate statistics
        total = sum(class_counts.values())
        balance_report = {}
        for class_id, count in class_counts.items():
            percentage = (count / total) * 100
            balance_report[class_id] = {
                'count': count,
                'percentage': percentage,
                'status': 'BALANCED' if 5 <= percentage <= 20 else 'IMBALANCED'
            }

        return balance_report
```

## **6. Deliverable: Ready-to-Use Dataset Package**

### **Final Dataset Package Contents**

```
deliverable_rice_pest_dataset_v1.0.zip/
│
├── DATASET_CARD.md
├── LICENSE.txt
├── CITATION.cff
│
├── data/
│   ├── train/
│   │   ├── images/           # 6,350 images
│   │   └── labels/          # YOLO format annotations
│   │
│   ├── val/
│   │   ├── images/          # 1,350 images
│   │   └── labels/
│   │
│   └── test/
│       ├── images/          # 1,350 images
│       └── labels/
│
├── metadata/
│   ├── classes.json         # Class definitions
│   ├── splits.json          # Train/val/test splits
│   ├── statistics.json      # Dataset statistics
│   ├── sources.csv          # Data sources
│   └── transformations.json # Applied preprocessing
│
├── scripts/
│   ├── load_dataset.py      # Dataset loader
│   ├── visualize_samples.py # Visualization tools
│   └── benchmark_baseline.py # Baseline model training
│
└── docs/
    ├── annotation_guidelines.pdf
    ├── collection_protocol.pdf
    └── technical_report.pdf  # Full technical report
```

### **Technical Report Template**

```markdown
# TECHNICAL REPORT: Dataset Preparation
## For AI/AR Rice Pest Detection System - Meta, Colombia

### 1. Executive Summary
- **Total Images Processed**: 9,050
- **Final Dataset Size**: 8,750 (after cleaning)
- **Classes**: 8 (6 diseases, 2 pests)
- **Average Images per Class**: 1,094
- **Augmentation Ratio**: 3:1 (original:augmented)

### 2. Data Sources
| Source | Images | Contribution | License |
|--------|--------|--------------|---------|
| PlantVillage | 3,200 | 36.6% | CC BY 4.0 |
| Field Collection (Meta) | 2,500 | 28.6% | Project Owned |
| Paddy Disease Dataset | 2,050 | 23.4% | Research Use |
| IRRI Database | 1,000 | 11.4% | Academic |

### 3. Cleaning Process
- Removed 300 images (3.3%) due to quality issues
- Standardized all images to 640x640 resolution
- Applied histogram equalization for consistency
- Verified GPS metadata for Colombian origin

### 4. Annotation Statistics
- **Total Annotations**: 24,850 bounding boxes
- **Average Objects per Image**: 2.84
- **Annotation Consistency Score**: 94.7%
- **Inter-annotator Agreement**: 0.87 (Cohen's Kappa)

### 5. Preprocessing Pipeline
1. **Normalization**: ImageNet mean/std
2. **Augmentation Applied**:
   - Geometric: Rotation, Flip, Scale
   - Photometric: Color jitter, Noise, Blur
   - Environmental: Rain, Shadow, Sun flare simulation
3. **Class Balancing**: SMOTE oversampling for minority classes

### 6. Dataset Splits
- **Training**: 70% (6,125 images)
- **Validation**: 15% (1,312 images)
- **Test**: 15% (1,313 images)
- **Stratification**: Maintained class distribution across splits

### 7. Quality Metrics
- **Image Quality Score**: 96.4/100
- **Annotation Accuracy**: 95.2%
- **Class Balance Index**: 0.82 (Good)
- **Spatial Distribution**: Representative of field conditions

### 8. Validation Results
- Expert Agronomist Review: 92% accuracy confirmation
- Cross-dataset Validation: 88.3% mAP on external test set
- Model Performance Baseline (YOLOv8n):
  - Training mAP@0.5: 94.7%
  - Validation mAP@0.5: 92.1%
  - Inference Speed: 112ms/image (Snapdragon 680)

### 9. Ethical Considerations
- All field data collected with farmer consent
- GDPR compliance for any personal metadata
- Clear attribution for all external sources
- Open access for academic research

### 10. Recommendations for Use
1. Start with transfer learning from provided baseline
2. Fine-tune with local field data as it becomes available
3. Regular dataset updates (quarterly recommended)
4. Continuous quality monitoring during production use

### Appendices
- A. Detailed Class Distribution Charts
- B. Annotation Quality Samples
- C. Preprocessing Code Repository
- D. Validation Committee Report
```

## **7. Implementation Timeline**

```python
# 4-Week Data Preparation Plan
timeline = {
    'Week 1': [
        'Data source identification and collection',
        'Initial dataset assembly (raw data)',
        'Data cleaning pipeline development'
    ],
    'Week 2': [
        'Annotation process setup',
        'Quality control mechanisms',
        'Initial labeling (20% of dataset)'
    ],
    'Week 3': [
        'Complete annotation process',
        'Data augmentation implementation',
        'Dataset splitting and balancing'
    ],
    'Week 4': [
        'Final quality validation',
        'Expert review and verification',
        'Dataset packaging and documentation'
    ]
}
```

## **8. Tools & Resources**

### **Essential Tools:**

1. **Annotation Tools**:
   
   - CVAT (open-source)
   - LabelImg
   - Roboflow (cloud-based)

2. **Processing Libraries**:
   
   - OpenCV
   - Albumentations
   - Scikit-image

3. **Validation Tools**:
   
   - FiftyOne (dataset visualization)
   - Weights & Biases (tracking)
   - TensorBoard

### **Colombian-Specific Resources:**

1. **Contact Points**:
   
   - Fedearroz Research Centers in Meta
   - Universidad de los Llanos
   - SENA Agricultural Programs

2. **Local Data Collection**:
   
   - Partner with agricultural extension officers
   - Train local technicians in data collection
   - Use existing farmer networks
