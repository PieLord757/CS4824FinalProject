import json
import os
import random
from pathlib import Path
from PIL import Image
import shutil

class YOLOtoCOCOConverter:
    def __init__(self, 
                 labeled_data_dir='labeled-data',
                 output_dir='coco-dataset',
                 train_ratio=0.7,
                 val_ratio=0.2,
                 test_ratio=0.1,
                 random_seed=42):
        """
        Convert YOLO format annotations to COCO format with train/val/test splits.
        
        Args:
            labeled_data_dir: Directory containing images/ and labels/ subdirs
            output_dir: Directory to save COCO-formatted dataset
            train_ratio: Percentage for training set
            val_ratio: Percentage for validation set
            test_ratio: Percentage for test set
            random_seed: Random seed for reproducible splits
        """
        self.labeled_data_dir = Path(labeled_data_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Paths
        self.images_dir = self.labeled_data_dir / 'images'
        self.labels_dir = self.labeled_data_dir / 'labels'
        
    def setup_output_directories(self):
        """Create output directory structure for COCO dataset."""
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Create annotations directory
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        
    def get_image_dimensions(self, image_path):
        """Get width and height of an image."""
        with Image.open(image_path) as img:
            return img.width, img.height
    
    def yolo_to_coco_bbox(self, yolo_bbox, img_width, img_height):
        """
        Convert YOLO format bbox to COCO format.
        
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        COCO: [x_min, y_min, width, height] (absolute pixels)
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert from normalized to absolute coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Convert to top-left corner format
        x_min = x_center_abs - (width_abs / 2)
        y_min = y_center_abs - (height_abs / 2)
        
        return [x_min, y_min, width_abs, height_abs]
    
    def split_dataset(self, image_files):
        """Randomly split dataset into train/val/test sets."""
        # Shuffle the image files randomly
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)
        
        total = len(shuffled_files)
        train_count = int(total * self.train_ratio)
        val_count = int(total * self.val_ratio)
        
        train_files = shuffled_files[:train_count]
        val_files = shuffled_files[train_count:train_count + val_count]
        test_files = shuffled_files[train_count + val_count:]
        
        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
    
    def create_coco_annotation(self, split_name, image_files):
        """
        Create COCO format annotation JSON for a specific split.
        
        Args:
            split_name: 'train', 'val', or 'test'
            image_files: List of image filenames for this split
            
        Returns:
            Dictionary in COCO format
        """
        coco_format = {
            "info": {
                "description": "Landing Zone Detection Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "CS4824",
                "date_created": "2025-11-27"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 0,
                    "name": "Landing Zone",
                    "supercategory": "zone"
                }
            ]
        }
        
        annotation_id = 0
        
        for img_id, image_file in enumerate(image_files):
            image_path = self.images_dir / image_file
            label_file = image_file.replace('.png', '.txt')
            label_path = self.labels_dir / label_file
            
            # Get image dimensions
            width, height = self.get_image_dimensions(image_path)
            
            # Add image info
            image_info = {
                "id": img_id,
                "file_name": image_file,
                "width": width,
                "height": height
            }
            coco_format["images"].append(image_info)
            
            # Read YOLO annotations if they exist
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.split()
                    class_id = int(parts[0])
                    yolo_bbox = [float(x) for x in parts[1:5]]
                    
                    # Convert to COCO format
                    coco_bbox = self.yolo_to_coco_bbox(yolo_bbox, width, height)
                    
                    # Calculate area
                    area = coco_bbox[2] * coco_bbox[3]
                    
                    # Create annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1
            
            # Copy image to split directory
            dest_path = self.output_dir / split_name / image_file
            shutil.copy(image_path, dest_path)
        
        return coco_format
    
    def convert(self):
        """Main conversion function."""
        print("Starting YOLO to COCO conversion...")
        print("="*60)
        
        # Setup directories
        self.setup_output_directories()
        
        # Get all image files
        image_files = sorted([f.name for f in self.images_dir.glob('*.png')])
        print(f"Found {len(image_files)} images")
        
        # Split dataset
        splits = self.split_dataset(image_files)
        
        print(f"\nDataset split:")
        print(f"  Train: {len(splits['train'])} images ({len(splits['train'])/len(image_files)*100:.1f}%)")
        print(f"  Val:   {len(splits['val'])} images ({len(splits['val'])/len(image_files)*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} images ({len(splits['test'])/len(image_files)*100:.1f}%)")
        
        # Create COCO annotations for each split
        for split_name, split_files in splits.items():
            if not split_files:
                print(f"\nSkipping {split_name} (no files)")
                continue
                
            print(f"\nProcessing {split_name} split...")
            coco_data = self.create_coco_annotation(split_name, split_files)
            
            # Save JSON file
            json_path = self.output_dir / 'annotations' / f'instances_{split_name}.json'
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"  Created {json_path}")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
        
        # Create dataset info file
        info = {
            "dataset_name": "Landing Zone Detection",
            "format": "COCO",
            "classes": ["Landing Zone"],
            "splits": {
                "train": len(splits['train']),
                "val": len(splits['val']),
                "test": len(splits['test'])
            },
            "total_images": len(image_files),
            "image_format": "png"
        }
        
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print(f"\nDataset structure created at: {self.output_dir}")
        print("\nDirectory structure:")
        print("  coco-dataset/")
        print("  ├── annotations/")
        print("  │   ├── instances_train.json")
        print("  │   ├── instances_val.json")
        print("  │   └── instances_test.json")
        print("  ├── train/")
        print("  │   └── [training images]")
        print("  ├── val/")
        print("  │   └── [validation images]")
        print("  ├── test/")
        print("  │   └── [test images]")
        print("  └── dataset_info.json")
        
        # Print sample from train set
        if splits['train']:
            print(f"\nSample train images: {splits['train'][:5]}")
        if splits['val']:
            print(f"Sample val images: {splits['val'][:5]}")
        if splits['test']:
            print(f"Sample test images: {splits['test'][:3]}")

def main():
    converter = YOLOtoCOCOConverter(
        labeled_data_dir='labeled-data',
        output_dir='coco-dataset',
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42  # Change this for different random splits
    )
    
    converter.convert()

if __name__ == '__main__':
    main()
