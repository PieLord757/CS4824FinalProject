import json
import yaml
from pathlib import Path

def verify_coco_format(json_path):
    """Verify COCO JSON format is valid."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    required_keys = ['info', 'images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: {key}"
    
    # Verify structure
    if not isinstance(data['images'], list):
        return False, "images must be a list"
    if not isinstance(data['annotations'], list):
        return False, "annotations must be a list"
    if not isinstance(data['categories'], list):
        return False, "categories must be a list"
    
    # Verify image IDs are unique
    image_ids = [img['id'] for img in data['images']]
    if len(image_ids) != len(set(image_ids)):
        return False, "Duplicate image IDs found"
    
    # Verify all annotations reference valid images
    valid_image_ids = set(image_ids)
    for ann in data['annotations']:
        if ann['image_id'] not in valid_image_ids:
            return False, f"Annotation references non-existent image_id: {ann['image_id']}"
    
    # Verify all annotations reference valid categories
    valid_category_ids = set(cat['id'] for cat in data['categories'])
    for ann in data['annotations']:
        if ann['category_id'] not in valid_category_ids:
            return False, f"Annotation references non-existent category_id: {ann['category_id']}"
    
    # Verify bbox format
    for ann in data['annotations']:
        if 'bbox' not in ann:
            return False, f"Annotation {ann['id']} missing bbox"
        if len(ann['bbox']) != 4:
            return False, f"Annotation {ann['id']} has invalid bbox format (should be [x, y, w, h])"
    
    return True, "COCO format is valid"

def verify_dataset_structure():
    """Verify the entire dataset structure."""
    base_dir = Path('coco-dataset')
    
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    issues = []
    
    # Check directory structure
    required_dirs = ['annotations', 'train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
            print(f"✗ Missing directory: {dir_name}/")
        else:
            print(f"✓ Found directory: {dir_name}/")
    
    # Check annotation files
    required_annotations = ['instances_train.json', 'instances_val.json', 'instances_test.json']
    for ann_file in required_annotations:
        ann_path = base_dir / 'annotations' / ann_file
        if not ann_path.exists():
            issues.append(f"Missing annotation file: {ann_path}")
            print(f"✗ Missing annotation: {ann_file}")
        else:
            print(f"✓ Found annotation: {ann_file}")
            
            # Verify JSON format
            valid, message = verify_coco_format(ann_path)
            if valid:
                print(f"  ✓ {message}")
            else:
                issues.append(f"{ann_file}: {message}")
                print(f"  ✗ {message}")
            
            # Count images and annotations
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"  - {len(data['images'])} images")
            print(f"  - {len(data['annotations'])} annotations")
            print(f"  - {len(data['categories'])} categories")
    
    # Verify images exist
    print("\nVerifying image files...")
    for split in ['train', 'val', 'test']:
        ann_path = base_dir / 'annotations' / f'instances_{split}.json'
        img_dir = base_dir / split
        
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        missing_images = []
        for img_info in data['images']:
            img_path = img_dir / img_info['file_name']
            if not img_path.exists():
                missing_images.append(img_info['file_name'])
        
        if missing_images:
            issues.append(f"{split} split: Missing {len(missing_images)} images")
            print(f"✗ {split}: Missing {len(missing_images)} images")
        else:
            print(f"✓ {split}: All images present")
    
    print("\n" + "=" * 60)
    if issues:
        print("VERIFICATION FAILED")
        print("=" * 60)
        for issue in issues:
            print(f"✗ {issue}")
        return False
    else:
        print("VERIFICATION PASSED ✓")
        print("=" * 60)
        print("Your dataset is properly formatted for RF-DETR training!")
        return True

def create_config_file():
    """Create configuration file for RF-DETR training."""
    
    config = {
        # Dataset configuration
        'data': {
            'train_ann_file': 'coco-dataset/annotations/instances_train.json',
            'train_img_dir': 'coco-dataset/train',
            'val_ann_file': 'coco-dataset/annotations/instances_val.json',
            'val_img_dir': 'coco-dataset/val',
            'test_ann_file': 'coco-dataset/annotations/instances_test.json',
            'test_img_dir': 'coco-dataset/test',
            'num_classes': 1,
            'class_names': ['Landing Zone']
        },
        
        # Model configuration
        'model': {
            'type': 'RT-DETR',
            'backbone': 'resnet50',  # or 'resnet18', 'resnet101'
            'num_queries': 100,  # Maximum number of objects to detect
            'num_classes': 1
        },
        
        # Training configuration
        'training': {
            'batch_size': 4,  # Adjust based on your GPU memory
            'num_epochs': 100,
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'lr_scheduler': 'MultiStepLR',
            'lr_drop_epochs': [60, 80],
            'clip_max_norm': 0.1,
            
            # Data augmentation
            'augmentation': {
                'random_flip': True,
                'random_crop': False,
                'color_jitter': True,
                'normalize': True
            }
        },
        
        # Evaluation configuration
        'evaluation': {
            'eval_interval': 5,  # Evaluate every N epochs
            'save_best_only': True,
            'metric': 'mAP'  # or 'loss'
        },
        
        # Output configuration
        'output': {
            'output_dir': 'output/landing_zone_detection',
            'checkpoint_dir': 'output/landing_zone_detection/checkpoints',
            'log_dir': 'output/landing_zone_detection/logs'
        },
        
        # Hardware configuration
        'device': 'cuda',  # or 'cpu', 'mps' for Mac
        'num_workers': 4,
        'seed': 42
    }
    
    # Save as YAML
    config_path = Path('config_landing_zone.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Configuration file created: {config_path}")
    print("\nConfiguration Summary:")
    print(f"  - Dataset: coco-dataset/")
    print(f"  - Classes: {config['data']['num_classes']} (Landing Zone)")
    print(f"  - Train images: 70")
    print(f"  - Val images: 20")
    print(f"  - Test images: 10")
    print(f"  - Batch size: {config['training']['batch_size']}")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    
    return config_path

def create_training_script():
    """Create a training script template."""
    
    training_script = '''#!/usr/bin/env python
# filepath: train_landing_zone.py
"""
Training script for Landing Zone Detection using RF-DETR
"""

import torch
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('config_landing_zone.yaml')
    
    print("=" * 60)
    print("RF-DETR TRAINING - Landing Zone Detection")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Classes: {config['data']['class_names']}")
    print("=" * 60)
    
    # TODO: Import RF-DETR library and set up training
    # This depends on the actual RF-DETR repository structure
    # Typically you would do something like:
    
    # from rf_detr import build_model, build_dataset, train
    # 
    # # Build model
    # model = build_model(config)
    # 
    # # Build datasets
    # train_dataset = build_dataset(
    #     config['data']['train_ann_file'],
    #     config['data']['train_img_dir']
    # )
    # val_dataset = build_dataset(
    #     config['data']['val_ann_file'],
    #     config['data']['val_img_dir']
    # )
    # 
    # # Train
    # train(model, train_dataset, val_dataset, config)
    
    print("\\nTo complete this training script:")
    print("1. Clone RF-DETR: git clone https://github.com/roboflow/rf-detr.git")
    print("2. Install dependencies: pip install -r rf-detr/requirements.txt")
    print("3. Update this script with actual RF-DETR imports and training logic")
    print("4. Run: python train_landing_zone.py")

if __name__ == '__main__':
    main()
'''
    
    script_path = Path('train_landing_zone.py')
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    print(f"\n✓ Training script template created: {script_path}")
    return script_path

def main():
    print("\nStep 1: Verifying dataset structure and COCO format...")
    dataset_valid = verify_dataset_structure()
    
    if not dataset_valid:
        print("\n❌ Dataset verification failed. Please fix the issues above.")
        return
    
    print("\nStep 2: Creating configuration file...")
    config_path = create_config_file()
    
    print("\nStep 3: Creating training script template...")
    script_path = create_training_script()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps to start training:")
    print("\n1. Install RF-DETR:")
    print("   git clone https://github.com/roboflow/rf-detr.git")
    print("   cd rf-detr")
    print("   pip install -r requirements.txt")
    print("\n2. Review and adjust config_landing_zone.yaml if needed")
    print("\n3. Complete the training script (train_landing_zone.py)")
    print("   with actual RF-DETR imports based on their repo structure")
    print("\n4. Start training:")
    print("   python train_landing_zone.py")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
