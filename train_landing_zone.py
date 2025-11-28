#!/usr/bin/env python
# filepath: train_landing_zone.py
"""
Training script for Landing Zone Detection using RF-DETR

This script trains an RF-DETR model on the Landing Zone detection dataset.
The dataset is organized in COCO format with train/val/test splits.
"""

import os
import json
import torch
from pathlib import Path
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase
from rfdetr.config import TrainConfig

def setup_roboflow_format(coco_dataset_dir='coco-dataset'):
    """
    Convert COCO format dataset to Roboflow format expected by RF-DETR.
    RF-DETR expects:
    - dataset_dir/train/_annotations.coco.json
    - dataset_dir/valid/_annotations.coco.json (note: valid, not val!)
    - dataset_dir/test/_annotations.coco.json
    - Images in dataset_dir/train/, dataset_dir/valid/, dataset_dir/test/
    """
    import shutil
    coco_path = Path(coco_dataset_dir)
    
    # Create symbolic links or copy files to match expected structure
    print("Checking dataset structure...")
    
    # RF-DETR uses 'valid' instead of 'val'
    valid_dir = coco_path / 'valid'
    if not valid_dir.exists():
        # Rename val to valid
        val_dir = coco_path / 'val'
        if val_dir.exists():
            val_dir.rename(valid_dir)
            print("✓ Renamed 'val' directory to 'valid'")
    
    # Check if _annotations.coco.json files exist
    train_ann = coco_path / 'train' / '_annotations.coco.json'
    valid_ann = coco_path / 'valid' / '_annotations.coco.json'
    test_ann = coco_path / 'test' / '_annotations.coco.json'
    
    if not train_ann.exists():
        # Create annotation files in the correct locations
        src_train = coco_path / 'annotations' / 'instances_train.json'
        src_val = coco_path / 'annotations' / 'instances_val.json'
        src_test = coco_path / 'annotations' / 'instances_test.json'
        
        if src_train.exists():
            shutil.copy(src_train, train_ann)
            shutil.copy(src_val, valid_ann)
            shutil.copy(src_test, test_ann)
            print("✓ Created _annotations.coco.json files")
    else:
        print("✓ Dataset structure is correct")
    
    return str(coco_path.absolute())

def train_landing_zone_model(
    model_size='nano',  # 'nano', 'small', 'medium', or 'base'
    dataset_dir='coco-dataset',
    output_dir='output/landing_zone_detection',
    batch_size=4,
    epochs=100,
    learning_rate=1e-4,
    num_workers=2,
    device=None,
    use_tensorboard=True,
    use_wandb=False,
    early_stopping=True,
    early_stopping_patience=10
):
    """
    Train an RF-DETR model for landing zone detection.
    
    Args:
        model_size: Size of the model ('nano', 'small', 'medium', 'base')
        dataset_dir: Path to COCO-format dataset
        output_dir: Directory to save training outputs
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        device: Device to train on (cuda/mps/cpu), auto-detected if None
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        early_stopping: Enable early stopping
        early_stopping_patience: Patience for early stopping
    """
    
    # Setup dataset directory
    dataset_dir = setup_roboflow_format(dataset_dir)
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print("=" * 70)
    print("RF-DETR TRAINING - Landing Zone Detection")
    print("=" * 70)
    print(f"Model size: {model_size.upper()}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)
    
    # Select model based on size
    model_classes = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'medium': RFDETRMedium,
        'base': RFDETRBase
    }
    
    if model_size not in model_classes:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(model_classes.keys())}")
    
    # Initialize model
    print(f"\nInitializing RF-DETR-{model_size.upper()} model...")
    # Train from scratch without pretrained weights to avoid inference mode issues
    model = model_classes[model_size](device=device, num_classes=1, pretrain_weights=None)
    
    # Create training configuration
    train_config = TrainConfig(
        dataset_file="roboflow",  # Use roboflow format
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        num_workers=num_workers,
        tensorboard=use_tensorboard,
        wandb=use_wandb,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        checkpoint_interval=5,  # Save checkpoint every 5 epochs
        use_ema=True,  # Use Exponential Moving Average
        class_names=['Landing Zone'],
        run_test=True  # Evaluate on test set after training
    )
    
    print("\nStarting training...")
    print(f"Training will save checkpoints to: {output_dir}")
    
    # Train the model
    try:
        model.train_from_config(train_config)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nCheckpoints saved to: {output_dir}")
        print(f"Best model: {output_dir}/checkpoint_best.pth")
        print("\nTo use your trained model:")
        print("  from rfdetr import RFDETRNano")
        print("  model = RFDETRNano(pretrain_weights='output/landing_zone_detection/checkpoint_best.pth')")
        print("  detections = model.predict(image, threshold=0.5)")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        raise

def main():
    """
    Main training function with default parameters.
    Modify these parameters as needed for your use case.
    """
    
    # Note: Using CPU for compatibility with all operations
    # MPS (Apple Silicon) has some unsupported operations
    train_landing_zone_model(
        model_size='nano',  # Start with nano for faster training
        dataset_dir='coco-dataset',
        output_dir='output/landing_zone_detection',
        batch_size=4,
        epochs=100,
        learning_rate=1e-4,
        num_workers=2,
        device='cpu',  # Use CPU for full compatibility
        use_tensorboard=True,
        use_wandb=False,
        early_stopping=True,
        early_stopping_patience=10
    )

if __name__ == '__main__':
    main()
