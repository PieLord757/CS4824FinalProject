import cv2
import numpy as np
import os
from pathlib import Path
import shutil

class LandingZoneDetector:
    def __init__(self, 
                 frames_dir='frames',
                 shaded_dir='frames-shaded',
                 output_dir='labeled-data',
                 brightness_threshold=200,
                 min_area=100):
        """
        Initialize the landing zone detector.
        
        Args:
            frames_dir: Directory containing original frames
            shaded_dir: Directory containing shaded frames with white landing zones
            output_dir: Directory to save labeled data
            brightness_threshold: Threshold for detecting white regions (0-255)
            min_area: Minimum area in pixels to consider a valid landing zone
        """
        self.frames_dir = Path(frames_dir)
        self.shaded_dir = Path(shaded_dir)
        self.output_dir = Path(output_dir)
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.preview_dir = self.output_dir / 'preview'
        
    def setup_directories(self):
        """Create or clean output directories."""
        # Remove existing output directory if it exists
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        # Create fresh directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_landing_zones(self, shaded_image):
        """
        Detect landing zones in a shaded image.
        
        Args:
            shaded_image: Grayscale or BGR image with white landing zones
            
        Returns:
            List of bounding boxes in YOLO format (class_id, x_center, y_center, width, height)
        """
        # Convert to grayscale if needed
        if len(shaded_image.shape) == 3:
            gray = cv2.cvtColor(shaded_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = shaded_image
            
        # Apply threshold to detect white regions
        _, binary = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Process contours and create bounding boxes
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_width = w / width
            norm_height = h / height
            
            bounding_boxes.append({
                'class_id': 0,
                'x_center': x_center,
                'y_center': y_center,
                'width': norm_width,
                'height': norm_height,
                'area': area,
                'pixel_coords': (x, y, w, h)  # For preview visualization
            })
        
        # Sort by area (largest first) and return only the largest
        if bounding_boxes:
            bounding_boxes.sort(key=lambda x: x['area'], reverse=True)
            return [bounding_boxes[0]]  # Return only the largest landing zone
        
        return []
    
    def draw_bounding_box(self, image, bbox):
        """Draw bounding box on image for preview."""
        height, width = image.shape[:2]
        x, y, w, h = bbox['pixel_coords']
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = "Landing Zone"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
        
        return image
    
    def save_yolo_annotation(self, bboxes, output_path):
        """Save bounding boxes in YOLO format."""
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                # YOLO format: class_id x_center y_center width height
                line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                f.write(line)
    
    def process_all_frames(self):
        """Process all frames and create labeled dataset."""
        self.setup_directories()
        
        # Get list of shaded images
        shaded_images = sorted(self.shaded_dir.glob('*.png'))
        
        stats = {
            'total': len(shaded_images),
            'with_zones': 0,
            'without_zones': 0
        }
        
        print(f"Processing {stats['total']} frames...")
        
        for shaded_path in shaded_images:
            frame_name = shaded_path.name
            original_path = self.frames_dir / frame_name
            
            if not original_path.exists():
                print(f"Warning: Original frame not found for {frame_name}")
                continue
            
            # Read images
            shaded_img = cv2.imread(str(shaded_path))
            original_img = cv2.imread(str(original_path))
            
            # Skip if images couldn't be loaded (corrupted files)
            if shaded_img is None or original_img is None:
                print(f"Warning: Could not load {frame_name}, skipping (corrupted file)...")
                continue
            
            # Detect landing zones
            bboxes = self.detect_landing_zones(shaded_img)
            
            # Update statistics
            if bboxes:
                stats['with_zones'] += 1
            else:
                stats['without_zones'] += 1
            
            # Copy original image to output
            output_image_path = self.images_dir / frame_name
            shutil.copy(original_path, output_image_path)
            
            # Save annotation
            annotation_path = self.labels_dir / frame_name.replace('.png', '.txt')
            self.save_yolo_annotation(bboxes, annotation_path)
            
            # Create preview with bounding boxes
            preview_img = original_img.copy()
            for bbox in bboxes:
                preview_img = self.draw_bounding_box(preview_img, bbox)
            
            preview_path = self.preview_dir / frame_name
            cv2.imwrite(str(preview_path), preview_img)
            
            print(f"Processed {frame_name}: {len(bboxes)} landing zone(s) detected")
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total frames processed: {stats['total']}")
        print(f"Frames with landing zones: {stats['with_zones']}")
        print(f"Frames without landing zones: {stats['without_zones']}")
        print(f"\nOutput saved to: {self.output_dir}")
        print(f"  - Images: {self.images_dir}")
        print(f"  - Labels: {self.labels_dir}")
        print(f"  - Preview: {self.preview_dir}")

def main():
    # Initialize detector with default parameters
    detector = LandingZoneDetector(
        frames_dir='frames',
        shaded_dir='frames-shaded',
        output_dir='labeled-data',
        brightness_threshold=150,  # Lowered from 200 to detect landing zones in new data
        min_area=100
    )
    
    # Process all frames
    detector.process_all_frames()

if __name__ == '__main__':
    main()
