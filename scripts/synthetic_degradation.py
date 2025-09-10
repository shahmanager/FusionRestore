import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse
import time

class ImageDegrader:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        print("ImageDegrader initialized with seed:", seed)
    
    def add_gaussian_noise(self, image, noise_level=None):
        """Add realistic film grain noise"""
        if noise_level is None:
            noise_level = random.uniform(8, 25)
        
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_motion_blur(self, image, blur_type='horizontal'):
        """Simulate camera shake or age-related blurring"""
        kernel_size = random.choice([3, 5, 7])
        
        if blur_type == 'horizontal':
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        elif blur_type == 'vertical':
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
        else:  # diagonal
            kernel = np.eye(kernel_size)
        
        kernel = kernel / kernel.sum()
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def add_fading(self, image, fade_factor=None):
        """Simulate age-related color fading"""
        if fade_factor is None:
            fade_factor = random.uniform(0.65, 0.9)
        
        faded = cv2.convertScaleAbs(image, alpha=fade_factor, beta=10)
        return faded
    
    def add_jpeg_artifacts(self, image, quality=None):
        """Simulate compression artifacts"""
        if quality is None:
            quality = random.randint(25, 60)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, 1)
        return decoded_img
    
    def add_scratches(self, image, num_scratches=None):
        """Add realistic physical damage"""
        if num_scratches is None:
            num_scratches = random.randint(1, 5)
        
        damaged = image.copy()
        h, w = damaged.shape[:2]
        
        for _ in range(num_scratches):
            # Random scratch parameters
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            end_x = start_x + random.randint(-100, 100)
            end_y = start_y + random.randint(-20, 20)
            
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            
            # Draw scratch (white, black, or gray)
            color = random.choice([(255, 255, 255), (0, 0, 0), (128, 128, 128)])
            thickness = random.randint(1, 2)
            cv2.line(damaged, (start_x, start_y), (end_x, end_y), color, thickness)
        
        return damaged
    
    def degrade_image_realistic(self, image_path, output_path, degradation_level='medium'):
        """Apply realistic combination of degradations"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read {image_path}")
            return False
        
        # Resize if too large (for performance)
        h, w = image.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        degraded = image.copy()
        
        # Apply degradations based on level
        if degradation_level == 'light':
            prob_noise, prob_blur, prob_fade = 0.4, 0.2, 0.3
        elif degradation_level == 'medium':
            prob_noise, prob_blur, prob_fade = 0.7, 0.4, 0.6
        else:  # heavy
            prob_noise, prob_blur, prob_fade = 0.9, 0.6, 0.8
        
        # Apply random degradations
        if random.random() < prob_noise:
            degraded = self.add_gaussian_noise(degraded)
        
        if random.random() < prob_blur:
            blur_type = random.choice(['horizontal', 'vertical', 'diagonal'])
            degraded = self.add_motion_blur(degraded, blur_type)
        
        if random.random() < prob_fade:
            degraded = self.add_fading(degraded)
        
        if random.random() < 0.3:  # Sometimes add compression
            degraded = self.add_jpeg_artifacts(degraded)
        
        if random.random() < 0.2:  # Occasionally add scratches
            degraded = self.add_scratches(degraded)
        
        # Save result
        success = cv2.imwrite(output_path, degraded)
        return success

def main():
    parser = argparse.ArgumentParser(description='Synthetic Image Degradation for FusionRestore')
    parser.add_argument('--input_folder', type=str, default='data/clean_images/lfw_subset',
                        help='Path to clean images folder')
    parser.add_argument('--output_folder', type=str, default='data/degraded_images',
                        help='Path to save degraded images')
    parser.add_argument('--num_images', type=int, default=250,
                        help='Number of images to process')
    parser.add_argument('--degradation_level', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'])
    
    args = parser.parse_args()
    
    print(f"Starting synthetic degradation...")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Processing {args.num_images} images with {args.degradation_level} degradation")
    
    # Create degrader
    degrader = ImageDegrader()
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Get list of images
    input_path = Path(args.input_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images in input folder")
    
    # Process images
    processed = 0
    failed = 0
    start_time = time.time()
    
    for image_path in image_files[:args.num_images]:
        output_path = os.path.join(args.output_folder, image_path.name)
        
        if degrader.degrade_image_realistic(str(image_path), output_path, 
                                            args.degradation_level):
            processed += 1
        else:
            failed += 1
        
        if processed % 25 == 0 and processed > 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed} images... ({processed/elapsed:.1f} images/sec)")
    
    total_time = time.time() - start_time
    print(f"\n=== Degradation Complete ===")
    print(f"Successfully processed: {processed} images")
    print(f"Failed: {failed} images")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/max(processed,1):.2f} seconds")
    print(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()
