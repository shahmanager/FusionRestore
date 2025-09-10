import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse
import time

class AdvancedImageDegrader:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        print("Advanced ImageDegrader initialized with enhanced degradation types")
    
    def add_gaussian_noise(self, image, noise_level=None):
        """Enhanced noise with variable intensity"""
        if noise_level is None:
            noise_level = random.uniform(15, 50)  # Increased range
        
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy_image.astype(np.uint8)
    
    def add_motion_blur(self, image, blur_strength=None):
        """Variable motion blur with random directions"""
        if blur_strength is None:
            blur_strength = random.choice([5, 7, 9, 11, 13])  # Stronger blur
        
        # Random blur direction
        angle = random.uniform(0, 180)
        kernel = self.get_motion_blur_kernel(blur_strength, angle)
        
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def get_motion_blur_kernel(self, size, angle):
        """Generate motion blur kernel with specific angle"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Calculate line endpoints based on angle
        dx = int(center * np.cos(np.radians(angle)))
        dy = int(center * np.sin(np.radians(angle)))
        
        # Draw line in kernel
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
        
        # Normalize
        kernel = kernel / kernel.sum()
        return kernel
    
    def add_heavy_scratches(self, image, num_scratches=None):
        """Realistic scratches with variable thickness and opacity"""
        if num_scratches is None:
            num_scratches = random.randint(5, 15)  # More scratches
        
        damaged = image.copy()
        h, w = damaged.shape[:2]
        
        for _ in range(num_scratches):
            # Random scratch parameters
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            
            # Longer, more realistic scratches
            length = random.randint(50, 200)
            angle = random.uniform(0, 360)
            
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))
            
            # Clamp to image bounds
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            
            # Variable scratch appearance
            scratch_type = random.choice(['white', 'black', 'transparent'])
            thickness = random.randint(1, 4)
            
            if scratch_type == 'white':
                color = (255, 255, 255)
            elif scratch_type == 'black':
                color = (0, 0, 0)
            else:  # transparent/semi-transparent
                color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            
            cv2.line(damaged, (start_x, start_y), (end_x, end_y), color, thickness)
        
        return damaged
    
    def add_dust_and_spots(self, image, num_spots=None):
        """Add dust spots and speckles"""
        if num_spots is None:
            num_spots = random.randint(30, 80)
        
        spotted = image.copy()
        h, w = spotted.shape[:2]
        
        for _ in range(num_spots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(1, 4)
            
            # Random spot color (dust can be light or dark)
            spot_type = random.choice(['light', 'dark', 'colored'])
            
            if spot_type == 'light':
                color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            elif spot_type == 'dark':
                color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            else:
                color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
            
            cv2.circle(spotted, (x, y), radius, color, -1)
        
        return spotted
    
    def add_color_fading(self, image, fade_type='uniform'):
        """Advanced color fading simulation"""
        h, w = image.shape[:2]
        faded = image.copy().astype(np.float32)
        
        if fade_type == 'uniform':
            # Uniform fading
            fade_factor = random.uniform(0.4, 0.8)
            faded = faded * fade_factor
            
        elif fade_type == 'corner':
            # Corner-based fading (common in old photos)
            center_x, center_y = w // 2, h // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    fade_factor = 0.3 + 0.7 * (1 - dist / max_dist)
                    faded[y, x] = faded[y, x] * fade_factor
        
        # Add yellowish tint (common in old photos)
        if random.random() > 0.5:
            faded[:, :, 0] = faded[:, :, 0] * 0.9  # Reduce blue
            faded[:, :, 1] = faded[:, :, 1] * 0.95  # Slightly reduce green
            faded[:, :, 2] = faded[:, :, 2] * 1.1   # Enhance red (yellowing)
        
        return np.clip(faded, 0, 255).astype(np.uint8)
    
    def add_vignette(self, image, intensity=None):
        """Add vignette effect (darkened edges)"""
        if intensity is None:
            intensity = random.uniform(0.3, 0.7)
        
        h, w = image.shape[:2]
        vignetted = image.copy().astype(np.float32)
        
        # Create vignette mask
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                vignette_factor = 1 - (dist / max_dist) * intensity
                vignette_factor = max(0.2, min(1.0, vignette_factor))
                vignetted[y, x] = vignetted[y, x] * vignette_factor
        
        return np.clip(vignetted, 0, 255).astype(np.uint8)
    
    def add_jpeg_artifacts(self, image, quality=None):
        """Heavy JPEG compression artifacts"""
        if quality is None:
            quality = random.randint(5, 40)  # Much lower quality
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, 1)
        return decoded_img
    
    def add_color_shift(self, image):
        """Simulate chemical color shifts"""
        shifted = image.copy()
        
        # Convert to HSV for hue shifting
        hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Random hue shift
        hue_shift = random.uniform(-30, 30)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Random saturation change
        sat_change = random.uniform(0.5, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_change, 0, 255)
        
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return shifted
    
    def degrade_image_advanced(self, image_path, output_path, degradation_level='heavy'):
        """Apply multiple random degradations with high variety"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read {image_path}")
            return False
        
        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        degraded = image.copy()
        
        # Define degradation types
        all_degradations = [
            'gaussian_noise', 'motion_blur', 'heavy_scratches', 
            'dust_and_spots', 'color_fading', 'vignette', 
            'jpeg_artifacts', 'color_shift'
        ]
        
        # Randomly select 3-6 degradation types to combine
        if degradation_level == 'light':
            num_degradations = random.randint(2, 3)
            intensities = {'low': 0.7, 'medium': 0.3, 'high': 0.0}
        elif degradation_level == 'medium':
            num_degradations = random.randint(3, 5)
            intensities = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
        else:  # heavy
            num_degradations = random.randint(4, 6)
            intensities = {'low': 0.1, 'medium': 0.4, 'high': 0.5}
        
        selected_degradations = random.sample(all_degradations, num_degradations)
        
        print(f"Applying degradations: {selected_degradations}")
        
        # Apply selected degradations
        if 'gaussian_noise' in selected_degradations:
            degraded = self.add_gaussian_noise(degraded)
        
        if 'motion_blur' in selected_degradations:
            degraded = self.add_motion_blur(degraded)
        
        if 'heavy_scratches' in selected_degradations:
            degraded = self.add_heavy_scratches(degraded)
        
        if 'dust_and_spots' in selected_degradations:
            degraded = self.add_dust_and_spots(degraded)
        
        if 'color_fading' in selected_degradations:
            fade_type = random.choice(['uniform', 'corner'])
            degraded = self.add_color_fading(degraded, fade_type)
        
        if 'vignette' in selected_degradations:
            degraded = self.add_vignette(degraded)
        
        if 'jpeg_artifacts' in selected_degradations:
            degraded = self.add_jpeg_artifacts(degraded)
        
        if 'color_shift' in selected_degradations:
            degraded = self.add_color_shift(degraded)
        
        # Save result
        success = cv2.imwrite(output_path, degraded)
        return success

def main():
    parser = argparse.ArgumentParser(description='Advanced Synthetic Image Degradation')
    parser.add_argument('--input_folder', type=str, default='data/clean_images/lfw_subset',
                        help='Path to clean images folder')
    parser.add_argument('--output_folder', type=str, default='data/degraded_images_v2',
                        help='Path to save degraded images')
    parser.add_argument('--num_images', type=int, default=250,
                        help='Number of images to process')
    parser.add_argument('--degradation_level', type=str, default='heavy',
                        choices=['light', 'medium', 'heavy'])
    
    args = parser.parse_args()
    
    print(f"=== Advanced Synthetic Degradation ===")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Processing {args.num_images} images with {args.degradation_level} degradation")
    
    # Create degrader
    degrader = AdvancedImageDegrader()
    
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
        
        if degrader.degrade_image_advanced(str(image_path), output_path, 
                                           args.degradation_level):
            processed += 1
        else:
            failed += 1
        
        if processed % 25 == 0 and processed > 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed} images... ({processed/elapsed:.1f} images/sec)")
    
    total_time = time.time() - start_time
    print(f"\n=== Advanced Degradation Complete ===")
    print(f"Successfully processed: {processed} images")
    print(f"Failed: {failed} images")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()
