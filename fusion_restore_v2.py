"""
FusionRestore V2: Improved with Better Image Quality
====================================================
Fixes:
1. Reduces haze/glow artifacts
2. Preserves sharpness and detail
3. Better color preservation
4. Adaptive blending based on quality
"""

import cv2
import numpy as np
import sys
import os

sys.path.append('GFPGAN')
sys.path.append('Real-ESRGAN')

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class FusionRestoreV2:
    """
    Improved FusionRestore with better quality control
    
    Key Improvements:
    - Sharpness preservation
    - Color consistency
    - Detail retention
    - No over-smoothing
    """
    
    def __init__(self):
        print("🚀 Initializing FusionRestore V2 (Improved)")
        self.gfpgan = self._init_gfpgan()
        self.realesrgan = self._init_realesrgan()
        self.face_detector = self._init_face_detector()
    
    def _init_gfpgan(self):
        """Initialize GFP-GAN"""
        try:
            model_path = 'GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth'
            gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✅ GFP-GAN loaded")
            return gfpgan
        except Exception as e:
            print(f"❌ GFP-GAN failed: {e}")
            return None
    
    def _init_realesrgan(self):
        """Initialize Real-ESRGAN with better settings"""
        try:
            # Use x2plus for better quality
            model_path = 'Real-ESRGAN/weights/RealESRGAN_x2plus.pth'
            
            if not os.path.exists(model_path):
                print("⚠️  Real-ESRGAN x2plus not found")
                return None
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=2)
            
            realesrgan = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=200,           # Smaller tiles for better quality
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            print("✅ Real-ESRGAN loaded")
            return realesrgan
        except Exception as e:
            print(f"❌ Real-ESRGAN failed: {e}")
            return None
    
    def _init_face_detector(self):
        """Initialize face detector"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            print("✅ Face detector loaded")
            return detector
        except:
            return None
    
    def sharpen_image(self, image, strength=0.5):
        """
        Apply controlled sharpening to reduce haze
        
        Args:
            image: Input image
            strength: Sharpening strength (0.0-1.0)
        """
        # Create sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 9.0
        
        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original based on strength
        result = cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)
        
        return result
    
    def enhance_contrast(self, image, clip_limit=2.0):
        """
        Enhance contrast without over-saturating
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def color_transfer(self, source, target):
        """
        Transfer color statistics from source to target
        Preserves original colors, reduces color shift
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        # Transfer statistics
        result_lab = target_lab.copy()
        for i in range(3):
            result_lab[:, :, i] = ((target_lab[:, :, i] - target_mean[i]) * 
                                   (source_std[i] / target_std[i]) + source_mean[i])
        
        # Clip and convert back
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def create_better_mask(self, image, expand_ratio=0.3):
        """
        Create face mask with better edge handling
        """
        if self.face_detector is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for (x, y, w, h) in faces:
            # Expand region
            pad_x = int(w * expand_ratio)
            pad_y = int(h * expand_ratio)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            
            # Create elliptical mask for smoother blending
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Smooth mask edges
        if np.sum(mask) > 0:
            mask = cv2.GaussianBlur(mask, (31, 31), 15)  # Less blur = sharper edges
        
        return mask
    
    def detail_preserving_fusion(self, face_img, bg_img, mask, original):
        """
        Advanced fusion that preserves detail and reduces artifacts
        
        Key improvements:
        - Preserves high-frequency details
        - Reduces haze/glow
        - Maintains color consistency
        """
        # Ensure matching sizes
        if face_img.shape != bg_img.shape:
            face_img = cv2.resize(face_img, (bg_img.shape[1], bg_img.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
        
        mask_resized = cv2.resize(mask, (bg_img.shape[1], bg_img.shape[0]))
        
        # Normalize mask
        mask_norm = mask_resized.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        # Extract high-frequency details from original
        original_resized = cv2.resize(original, (bg_img.shape[1], bg_img.shape[0]))
        original_blur = cv2.GaussianBlur(original_resized, (5, 5), 0)
        details = original_resized.astype(np.float32) - original_blur.astype(np.float32)
        
        # Basic fusion
        fused = (face_img.astype(np.float32) * mask_3ch +
                bg_img.astype(np.float32) * (1.0 - mask_3ch))
        
        # Add back original details (preserves texture)
        detail_weight = 0.35  # Adjust this: 0.2-0.5
        fused = fused + (details * detail_weight)
        
        # Clip and convert
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        
        return fused
    
    def process(self, degraded_image):
        """
        Complete V2 processing pipeline
        """
        print("\n" + "="*70)
        print("Processing with FusionRestore V2 (Improved)")
        print("="*70)
        
        results = {'original': degraded_image}
        
        # Step 1: GFP-GAN face enhancement
        if self.gfpgan is not None:
            print("\n[1/5] Face Enhancement...")
            try:
                _, _, face_enhanced = self.gfpgan.enhance(
                    degraded_image,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                results['gfpgan'] = face_enhanced
                print("  ✅ Face enhanced")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                face_enhanced = degraded_image.copy()
                results['gfpgan'] = face_enhanced
        else:
            face_enhanced = degraded_image.copy()
            results['gfpgan'] = face_enhanced
        
        # Step 2: Real-ESRGAN background enhancement
        if self.realesrgan is not None:
            print("[2/5] Background Enhancement...")
            try:
                bg_enhanced, _ = self.realesrgan.enhance(degraded_image, outscale=2)
                results['esrgan'] = bg_enhanced
                print("  ✅ Background enhanced")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                bg_enhanced = cv2.resize(degraded_image, None, fx=2, fy=2,
                                        interpolation=cv2.INTER_CUBIC)
                results['esrgan'] = bg_enhanced
        else:
            bg_enhanced = cv2.resize(degraded_image, None, fx=2, fy=2,
                                    interpolation=cv2.INTER_CUBIC)
            results['esrgan'] = bg_enhanced
        
        # Step 3: Create face mask
        print("[3/5] Creating intelligent mask...")
        face_mask = self.create_better_mask(degraded_image, expand_ratio=0.25)
        results['mask'] = face_mask
        
        # Step 4: Detail-preserving fusion
        print("[4/5] Applying improved fusion...")
        fused = self.detail_preserving_fusion(
            face_enhanced, bg_enhanced, face_mask, degraded_image
        )
        
        # Step 5: Post-processing to reduce haze
        print("[5/5] Applying post-processing...")
        
        # Sharpen to reduce haze
        fused = self.sharpen_image(fused, strength=0.3)
        
        # Enhance contrast
        fused = self.enhance_contrast(fused, clip_limit=1.8)
        
        # Color correction (match original colors)
        fused = self.color_transfer(degraded_image, fused)
        
        results['fusionrestore_v2'] = fused
        
        # Also create naive combination for comparison
        naive = cv2.resize(face_enhanced, (bg_enhanced.shape[1], bg_enhanced.shape[0]))
        results['naive'] = naive
        
        print("\n✅ Processing complete!")
        
        return results


def test_improved_fusion():
    """Test improved fusion on sample images"""
    
    print("\n" + "="*70)
    print("🧪 TESTING FUSIONRESTORE V2 (IMPROVED)")
    print("="*70)
    
    fusion_v2 = FusionRestoreV2()
    
    # Test images
    input_folder = 'data/degraded_images_v2'
    output_folder = 'outputs/fusionrestore_v2_test'
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n{'#'*70}")
        print(f"# Image {idx}: {img_file}")
        print(f"{'#'*70}")
        
        input_path = os.path.join(input_folder, img_file)
        degraded = cv2.imread(input_path)
        
        # Process
        results = fusion_v2.process(degraded)
        
        # Save all stages
        base_name = os.path.splitext(img_file)[0]
        
        stages = {
            '1_original': results['original'],
            '2_gfpgan': results['gfpgan'],
            '3_esrgan': results['esrgan'],
            '4_naive': results['naive'],
            '5_fusionrestore_v2': results['fusionrestore_v2']
        }
        
        for stage_name, image in stages.items():
            output_path = os.path.join(output_folder, 
                                      f"img{idx}_{stage_name}_{base_name}.jpg")
            cv2.imwrite(output_path, image)
        
        # Create side-by-side comparison
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        titles = ['Original', 'GFP-GAN', 'Real-ESRGAN', 'Naive', 'FusionRestore V2']
        
        for i, (title, (stage_name, img)) in enumerate(zip(titles, stages.items())):
            # Resize for display
            display_img = cv2.resize(img, (400, 400))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(display_img)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle(f'FusionRestore V2: Image {idx} Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'comparison_{idx}.png'),
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n📁 Results saved in: {output_folder}")


if __name__ == "__main__":
    test_improved_fusion()
