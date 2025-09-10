import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def validate_degradation(clean_folder, degraded_folder, num_samples=4):
    """Compare clean vs degraded images"""
    
    clean_images = list(Path(clean_folder).glob('*.jpg'))[:num_samples]
    
    if len(clean_images) == 0:
        print("No images found in clean folder!")
        return
    
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
    fig.suptitle('Clean vs Degraded Images - Validation', fontsize=16)
    
    for i, clean_path in enumerate(clean_images):
        degraded_path = Path(degraded_folder) / clean_path.name
        
        if degraded_path.exists():
            # Load images
            clean_img = cv2.imread(str(clean_path))
            degraded_img = cv2.imread(str(degraded_path))
            
            # Convert BGR to RGB for matplotlib
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[0, i].imshow(clean_img)
            axes[0, i].set_title(f'Clean {i+1}', fontsize=12)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(degraded_img)
            axes[1, i].set_title(f'Degraded {i+1}', fontsize=12)
            axes[1, i].axis('off')
        else:
            print(f"Degraded version not found for {clean_path.name}")
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Validation complete! Check 'validation_results.png' for visual comparison.")

if __name__ == "__main__":
    validate_degradation("data/clean_images/lfw_subset", "data/degraded_images_v2")
