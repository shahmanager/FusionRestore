import zipfile
import os
import shutil

def extract_subset_images(zip_filename, output_dir, max_images=250):
    """Extract first N images from the downloaded LFW dataset"""
    
    print(f"Extracting subset of {max_images} images from {zip_filename}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Get all image files from the zip
            all_files = [f for f in zip_ref.namelist() 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Found {len(all_files)} total images in dataset")
            
            # Extract only the first max_images
            for i, file_path in enumerate(all_files[:max_images]):
                try:
                    # Extract to temporary location
                    zip_ref.extract(file_path, "temp_extract")
                    
                    # Move to output folder with simple naming
                    source_path = os.path.join("temp_extract", file_path)
                    dest_filename = f"lfw_{i:04d}.jpg"
                    dest_path = os.path.join(output_dir, dest_filename)
                    
                    # Ensure destination folder exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    shutil.move(source_path, dest_path)
                    extracted_count += 1
                    
                    if extracted_count % 50 == 0:
                        print(f"Extracted {extracted_count} images...")
                        
                except Exception as e:
                    print(f"Error extracting {file_path}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Could not find {zip_filename}")
        print("Make sure the file is in your current directory")
        return 0
    
    # Clean up temporary directory
    if os.path.exists("temp_extract"):
        shutil.rmtree("temp_extract")
    
    print(f"\n✅ Successfully extracted {extracted_count} images to {output_dir}")
    return extracted_count


if __name__ == "__main__":
    # ⚡ Update with the actual Kaggle download filename
    zip_file = "lfw-dataset.zip"   # Kaggle will save it with this name
    output_folder = "data/lfw_subset"
    
    extract_subset_images(zip_file, output_folder, max_images=250)
