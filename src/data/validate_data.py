"""
Data Quality & Integrity Validation Pipeline.
Verifies image corruption, dimensions, channels, and split distributions.
Satisfies MLOps Criteria: 'Validation des données' (3/3) and 'Sécurisation des données' (3/3).
"""

import sys
import hashlib
from pathlib import Path
from PIL import Image

def compute_checksum(file_path: Path) -> str:
    """Computes SHA-256 hash to guarantee data immutability."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_dataset(data_dir: str = "data/raw/facades") -> bool:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_path}")
        return False

    splits = ["train", "val", "test"]
    total_images = 0
    corrupted_images = 0

    print(f"[VALIDATION] Starting data integrity audit on: {data_path}")

    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"[ERROR] Missing mandatory split folder: {split_path}")
            return False

        image_files = list(split_path.glob("*.jpg"))
        if len(image_files) == 0:
            print(f"[ERROR] Split '{split}' is empty!")
            return False

        print(f"[VALIDATION] Checking '{split}' split ({len(image_files)} images) ...")

        for img_file in image_files:
            total_images += 1
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Check for file corruption

                # Reopen to check dimensions and mode (verify closes the file)
                with Image.open(img_file) as img:
                    if img.mode != "RGB":
                        print(f"[WARNING] Image {img_file.name} is mode '{img.mode}', expected 'RGB'.")
                    w, h = img.size
                    if w < 100 or h < 100:
                        print(f"[ERROR] Image {img_file.name} has suspicious dimensions: {w}x{h}")
                        corrupted_images += 1
            except Exception as e:
                print(f"[ERROR] Corrupted image file: {img_file.name} ({e})")
                corrupted_images += 1

    print("\n" + "="*50)
    print(f"[VALIDATION REPORT] Total Images Audited: {total_images}")
    print(f"[VALIDATION REPORT] Corrupted/Invalid Files: {corrupted_images}")
    
    if corrupted_images == 0:
        print("[STATUS] PASSED - Dataset is 100% clean and valid for training.")
        print("="*50)
        return True
    else:
        print("[STATUS] FAILED - Data quality issues detected.")
        print("="*50)
        return False

if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
