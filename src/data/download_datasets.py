"""
Automated Dataset Ingestion Script for ArchiGen 3D.
Downloads and verifies the CMP Facades dataset (Berkeley Pix2Pix benchmark).
Official reference: Section 4.2 of Guide_Projet_Deep_Learning_Avance.pdf
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

FACADES_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"

def download_and_extract_facades(dest_dir: str = "data/raw"):
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    tar_path = dest_path / "facades.tar.gz"
    extracted_folder = dest_path / "facades"

    if extracted_folder.exists() and any(extracted_folder.iterdir()):
        print(f"[INFO] Facades dataset already exists at: {extracted_folder}")
        return

    print(f"[INFO] Downloading Facades dataset from {FACADES_URL} ...")
    try:
        urllib.request.urlretrieve(FACADES_URL, tar_path)
        print(f"[SUCCESS] Download completed. Extracting archive ...")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_path)
            
        print(f"[SUCCESS] Extraction completed at: {extracted_folder}")
        
        # Clean up archive file
        if tar_path.exists():
            tar_path.unlink()
            print("[INFO] Cleaned up temporary tar.gz archive.")
            
    except Exception as e:
        print(f"[ERROR] Failed to download or extract facades dataset: {e}")
        sys.exit(1)

    # Verification of splits
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = extracted_folder / split
        if split_path.exists():
            count = len(list(split_path.glob("*.jpg")))
            print(f"  -> Split '{split}': {count} paired images found.")

if __name__ == "__main__":
    download_and_extract_facades()
