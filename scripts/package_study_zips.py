import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def package_zips():
    print("=" * 80)
    print("PACKAGING STUDY ZIP BUNDLES OUTSIDE OUTPUTS/ AND OUTSIDE ONEDRIVE")
    print("=" * 80)

    # Destination directory outside outputs/ and outside OneDrive
    # c:\Users\ayebk\OneDrive -> outside is c:\Users\ayebk\blind_study_bundles
    target_dir = Path("C:/Users/ayebk/blind_study_bundles")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target distribution folder: {target_dir}")

    source_base = PROJECT_ROOT / "outputs" / "blind_study" / "shared_bundles"

    studies = [
        ("study_1", target_dir / "study_1_rater_bundle.zip"),
        ("study_2", target_dir / "study_2_rater_bundle.zip")
    ]

    for study_id, zip_path in studies:
        src_dir = source_base / study_id
        assert src_dir.exists(), f"Source directory missing: {src_dir}"

        print(f"\nBuilding {zip_path.name} from {src_dir}...")
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    full_p = Path(root) / f
                    arcname = full_p.relative_to(src_dir)
                    zf.write(full_p, arcname)

        print(f"  Created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

        # Verification of zip contents
        print(f"  Verifying entries in {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            print(f"    Total files in zip: {len(names)}")
            html_entries = [n for n in names if n.endswith(".html")]
            png_entries = [n for n in names if n.endswith(".png")]
            other_entries = [n for n in names if not (n.endswith(".html") or n.endswith(".png"))]

            assert len(html_entries) == 1, f"Expected 1 html entry, got {html_entries}"
            assert len(png_entries) == 64, f"Expected 64 png entries, got {len(png_entries)}"
            assert len(other_entries) == 0, f"Unexpected non-media entries: {other_entries}"

            # Check for ANY key files or secret terms in filenames
            for n in names:
                n_lower = n.lower()
                assert "key" not in n_lower, f"Zip contains key file: {n}"
                assert "manifest" not in n_lower, f"Zip contains manifest: {n}"
                assert "engine" not in n_lower, f"Zip contains engine term: {n}"
                assert "random" not in n_lower, f"Zip contains random term: {n}"
                assert "baseline" not in n_lower, f"Zip contains baseline term: {n}"
                assert n.endswith((".html", ".png")), f"Zip contains unexpected file: {n}"

        print(f"  CONFIRMED: {zip_path.name} contains ONLY index.html and 64 anonymous PNG images. Zero keys present.")

    print("\n" + "=" * 80)
    print("ZIP PACKAGING COMPLETED AND VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    package_zips()
