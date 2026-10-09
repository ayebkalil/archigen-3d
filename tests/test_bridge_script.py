import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.vectorizer import FloorplanVectorizer
from src.bridge.facade_bridge import FacadeBridge

def run_test():
    ds = CubiCasaDataset(split='val', img_size=512, augment=False)
    sample = ds[0]
    struct_mask = sample['struct_mask'].numpy()
    room_mask = sample['room_mask'].numpy()

    vectorizer = FloorplanVectorizer()
    layout = vectorizer.vectorize(
        mask=sample['mask'].numpy(),
        struct_mask=struct_mask,
        room_mask=room_mask,
        scale_override_m_per_px=0.03
    )

    walls = layout.get("walls", [])
    rooms = layout.get("rooms", [])
    ext_walls = [w for w in walls if w.get("is_exterior", False)]
    print(f"Layout extracted: {len(walls)} walls, {len(rooms)} rooms")
    print(f"Exterior walls: {len(ext_walls)}")

    bridge = FacadeBridge()
    result = bridge.process_layout_facades(layout, num_floors=2, output_dir="models/samples/facades_gt_test")
    print(f"Front wall chosen: {result['front_wall_id']}")
    print(f"Processed {result['exterior_walls_count']} exterior walls")

    for w_id, fac_res in result['facade_results'].items():
        print(f"Wall {w_id}: length={fac_res['wall_length_m']}m, height={fac_res['wall_height_m']}m, is_front={fac_res['is_front']}")

if __name__ == "__main__":
    run_test()
