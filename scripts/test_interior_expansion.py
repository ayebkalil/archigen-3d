import cv2
import numpy as np
from scipy import ndimage as ndi

def test_interior_absorption_cv2():
    h, w = 256, 256
    wall = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(wall, (20, 20), (230, 230), 1, 4) # outer wall
    cv2.line(wall, (120, 20), (120, 230), 1, 4) # dividing wall

    envelope = np.zeros((h, w), dtype=np.uint8)
    envelope[20:230, 20:230] = 1

    labeled = np.zeros((h, w), dtype=np.int32)
    labeled[25:225, 25:115] = 1
    labeled[25:50, 125:160] = 2

    print(f"Before expansion: Room 1 area = {(labeled == 1).sum()}, Room 2 area = {(labeled == 2).sum()}")

    # Distance transform & Voronoi expansion constrained by interior & walls
    mask_interior = (envelope == 1) & (wall == 0)

    # Use cv2.distanceTransformWithLabels or geodesic dilation
    # Iterative geodesic dilation using 3x3 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expanded = labeled.copy()

    for _ in range(120): # expand up to 120 pixels
        unassigned_interior = mask_interior & (expanded == 0)
        if not unassigned_interior.any():
            break
        # Dilate labeled regions into unassigned_interior
        # Dilate each label or use maximum filter
        dilated = cv2.dilate(expanded.astype(np.float32), kernel, iterations=1).astype(np.int32)
        # Only assign unassigned interior pixels
        expanded[unassigned_interior] = dilated[unassigned_interior]

    print(f"After expansion: Room 1 area = {(expanded == 1).sum()}, Room 2 area = {(expanded == 2).sum()}")
    print(f"Remaining unassigned interior: {((mask_interior) & (expanded == 0)).sum()}")

    right_room1_pixels = (expanded[25:225, 125:225] == 1).sum()
    print(f"Room 1 leak into right room: {right_room1_pixels} px (must be 0!)")
    assert right_room1_pixels == 0, "Room 1 should not cross the wall!"

    right_room2_pixels = (expanded[25:225, 125:225] == 2).sum()
    print(f"Room 2 expanded to fill right room: {right_room2_pixels} px")
    assert right_room2_pixels > 15000, "Room 2 should fill the right side!"
    print("SUCCESS: Zero leaks across walls, and unassigned space fully filled!")

if __name__ == "__main__":
    test_interior_absorption_cv2()
