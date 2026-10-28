import cv2
import numpy as np

def test_dt_labels():
    h, w = 256, 256
    wall = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(wall, (20, 20), (230, 230), 1, 4)
    cv2.line(wall, (120, 20), (120, 230), 1, 4)

    envelope = np.zeros((h, w), dtype=np.uint8)
    envelope[20:230, 20:230] = 1

    labeled = np.zeros((h, w), dtype=np.int32)
    labeled[25:225, 25:115] = 1
    labeled[25:50, 125:160] = 2

    mask_interior = (envelope == 1) & (wall == 0)

    # Fast 8-connected flood/dilation with cv2.distanceTransformWithLabels or BFS queue
    # An open space BFS or connected components of (mask_interior & (labeled == 0)):
    unassigned = mask_interior & (labeled == 0)
    num_u, labels_u = cv2.connectedComponents(unassigned.astype(np.uint8))
    print(f"Number of unassigned connected components: {num_u - 1}")

    expanded = labeled.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # For each unassigned component:
    for u_lbl in range(1, num_u):
        comp = (labels_u == u_lbl).astype(np.uint8)
        # Find adjacent labeled rooms
        dilated_comp = cv2.dilate(comp, kernel, iterations=1)
        touching_labels = expanded[(dilated_comp > 0) & (expanded > 0)]
        if len(touching_labels) > 0:
            # Most common touching room label
            from collections import Counter
            best_lbl = Counter(touching_labels).most_common(1)[0][0]
            expanded[comp > 0] = best_lbl
            print(f"Component {u_lbl} (area {comp.sum()} px) absorbed into Room {best_lbl}!")

    print(f"After component absorption: Room 1 = {(expanded == 1).sum()}, Room 2 = {(expanded == 2).sum()}")
    print(f"Remaining unassigned: {((mask_interior) & (expanded == 0)).sum()}")

if __name__ == "__main__":
    test_dt_labels()
