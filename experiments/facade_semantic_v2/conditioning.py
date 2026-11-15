"""Human-label canonicalization and seven explicit spatial control channels."""
import cv2
import numpy as np
from PIL import Image

CLASSES = ('background', 'wall', 'window', 'door', 'balcony', 'plant')
CHANNELS = ('edges',) + CLASSES
IRF_RGB = np.array([[0, 0, 0], [128, 0, 128], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]], np.uint8)
IRF_TO_CANONICAL = np.array([0, 5, 1, 2, 3, 4], np.uint8)
DISPLAY_RGB = np.array([[0, 0, 0], [180, 180, 180], [40, 160, 255], [255, 150, 30], [160, 80, 220], [70, 160, 60]], np.uint8)


def cmp_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.min() < 0 or labels.max() > 12:
        raise ValueError('Expected original CMP class indices 0..12')
    mapped = np.ones(labels.shape, np.uint8)
    mapped[np.isin(labels, [0, 1])] = 0
    mapped[np.isin(labels, [3, 8, 12])] = 2
    mapped[labels == 4] = 3
    mapped[labels == 7] = 4
    return mapped


def irf_labels(image):
    colours = np.asarray(image.convert('RGB'))
    matches = (colours[:, :, None, :] == IRF_RGB[None, None]).all(axis=-1)
    if not matches.any(axis=-1).all():
        raise ValueError('Unexpected IRFs label colours; predictions are not accepted as annotations')
    return IRF_TO_CANONICAL[matches.argmax(axis=-1)]


def control_channels(labels):
    labels = np.asarray(labels, np.uint8)
    if labels.ndim != 2 or labels.max() >= len(CLASSES):
        raise ValueError('Expected canonical facade labels 0..5')
    one_hot = (labels[None] == np.arange(len(CLASSES))[:, None, None]).astype(np.float32)
    edge = np.zeros(labels.shape, np.uint8)
    for index in np.unique(labels):
        edge |= cv2.Canny((labels == index).astype(np.uint8) * 255, 100, 200)
    return np.concatenate((edge[None].astype(np.float32) / 255, one_hot), axis=0)


def letterbox(photo, labels, size=512):
    if photo.size != tuple(labels.shape[::-1]):
        raise ValueError('Unaligned photo and annotation')
    width, height = photo.size
    scale = min(size / width, size / height)
    new = (max(1, round(width * scale)), max(1, round(height * scale)))
    x, y = (size - new[0]) // 2, (size - new[1]) // 2
    canvas = Image.new('RGB', (size, size), (127, 127, 127))
    canvas.paste(photo.resize(new, Image.Resampling.LANCZOS), (x, y))
    mask = np.zeros((size, size), np.uint8)
    mask[y:y + new[1], x:x + new[0]] = np.asarray(Image.fromarray(labels).resize(new, Image.Resampling.NEAREST))
    valid = np.zeros((size, size), np.float32)
    valid[y:y + new[1], x:x + new[0]] = 1
    return canvas, mask, valid


def facade_crop(photo, labels, rng=None, strip_probability=0):
    """Aligned photo/GT crops; storey strips imitate wide production wall inputs."""
    facade = np.isin(labels, [1, 2, 3, 4])
    yy, xx = np.where(facade)
    if not len(xx):
        raise ValueError('No facade region in annotation')
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    if rng is not None and rng.random() < strip_probability:
        count, _, stats, centers = cv2.connectedComponentsWithStats((labels == 2).astype(np.uint8))
        components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 25]
        if components:
            chosen = int(rng.choice(components))
            median_height = float(np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in components]))
            row = [i for i in components if abs(centers[i, 1] - centers[chosen, 1]) <= median_height * .65]
            lo = min(stats[i, cv2.CC_STAT_TOP] for i in row)
            hi = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in row)
            y0 = max(y0, int(lo - .45 * median_height))
            y1 = min(y1, int(hi + .65 * median_height))
            # Keep full door bodies intersecting this storey's windows.
            n, _, doors, _ = cv2.connectedComponentsWithStats((labels == 3).astype(np.uint8))
            for i in range(1, n):
                top, height = doors[i, cv2.CC_STAT_TOP], doors[i, cv2.CC_STAT_HEIGHT]
                if top < y1 and top + height > y0:
                    y0, y1 = min(y0, top), max(y1, top + height)
    box = (x0, max(0, y0), x1, min(labels.shape[0], y1))
    return photo.crop(box), labels[box[1]:box[3], box[0]:box[2]].copy()
