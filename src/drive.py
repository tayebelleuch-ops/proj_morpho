"""Loading the DRIVE dataset (paths, image readers, train/validation/test split)."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "data" / "DRIVE"
RESULTS = ROOT / "results"

# DRIVE numbering: training images are 21-40, test images are 01-20.
TRAIN_IDS = list(range(21, 41))
TEST_IDS = list(range(1, 21))

# 4 training images held out for validation (model selection and threshold choice).
# Fixed explicitly, spread over the training set, so every pipeline uses the same split.
VAL_IDS = [24, 29, 34, 39]
FIT_IDS = [i for i in TRAIN_IDS if i not in VAL_IDS]


def read_rgb(path):
    """Read a .tif fundus image as RGB uint8 (H, W, 3).

    The DRIVE .tif files are LZW-compressed, which skimage can't read without
    imagecodecs, so OpenCV is used. OpenCV returns BGR, hence the conversion.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_binary(path):
    """Read a .gif mask or annotation as a boolean array.

    Uses > 0 because test/2nd_manual stores vessels as 1, the other files as 255.
    """
    return np.array(Image.open(path)) > 0


def load_sample(image_id):
    """Return (image RGB, field-of-view mask, 1st-observer vessel annotation) for one image id."""
    if image_id in TRAIN_IDS:
        split, stem = "training", f"{image_id}_training"
    else:
        split, stem = "test", f"{image_id:02d}_test"
    image = read_rgb(DRIVE / split / "images" / f"{stem}.tif")
    fov = read_binary(DRIVE / split / "mask" / f"{stem}_mask.gif")
    vessels = read_binary(DRIVE / split / "1st_manual" / f"{image_id:02d}_manual1.gif")
    return image, fov, vessels
