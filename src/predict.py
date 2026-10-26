"""Running the trained U-Net on full DRIVE images.

    .venv/bin/python -m src.predict --checkpoint results/unet_best.pt

saves the probability maps of the validation and test images to results/probs/
(next to the checkpoint). train_unet.py already does this at the end of training.
"""

import argparse
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.drive import TEST_IDS, VAL_IDS, load_sample

# The encoder was pre-trained on ImageNet, so inputs are normalised with ImageNet statistics.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint, device):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)  # weights come from the checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return model.to(device).eval()


def to_tensor(image):
    """RGB uint8 (H, W, 3) -> normalised float tensor (3, H, W)."""
    x = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def predict_proba(model, image, device):
    """Return the vessel probability map (H, W) float32 in [0, 1] for one full RGB image.

    The U-Net downsamples 5 times, so the input is padded to a multiple of 32
    (584x565 -> 608x576) and the output is cropped back to the original size.
    The raw image is padded with black pixels, like the dark area that already
    surrounds the field of view.
    """
    model.eval()
    h, w = image.shape[:2]
    pad_h, pad_w = (-h) % 32, (-w) % 32
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    x = to_tensor(padded).unsqueeze(0).to(device)
    logits = model(x)[0, 0, :h, :w]
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def binarize(prob, threshold=0.5):
    """Probability map -> boolean vessel mask."""
    return prob >= threshold


def save_probability_maps(model, device, out_dir):
    """Save the probability maps of the validation and test images as .npy files.

    Saving them once lets the threshold and the morphological post-processing be
    tuned later (on the validation maps only) without running the network again.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in [("val", VAL_IDS), ("test", TEST_IDS)]:
        for image_id in ids:
            image, _, _ = load_sample(image_id)
            np.save(out_dir / f"{split}_{image_id:02d}.npy", predict_proba(model, image, device))
    print(f"saved {len(VAL_IDS)} validation + {len(TEST_IDS)} test probability maps to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    save_probability_maps(load_model(args.checkpoint, device), device, args.checkpoint.parent / "probs")
