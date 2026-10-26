"""Train the U-Net (pipeline B) on the DRIVE training images.

    .venv/bin/python -m src.train_unet                                   # full training -> results/
    .venv/bin/python -m src.train_unet --epochs 2 --out results/smoke   # quick check

- Model: smp.Unet with a ResNet-34 encoder pre-trained on ImageNet.
- Training data: random 256x256 patches from the 16 training images not in VAL_IDS,
  with flips, rotation and a small brightness change.
- Loss: BCE + Dice, computed only on pixels inside the field-of-view (FOV) mask.
- Validation: the 4 images in VAL_IDS, never trained on. After each epoch the model is
  run on these full images and the checkpoint with the best mean Dice (threshold 0.5,
  inside the FOV) is kept as <out>/unet_best.pt.
- At the end, the probability maps of the validation and test images are saved to
  <out>/probs/ so the threshold and the morphological post-processing can be tuned later
  (on the validation maps only) without running the network again.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from src.drive import FIT_IDS, RESULTS, VAL_IDS, load_sample
from src.metrics import dice_in_fov
from src.predict import binarize, load_model, predict_proba, save_probability_maps, to_tensor

PATCH = 256


class PatchDataset(Dataset):
    """Random augmented patches. One epoch = `patches_per_image` patches from each image."""

    def __init__(self, image_ids, patches_per_image, seed):
        self.samples = [load_sample(i) for i in image_ids]  # all images fit in memory
        self.patches_per_image = patches_per_image
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        image, fov, vessels = self.samples[index % len(self.samples)]
        rng = self.rng

        # Random 256x256 crop.
        h, w = fov.shape
        y = rng.integers(0, h - PATCH + 1)
        x = rng.integers(0, w - PATCH + 1)
        image = image[y:y + PATCH, x:x + PATCH]
        fov = fov[y:y + PATCH, x:x + PATCH].astype(np.uint8)
        vessels = vessels[y:y + PATCH, x:x + PATCH].astype(np.uint8)

        # Random flips.
        if rng.random() < 0.5:
            image, fov, vessels = image[:, ::-1], fov[:, ::-1], vessels[:, ::-1]
        if rng.random() < 0.5:
            image, fov, vessels = image[::-1], fov[::-1], vessels[::-1]

        # Random rotation by any angle. The corners that become empty get fov = 0,
        # so they are ignored by the loss. Masks use nearest-neighbour to stay binary.
        rot = cv2.getRotationMatrix2D((PATCH / 2, PATCH / 2), rng.uniform(-180, 180), 1.0)
        image = cv2.warpAffine(np.ascontiguousarray(image), rot, (PATCH, PATCH), flags=cv2.INTER_LINEAR)
        fov = cv2.warpAffine(np.ascontiguousarray(fov), rot, (PATCH, PATCH), flags=cv2.INTER_NEAREST)
        vessels = cv2.warpAffine(np.ascontiguousarray(vessels), rot, (PATCH, PATCH), flags=cv2.INTER_NEAREST)

        # Small brightness/contrast change: x * [0.9, 1.1] + [-10, 10] grey levels.
        image = np.clip(image * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10), 0, 255).astype(np.uint8)

        return (to_tensor(image),
                torch.from_numpy(vessels[None].astype(np.float32)),
                torch.from_numpy(fov[None].astype(np.float32)))


def masked_bce_dice_loss(logits, target, fov):
    """BCE + soft Dice, both restricted to FOV pixels (fov = 1 inside, 0 outside)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * fov).sum() / fov.sum().clamp(min=1)

    # Soft Dice over the whole batch at once, so patches with few FOV pixels can't blow it up.
    prob = torch.sigmoid(logits) * fov
    target = target * fov
    dice = (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + (1 - dice)


def validate(model, val_samples, device):
    """Mean Dice over the full validation images, threshold 0.5, inside the FOV."""
    scores = [dice_in_fov(binarize(predict_proba(model, image, device)), vessels, fov)
              for image, fov, vessels in val_samples]
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device: {device} ({name})")
    print(f"training images: {FIT_IDS}")
    print(f"validation images: {VAL_IDS}")

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "unet_best.pt"

    train_loader = DataLoader(PatchDataset(FIT_IDS, args.patches_per_image, args.seed),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_samples = [load_sample(i) for i in VAL_IDS]

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = -1.0
    with open(args.out / "unet_train_log.csv", "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_loss", "val_dice", "lr", "seconds"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for x, target, fov in train_loader:
                x, target, fov = x.to(device), target.to(device), fov.to(device)
                loss = masked_bce_dice_loss(model(x), target, fov)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            val_dice = validate(model, val_samples, device)
            seconds = time.time() - start
            log.writerow([epoch, f"{np.mean(losses):.4f}", f"{val_dice:.4f}", f"{lr:.2e}", f"{seconds:.1f}"])
            log_file.flush()

            marker = ""
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice,
                            "val_ids": VAL_IDS, "fit_ids": FIT_IDS}, checkpoint)
                marker = "  <- best, saved"
            print(f"epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}  "
                  f"val Dice {val_dice:.4f}  ({seconds:.0f}s){marker}")

    print(f"best validation Dice {best_dice:.4f}, checkpoint {checkpoint}")

    # Probability maps from the best model, for later threshold / post-processing tuning.
    save_probability_maps(load_model(checkpoint, device), device, args.out / "probs")


if __name__ == "__main__":
    main()
