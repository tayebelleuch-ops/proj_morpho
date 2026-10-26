"""Evaluation metrics, always computed inside the field-of-view (FOV) mask."""

import numpy as np
from skimage.morphology import skeletonize


def dice_in_fov(pred, truth, fov):
    """Dice = 2|P∩T| / (|P| + |T|), counting only pixels inside the FOV. Inputs are boolean (H, W)."""
    p, t = pred[fov], truth[fov]
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * np.logical_and(p, t).sum() / denom


def cldice_in_fov(pred, truth, fov):
    """clDice (Shit et al., 2021): Dice-like score on vessel centrelines. Inputs are boolean (H, W).

    - topology precision = fraction of the predicted skeleton that lies inside the true vessels
    - topology sensitivity = fraction of the true skeleton covered by the prediction
    clDice is their harmonic mean. A vessel cut in two loses part of the true skeleton,
    so clDice drops more than Dice. Both masks are cut to the FOV before skeletonising.
    """
    p, t = pred & fov, truth & fov
    skel_p, skel_t = skeletonize(p), skeletonize(t)
    tprec = (skel_p & t).sum() / max(skel_p.sum(), 1)
    tsens = (skel_t & p).sum() / max(skel_t.sum(), 1)
    return 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)
