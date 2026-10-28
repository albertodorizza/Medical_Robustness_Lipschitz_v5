"""
fullres_certificate/test_fullres.py
===================================
Checks of the full-resolution certificate that need no checkpoint and no GPU:
a random 1x1 head on random features, with random and with adversarial drifts.

    python -m fullres_certificate.test_fullres

1. Upsampling the head output reproduces the model's own logits (the head
   commutes with bilinear interpolation), so ``certify_batch`` sees the real
   prediction.
2. No full-resolution pixel flips while called safe, for random drifts and for
   drifts that push every cell's gap down by exactly its allowed amount.
3. The grid certificate alone does NOT cover full resolution: with a drift
   that leaves every cell safe, boundary pixels can still flip.  This is the
   reason the module exists, so the test asserts it rather than assuming it.
"""

import os
import sys
import types
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The certificate code imports the pipeline helpers, which pull in torchvision
# and wandb.  Neither is needed for these checks: stub them when missing so the
# test runs on a laptop as well as on the server.
for name in ("torchvision", "torchvision.models", "torchvision.transforms",
             "torchvision.transforms.functional", "wandb", "timm"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = mock.MagicMock()

from fullres_certificate.certificate_seg_fullres import _margin, certify_batch  # noqa: E402

torch.manual_seed(0)
B, D, K, Hf, Wf, H, W = 4, 16, 2, 14, 14, 224, 224


def _up(t):
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)


def _gamma(head):
    w = head.weight.flatten(1)
    return max((w[a] - w[b]).norm().item() for a in range(K) for b in range(K) if a != b)


def _batch(head, feat_c, feat_k, target):
    lr_c, lr_k = head(feat_c), head(feat_k)
    return certify_batch(feat_c, feat_k, lr_c, lr_k, _up(lr_c), _up(lr_k),
                         target, _gamma(head))


@torch.no_grad()
def main():
    head = torch.nn.Conv2d(D, K, 1).double()
    feat_c = torch.randn(B, D, Hf, Wf, dtype=torch.float64)
    target = (_up(head(feat_c)).argmax(1) == 1).long()  # plausible lesion masks
    gamma = _gamma(head)

    # 1. head and upsampling commute
    x = torch.randn(B, D, Hf, Wf, dtype=torch.float64)
    assert torch.allclose(_up(head(x)), head.weight.flatten(1)[None, :, :, None, None]
                          .mul(_up(x)[:, None]).sum(2)
                          + head.bias[None, :, None, None], atol=1e-10)

    # 2a. random drifts of several sizes
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
        feat_k = feat_c + scale * torch.randn_like(feat_c)
        r = _batch(head, feat_c, feat_k, target)
        assert r["viol_pix_full"].sum() == 0, ("random", scale)
        assert r["viol_pix_grid"].sum() == 0, ("random grid", scale)
        ok = ~torch.isnan(r["bound_full"])
        drop = (r["dice_clean_full"] - r["dice_corr_full"])[ok]
        assert (drop <= r["bound_full"][ok] + 1e-9).all(), ("dice", scale)

    # 2b. worst case for the binary gap: every cell moves its gap g1-g0 toward
    # the other class by the full allowed amount Gamma ||dz||.
    w = head.weight.flatten(1)
    u = (w[1] - w[0]) / (w[1] - w[0]).norm()
    for scale in (1e-2, 1e-1, 1.0):
        gap = (head(feat_c)[:, 1] - head(feat_c)[:, 0])        # (B, Hf, Wf)
        sign = torch.where(gap > 0, -1.0, 1.0).double()
        feat_k = feat_c + scale * sign[:, None] * u[None, :, None, None]
        r = _batch(head, feat_c, feat_k, target)
        assert r["viol_pix_full"].sum() == 0, ("adversarial", scale)

    # 3. a drift that leaves every CELL safe can still flip PIXELS.  Raise the
    # gap g1-g0 of every cell by 0.8 |gap|: a background cell (gap < 0) keeps
    # its sign, and its drift Gamma ||dz|| = 0.8 |gap| is below its margin, so
    # the grid certificate calls every cell safe.  A boundary pixel that
    # mixes a weak background cell with a foreground cell can still cross 0.
    gap = head(feat_c)[:, 1] - head(feat_c)[:, 0]
    step = 0.8 * gap.abs() / (w[1] - w[0]).norm()
    feat_k = feat_c + step[:, None] * u[None, :, None, None]
    r = _batch(head, feat_c, feat_k, target)
    cells_flipped = (head(feat_c).argmax(1) != head(feat_k).argmax(1)).sum().item()
    px_flipped = float(r["flip_full"].sum()) * H * W
    print(f"[3] cells at risk: {float(r['at_risk_grid'].sum()):.0f}, "
          f"cells flipped: {cells_flipped}, pixels flipped: {px_flipped:.0f}, "
          f"of which unseen by the grid: "
          f"{float(r['flip_unseen_by_grid'].sum()) * H * W:.0f}")
    assert float(r["at_risk_grid"].sum()) == 0 and cells_flipped == 0
    assert px_flipped > 0, "construction should flip boundary pixels"
    assert r["viol_pix_full"].sum() == 0, "full-res certificate missed a flip"

    print(f"Gamma = {gamma:.3f}")
    print("all checks passed")


if __name__ == "__main__":
    main()
