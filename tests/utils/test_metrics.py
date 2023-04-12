"""
Test metrics
"""
import torch

from nerfstudio.utils.metrics import LPIPSModule, PSNRModule, SSIMModule


def test_psnr():
    """Test PSNR metric"""
    bs, h, w = 10, 100, 200
    pred = torch.rand((bs, 3, h, w))
    target = torch.rand((bs, 3, h, w))
    mask = torch.rand((bs, 1, h, w)) > 0.5

    psnr_module = PSNRModule()

    # test non-masked version
    psnr = psnr_module(preds=pred, target=target)
    assert psnr.shape == (bs,)
    assert torch.min(psnr) >= 0

    # test masked version
    psnr = psnr_module(preds=pred, target=target, mask=mask)
    assert psnr.shape == (bs,)
    assert torch.min(psnr) >= 0


def test_ssim():
    """Test SSIM metrics"""
    bs, h, w = 10, 100, 200
    pred = torch.rand((bs, 3, h, w))
    target = torch.rand((bs, 3, h, w))
    mask = torch.rand((bs, 1, h, w)) > 0.5

    ssim_module = SSIMModule()

    # test non-masked version
    ssim = ssim_module(preds=pred, target=target)
    assert ssim.shape == (bs,)

    # test masked version
    ssim = ssim_module(preds=pred, target=target, mask=mask)
    assert ssim.shape == (bs,)


def test_lpips():
    """Test LPIPS metrics"""
    bs, h, w = 10, 100, 200
    pred = torch.rand((bs, 3, h, w))
    target = torch.rand((bs, 3, h, w))
    mask = torch.rand((bs, 1, h, w)) > 0.5

    lpips_module = LPIPSModule()

    # test non-masked version
    lpips = lpips_module(preds=pred, target=target)
    print(lpips)
    assert lpips.shape == (bs,)

    # test masked version
    lpips = lpips_module(preds=pred, target=target, mask=mask)
    assert lpips.shape == (bs,)
