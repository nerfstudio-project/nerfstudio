import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm


class PyramidInterpolator(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_ratio" in cfg
        assert "stride_ratio" in cfg
        assert "image_shape" in cfg

        self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
        self.stride = int(self.kernel_size * cfg["stride_ratio"])
        self.padding = self.kernel_size // 2
        self.center_x = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_y = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_x = torch.from_numpy(self.center_x).to(device)
        self.center_y = torch.from_numpy(self.center_y).to(device)

        self.model = model
        self.embed_size = self.model.embedding_dim
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        assert self.model is not None, "model must be provided to generate features"
        assert image_list is not None, "image_list must be provided to generate features"

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        for img in tqdm(image_list, desc="Embedding images", leave=False):
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), unfold_func))
        self.data = torch.from_numpy(np.stack(img_embeds)).half()

    def __call__(self, img_points, mode="interp"):
        # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        # return: (B, 512)
        img_points = img_points.to(self.device)
        img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

        x_ind = torch.searchsorted(self.center_x, img_points_x, side="left") - 1
        y_ind = torch.searchsorted(self.center_y, img_points_y, side="left") - 1
        return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

    def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y):
        topleft = self.data[img_ind, x_ind, y_ind].to(self.device)
        topright = self.data[img_ind, x_ind + 1, y_ind].to(self.device)
        botleft = self.data[img_ind, x_ind, y_ind + 1].to(self.device)
        botright = self.data[img_ind, x_ind + 1, y_ind + 1].to(self.device)

        x_stride = self.stride
        y_stride = self.stride
        right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).half()
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).half()
        return torch.lerp(top, bot, bot_w[:, None])

    def _embed_clip_tiles(self, image, unfold_func):
        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        aug_imgs = torch.cat([image])

        tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")

        with torch.no_grad():
            clip_embeds = self.model.encode_image(tiles)
        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()

    # def upsample_interp(self, input_ids, N_samp):
    #     B = input_ids.shape[0]
    #     if B == 0:
    #         return None
    #     img_ind, img_points_x, img_points_y = input_ids[:, 0], input_ids[:, 1], input_ids[:, 2]

    #     x_ind = torch.searchsorted(self.center_x, img_points_x, side="left") - 1
    #     y_ind = torch.searchsorted(self.center_y, img_points_y, side="left") - 1
    #     gt_embeds = self.interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

    #     blur_size = 0.5 * self.kernel_size

    #     x_start = img_points_x - blur_size / 2  # (N_patch, 1)
    #     x_start = x_start[:, None].repeat_interleave(N_samp, 0)  # (N_patcH*N_samp,1)
    #     y_start = img_points_y - blur_size / 2
    #     y_start = y_start[:, None].repeat_interleave(N_samp, 0)

    #     new_xs = torch.rand((N_samp * B, 1), device=self.device) * blur_size + x_start
    #     new_ys = torch.rand((N_samp * B, 1), device=self.device) * blur_size + y_start
    #     new_xs = torch.clip(new_xs, 0, self.imlist_shape[2] - 1)
    #     new_ys = torch.clip(new_ys, 0, self.imlist_shape[3] - 1)
    #     new_imids = img_ind[:, None].repeat_interleave(N_samp, 0)
    #     return torch.concat((new_imids, new_xs.long(), new_ys.long()), dim=1), gt_embeds.to(self.device)
