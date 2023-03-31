import dataclasses
import os
from dataclasses import field

import clip
import numpy as np
import torch
from tqdm import tqdm

from nerfstudio.data.utils.feature_dataloader import FeatureDataloader
from nerfstudio.data.utils.pyramid_interpolator import PyramidInterpolator
from nerfstudio.pipelines.lerf_encoders import ImageEncoder


@dataclasses.dataclass
class PatchedDataloader(FeatureDataloader):
    model: ImageEncoder = None

    def __post_init__(self):
        assert "tile_size_range" in self.cfg
        assert "tile_size_res" in self.cfg
        assert "stride_scaler" in self.cfg

        self.cfg["tile_sizes"] = torch.linspace(*self.cfg["tile_size_range"], self.cfg["tile_size_res"]).to(self.device)
        self.try_load(self.cache_dir)
        self.embed_size = self.model.embedding_dim

    def __call__(self, img_points, scale=None):
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def create(self):
        self.data_dict = {}
        for i, tr in enumerate(tqdm(self.cfg["tile_sizes"], desc="scale", leave=False)):
            stride_scaler = np.interp(tr.item(), [0.05, 0.15], [0.6, self.cfg["stride_scaler"]])
            self.data_dict[i] = PyramidInterpolator(
                image_list=self.image_list,
                device=self.device,
                model=self.model,
                tile_ratio=tr,
                stride_ratio=tr * stride_scaler,
            )
        for level in self.data_dict:
            self.data_dict[level].image_list = None
            self.data_dict[level].img_embeds_np = None

    # def sample_pixels(self, N_patch: int, N_samp: int, interp=True):
    # remnant code for deblur LERF (could be interesting still)
    #     """
    #     returns {'indices':(N_patch*N_samp, 3), 'image',(N_patch*N_samp, 3),
    #             'clip': (N_patch, 512)}, clip_scale:((N_patch*N_samp, 1))
    #     """
    #     # 1. sample patch_ids: N_patch samples into the whole dataset. (N_patch,4) 4 == (im_id,x_ind,y_ind,scale)
    #     im_ids = torch.randint(self.image_list.shape[0], (N_patch, 1))
    #     col_ind = torch.randint(self.image_list.shape[3], (N_patch, 1))
    #     row_ind = torch.randint(self.image_list.shape[2], (N_patch, 1))
    #     scale_ind = torch.randint(len(self.data_dict), (N_patch, 1))
    #     patch_ids = torch.concat([im_ids, row_ind, col_ind, scale_ind], dim=-1).to(self.device)
    #     # 3. use the interpolators to upsample each patch sample into (N_patch*N_samp,4)
    #     #         4 == (im_id,x_ind,y_ind,scale), and (N_patch,512)
    #     #       this is batch['indices'] (after removing scale) and batch['clip']
    #     upsampled_ids, clips, scales = [], [], []
    #     for i in self.data_dict:
    #         relevant_ids = patch_ids[patch_ids[:, 3] == i, :3]
    #         if interp:
    #             res = self.data_dict[i].upsample_interp(relevant_ids, N_samp)
    #         else:
    #             res = self.data_dict[i].upsample(relevant_ids, N_samp)
    #         if res is None:
    #             continue
    #         u, c = res
    #         s = (
    #             torch.rand((u.shape[0], 1), device=u.device, dtype=torch.float32) * 0.1
    #             + self.cfg["tile_sizes"][i] / 2
    #             - 0.05
    #         )
    #         upsampled_ids.append(u)
    #         clips.append(c)
    #         scales.append(s)
    #     upsampled_ids, clips, scales = (
    #         torch.concat(upsampled_ids, dim=0),
    #         torch.concat(clips, dim=0),
    #         torch.concat(scales, dim=0),
    #     )
    #     # 4. index into the images to get the color values (this is batch['image'])
    #     image = self.image_list[upsampled_ids[:, 0], :, upsampled_ids[:, 1], upsampled_ids[:, 2]]
    #     # 5. clip_scale is the 4th dim of step 3
    #     return {"image": image, "clip": clips, "indices": upsampled_ids.cpu()}, scales

    def load(self, cache_path):
        assert os.path.exists(cache_path)
        self.data_dict = {}
        for i, tr in enumerate(tqdm(self.cfg["tile_sizes"], desc="scale", leave=False)):
            clip_embeds = np.load(os.path.join(cache_path, f"cache{i}.npy"))
            stride_scaler = np.interp(tr.item(), [0.05, 0.15], [0.6, self.cfg["stride_scaler"]])
            self.data_dict[i] = PyramidInterpolator(
                image_list=self.image_list,
                device=self.device,
                img_embeds_np=clip_embeds,
                tile_ratio=tr,
                stride_ratio=tr * stride_scaler,
            )
            self.data_dict[i].image_list = None
            self.data_dict[i].img_embeds_np = None

    def save(self, cache_path):
        os.makedirs(cache_path, exist_ok=True)
        for i, interp in self.data_dict.items():
            np.save(os.path.join(cache_path, f"cache{i}.npy"), interp.img_embeds.detach().cpu().numpy())

    def _random_scales(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        tile_sizes = self.cfg["tile_sizes"]

        random_scale_bin = torch.randint(tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (tile_sizes[1] - tile_sizes[0]) / (tile_sizes[-1] - tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(tile_sizes) - 1):
            bottom_interp[random_scale_bin == i] = self.data_dict[i](img_points[random_scale_bin == i])
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](img_points[random_scale_bin == i])

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale):
        tile_sizes = self.cfg["tile_sizes"]

        scale_bin = torch.floor(
            (scale - tile_sizes[0]) / (tile_sizes[-1] - tile_sizes[0]) * (tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - tile_sizes[scale_bin]) / (tile_sizes[scale_bin + 1] - tile_sizes[scale_bin])
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale
