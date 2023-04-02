import os

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm


class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg

        cfg["tile_sizes"] = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        cfg["strider_scaler_list"] = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in cfg["tile_sizes"]]

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, scale=None):
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        raise FileNotFoundError  # trigger create

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(tqdm(self.cfg["tile_sizes"], desc="Scales")):
            stride_scaler = self.cfg["strider_scaler_list"][i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={"tile_ratio": tr.item(), "stride_ratio": stride_scaler, "image_shape": self.cfg["image_shape"]},
                device=self.device,
                model=self.model,
                image_list=image_list,
                cache_path=f"{self.cache_path}/level_{i}.npy",
            )

    def save(self):
        # don't save anything, PatchEmbeddingDataloader will save itself
        pass

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
