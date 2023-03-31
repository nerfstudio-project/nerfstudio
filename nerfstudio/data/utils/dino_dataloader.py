import dataclasses
import os
import typing

import numpy as np
import torch
from tqdm import tqdm

from nerfstudio.data.utils.dino_extractor import ViTExtractor
from nerfstudio.data.utils.feature_dataloader import FeatureDataloader


@dataclasses.dataclass
class DinoDataloader(FeatureDataloader):
    def __post_init__(self):
        self.cfg["dino_vit"] = {}
        self.cfg["dino_vit"]["model_type"] = "dino_vits8"
        self.cfg["dino_vit"]["stride"] = 8
        self.cfg["dino_vit"]["load_size"] = 500
        self.cfg["dino_vit"]["layer"] = 11
        self.cfg["dino_vit"]["facet"] = "key"
        self.cfg["dino_vit"]["bin"] = False
        self.try_load(self.cache_dir)
        print("after load dino", torch.cuda.memory_allocated())

    def create(self):
        extractor = ViTExtractor(
            self.cfg["dino_vit"]["model_type"],
            self.cfg["dino_vit"]["stride"],
        )

        preproc_image_lst = extractor.preprocess(self.image_list, self.cfg["dino_vit"]["load_size"])[0].to(self.device)

        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(self.image_list), leave=False):
            descriptors = extractor.extract_descriptors(
                image.unsqueeze(0),
                [self.cfg["dino_vit"]["layer"]],
                self.cfg["dino_vit"]["facet"],
                self.cfg["dino_vit"]["bin"],
            )
            descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
            dino_embeds.append(descriptors.cpu().detach())

        self.data_dict = {}
        self.data_dict["all"] = torch.stack(dino_embeds, dim=0)
        self.cfg["img_scale"] = (
            self.data_dict["all"].shape[1] / self.image_list.shape[2],
            self.data_dict["all"].shape[2] / self.image_list.shape[3],
        )

    def load(self, cache_path):
        self.data_dict = {}
        self.data_dict["all"] = torch.from_numpy(np.load(os.path.join(cache_path, "dino.npy")))

        self.cfg["img_scale"] = (
            self.data_dict["all"].shape[1] / self.image_list.shape[2],
            self.data_dict["all"].shape[2] / self.image_list.shape[3],
        )

    def save(self, cache_path):
        np.save(os.path.join(cache_path, "dino.npy"), self.data_dict["all"].cpu().numpy())

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        x_ind, y_ind = (img_points[:, 1] * self.cfg["img_scale"][0]).long(), (
            img_points[:, 2] * self.cfg["img_scale"][1]
        ).long()
        return (self.data_dict["all"][img_points[:, 0].long(), x_ind, y_ind]).to(self.device)


# @dataclasses.dataclass
# class DinoDataloader():
#     image_lst: torch.Tensor # (N, 3, H, W)
#     device: torch.device
#     dino_embeds: np.ndarray = None
#     dino_embeds_smap: np.ndarray = None
#     do_cache: bool = True
#     cache_path: str = None
#     load_size: int = 500
#     stride: int = 8
#     model_type: str = 'dino_vits8'
#     facet: str = 'key'
#     layer: typing.Tuple[int] = (11,)
#     all_layers: bool = False
#     bin: bool = False

#     def __post_init__(self):
#         # if cached version exists + want to use cache ver, use it
#         if self.all_layers:
#             self.layer = list(range(12))

#         if (
#             self.do_cache and self.cache_path is not None
#             and os.path.exists(os.path.join(self.cache_path, f"cache_dino.npy"))
#             and os.path.exists(os.path.join(self.cache_path, f"cache_dino_smap.npy"))
#             ):
#             self.dino_embeds = np.load(os.path.join(self.cache_path, f"cache_dino.npy"))
#             self.dino_embeds_smap = np.load(os.path.join(self.cache_path, f"cache_dino_smap.npy"))

#             if len(self.layer) == 1:
#                 self.dino_embeds = self.dino_embeds[:, :, :, :]
#             else:
#                 self.dino_embeds = self.dino_embeds[:, self.layer, :, :]
#         else:
#             self.calculate_dino()

#         self.dino_img_scale = (
#             self.dino_embeds.shape[2]/self.image_lst.shape[2],
#             self.dino_embeds.shape[3]/self.image_lst.shape[3]
#             )

#         print(self.dino_embeds.shape)
#         del self.image_lst

#     def calculate_dino(self):
#         self.extractor = ViTExtractor(self.model_type, self.stride, device=self.device)
#         self.dino_embeds_smap = []
#         self.dino_embeds = []

#         preproc_image_lst = self.extractor.preprocess(self.image_lst, self.load_size)[0].to(self.device)
#         for image in tqdm(preproc_image_lst, desc='dino', total=len(self.image_lst), leave=False):
#             descriptors = self.extractor.extract_descriptors(image.unsqueeze(0), self.layer, self.facet, self.bin)
#             descriptors = descriptors.reshape((len(self.layer), self.extractor.num_patches[0], self.extractor.num_patches[1], -1))
#             self.dino_embeds.append(descriptors.cpu().detach().numpy())

#             saliency_maps = self.extractor.extract_saliency_maps(image.unsqueeze(0))
#             saliency_maps = saliency_maps.reshape((self.extractor.num_patches[0], self.extractor.num_patches[1], -1))
#             self.dino_embeds_smap.append(saliency_maps.cpu().detach().numpy())

#         self.dino_embeds = np.stack(self.dino_embeds, axis=0) # (# of imgs, # of layers, # of patches, # of patches, descriptor_dim)
#         self.dino_embeds_smap = np.stack(self.dino_embeds_smap, axis=0)

#         if self.do_cache and self.cache_path is not None:
#             os.makedirs(self.cache_path,exist_ok=True)
#             np.save(os.path.join(self.cache_path, f"cache_dino.npy"), self.dino_embeds)
#             np.save(os.path.join(self.cache_path, f"cache_dino_smap.npy"), self.dino_embeds_smap)

#         self.cleanup()

#     def cleanup(self):
#         del self.extractor
#         torch.cuda.empty_cache()

#     def __call__(self, img_points, layer=None):
#         # img_points: (B, 3) # (img_ind, x, y)
#         # return: (B, descriptor_dim), (B, smap_dim), (B, 1)

#         img_ind = img_points[:, 0]
#         x_ind = (img_points[:, 1]*self.dino_img_scale[0]).long()
#         y_ind = (img_points[:, 2]*self.dino_img_scale[1]).long()

#         if layer is None:
#             layer = np.random.randint(low=0, high=len(self.layer), size=(img_points.shape[0],1))
#             # layer = np.random.choice(self.layer, size=(img_points.shape[0], 1))

#         saliency = self.dino_embeds_smap[img_ind, x_ind, y_ind]

#         # nearest neighbor interpolation
#         return (
#             torch.from_numpy(self.dino_embeds[img_ind, layer.flatten(), x_ind, y_ind]).to(self.device),
#             torch.from_numpy(saliency).to(self.device),
#             (torch.Tensor(self.layer)[layer]/11).to(self.device),
#         )

if __name__ == "__main__":
    import itertools
    import time

    import torchvision
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_inputs = [
        Image.open("/home/jkerr/nerfstudio/data/cmk_mug/images/frame_00001.jpg"),
        Image.open("/home/jkerr/nerfstudio/data/cmk_mug/images/frame_00007.jpg"),
        Image.open("/home/jkerr/nerfstudio/data/cmk_mug/images/frame_00056.jpg"),
    ]
    dl = DinoDataloader(img_inputs, device)

    img_lst = torch.cat([torchvision.transforms.ToTensor()(img).unsqueeze(0) for img in img_inputs]).to(device)
    x = np.arange(0, img_lst.shape[2], 10)
    y = np.arange(0, img_lst.shape[3], 10)
    img_points = torch.from_numpy(np.stack(list(itertools.product(np.arange(img_lst.shape[0]), x, y))))
    inds, gx_, gy_ = img_points[:, 0].long(), img_points[:, 1].long(), img_points[:, 2].long()

    start = time.time()
    dino_embeds = dl(img_points.numpy())
    end = time.time()
    print(end - start)
