"""
Code to test ray generation.
"""

import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mattport.nerf.dataset.collate import CollateIterDataset, collate_batch_size_one
from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import get_dataset_inputs_dict
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.utils.io import get_absolute_path
from mattport.viewer.plotly import visualize_dataset


def visualize_batch(batch):
    """Returns an image from a batch."""
    assert "image" in batch, "You must call collate_batch with keep_full_image=True for this to work."
    # set the color of the sampled rays
    c, y, x = [i.flatten() for i in torch.split(batch["local_indices"], 1, dim=-1)]
    batch["image"][c, y, x] = 1.0

    # batch["image"] is num_images, h, w, 3
    images = torch.split(batch["image"], 1, dim=0)
    image_list = [image[0] for image in images]
    image = torch.cat(image_list, dim=1)  # cat along the width dimension
    image = (image * 255.0).to(torch.uint8)
    return image


def test_dataloader(visualize=False):
    """Testing for the dataloader from input dataset parameters to rays."""

    data_directory = "tests/data/lego_test"
    dataset_type = "blender"
    downscale_factor = 1
    num_images_to_sample_from = 1
    num_times_to_repeat_images = 40
    num_rays_per_batch = 1024
    num_workers = 0

    data_directory = get_absolute_path(data_directory)
    dataset_inputs = get_dataset_inputs_dict(
        data_directory=data_directory, dataset_type=dataset_type, downscale_factor=downscale_factor
    )["train"]
    image_dataset = ImageDataset(
        image_filenames=dataset_inputs.image_filenames, downscale_factor=dataset_inputs.downscale_factor
    )
    iter_dataset = CollateIterDataset(
        image_dataset,
        collate_fn=lambda batch_list: collate_batch(batch_list, num_rays_per_batch, keep_full_image=True),
        num_samples_to_collate=num_images_to_sample_from,
        num_times_to_repeat=num_times_to_repeat_images,
    )
    dataloader = DataLoader(
        iter_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_batch_size_one,
        pin_memory=True,
    )
    dataloader_iter = iter(dataloader)

    ray_generator = RayGenerator(dataset_inputs.intrinsics, dataset_inputs.camera_to_world)

    num_batches = 10
    for _ in tqdm(range(num_batches)):
        batch = next(dataloader_iter)
        ray_bundle = ray_generator.forward(batch["indices"])

    # visualize the batch
    image = visualize_batch(batch)
    if visualize:
        imageio.imwrite("temp0.png", image)

    # visualize the RayBundle
    small_ray_bundle = ray_bundle.sample(100)
    fig = visualize_dataset(camera_origins=dataset_inputs.camera_to_world[:, :3, 3], ray_bundle=small_ray_bundle)
    if visualize:
        fig.write_image("temp1.png")
        fig.write_html("temp1.html")
    assert True


if __name__ == "__main__":
    test_dataloader(visualize=True)
