"""
Code to test ray generation.
"""

import imageio
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader
from tqdm import tqdm

from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import get_dataset_inputs
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.utils.io import get_absolute_path
from mattport.viewer.plotly import visualize_dataset


def visualize_batch(batch):
    """Returns an image from a batch."""
    assert "image" in batch, "You must call collate_batch with keep_full_image=True for this to work."
    # set the color of the sampled rays
    c, y, x = [i.flatten() for i in torch.split(batch.indices, 1, dim=-1)]
    batch.image[c, y, x] = 1.0

    # batch.image is num_images, h, w, 3
    images = torch.split(batch.image, 1, dim=0)
    image_list = [image[0] for image in images]
    image = torch.cat(image_list, dim=1)  # cat along the width dimension
    image = (image * 255.0).to(torch.uint8)
    return image


def test_dataloader():
    """Testing for the dataloader from input dataset parameters to rays."""

    with initialize(config_path="configs"):
        cfg = compose(config_name="blender_lego.yaml")

    cfg.dataset.data_directory = get_absolute_path(
        cfg.dataset.data_directory
    )  # TODO(ethan): create a helper function for this
    dataset_inputs = get_dataset_inputs(**cfg.dataset)
    dataset = ImageDataset(
        image_filenames=dataset_inputs.image_filenames, downscale_factor=dataset_inputs.downscale_factor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.num_images_to_sample_from,
        collate_fn=lambda batch: collate_batch(batch, cfg.dataloader.num_rays_per_batch, keep_full_image=True),
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
    )
    dataloader_iter = iter(dataloader)

    ray_generator = RayGenerator(dataset_inputs.intrinsics, dataset_inputs.camera_to_world)

    num_batches = 10
    for _ in tqdm(range(num_batches)):
        batch = next(dataloader_iter)
        ray_bundle = ray_generator.forward(batch.indices)

    # visualize the batch
    image = visualize_batch(batch)
    imageio.imwrite("temp0.png", image)

    # visualize the RayBundle
    small_ray_bundle = ray_bundle.sample(100)
    fig = visualize_dataset(camera_origins=dataset_inputs.camera_to_world[:, :3, 3], ray_bundle=small_ray_bundle)
    fig.write_image("temp1.png")
    fig.write_html("temp1.html")


if __name__ == "__main__":
    test_dataloader()
