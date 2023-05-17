'''
Script for visualizing feature maps

    conda install on hal:   conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit 
                            conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
'''

import numpy as np
import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.io.image import read_image
from PIL import Image, ImageOps
import cv2
from matplotlib import pyplot
from sklearn import decomposition
import math
import argparse
import mediapy as media


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="/home/maturk/git/nerfstudio/data/desk", help='Path to root of video frame data')
parser.add_argument('--vis', type=bool, default=True, help='Visualize data as video')
parser.add_argument('--video_out', type=str, default='/home/maturk/git/nerfstudio/renders/custom/video_desk.mp4', help="Create video of features and write to this path.")
parser.add_argument('--image_in', type=str, default ='/home/maturk/git/nerfstudio/data/desk/images/frame_00002.png', help='Input file path of single image')
parser.add_argument('--image_out', type=str, default ='/home/maturk/git/nerfstudio/scripts/feat.png', help='Path to store output visualization')


class Dino:

    def __init__(self):
        self.normalize = normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
        self.model = torch.hub.load('facebookresearch/dino:main',
                                    'dino_vits8').half().cuda()
        self.model.eval()

    @property
    def shape(self):
        return (90, 120)

    def __call__(self, x):
        B, C, H, W = x.shape # C, H, W
        x = self.normalize(x)
        x = self.model.get_intermediate_layers(x.half())
        width_out = W // 8
        height_out = H // 8
        return x[0][:, 1:, :].reshape(B, height_out, width_out,
                                      384).detach().cpu().numpy()
   

def extract_features(images):
    extractor = Dino()
    features = extractor(images)
    return features


def visualize_features(image,features, image_out): # visualize input image and extracted features
    # PCA
    N, H, W, C = features[:].shape
    X = features[:].reshape(N * H * W, C)
    pca = decomposition.PCA(n_components=3) 
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.fit_transform(subset)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    range = maximum - minimum

    mapped = pca.transform(features.reshape(H * W, C)).reshape(H, W, 3) # H, W
    normalized = np.clip(
            (mapped - minimum) / range, 0, 1)
    _, axis = pyplot.subplots(1,2)
    axis[1].imshow(normalized.transpose(1,0,2))
    axis[0].imshow(image.transpose(0,3,2,1).squeeze())
    pyplot.tight_layout()
    pyplot.savefig(image_out)


def visualize_image(image, image_out): # visualize input image
    _, axis = pyplot.subplots(1)
    axis.imshow(image.transpose(0,3,2,1).squeeze())
    pyplot.tight_layout()
    pyplot.savefig(image_out)


def video(data, extractor, video_out): # visualize video stream features
    import glob
    import os
    from skvideo.io.ffmpeg import FFmpegWriter
    image_paths = sorted(glob.glob(data + "/images/*.png"))[:]
    # batching features for memory
    batch_size = 1
    extracted = []
    for i in range(math.ceil(len(image_paths) / batch_size)):
        batch = image_paths[i * batch_size:(i + 1) * batch_size]
        images = torch.stack([read_image(p) for p in batch]).cuda()
        images = F.interpolate(images, [720, 960])
        features = extractor(images / 255.)
        extracted += [f for f in features]
    extracted = np.stack(extracted)
    features = extracted
    
    # PCA
    N, H, W, C = features[:].shape
    X = features[:].reshape(N * H * W, C)
    pca = decomposition.PCA(n_components=3) 
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.fit_transform(subset)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    diff = maximum - minimum

    # save paths
    if not os.path.exists(os.path.join(data, 'features_vis')):
        os.makedirs(os.path.join(data, 'features_vis'))

    with media.VideoWriter(video_out, fps = 5, shape=(features[0].shape[0], features[0].shape[1])) as writer:
        for feature, path in zip(features,image_paths): 
            mapped = pca.transform(feature.reshape(H * W, C)).reshape(H, W, 3) # H, Wl
            normalized = np.clip(
                    (mapped - minimum) / diff, 0, 1)
            frame = (normalized * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(os.path.join(data, 'features_vis'),'features_'+os.path.basename(path).split('/')[-1].split('_')[1]),frame)
            writer.add_image(frame)
    writer.close()


def main():
    opt = parser.parse_args()
    extractor = Dino()

    image = np.array(Image.open(opt.image_in), dtype=np.float32) / 255.
    size = (960,720) #default. image = F.interpolate(image, [720, 960])
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA) #INTER_CUBIC
    image = image[np.newaxis,:].transpose(0,3,2,1)
    features = extractor(torch.Tensor(image).cuda())

    if opt.vis:
        #visualize_image(image, opt.image_out)
        #visualize_features(image,features, opt.image_out)
        video(opt.data, extractor, opt.video_out)


if __name__ == "__main__":
    main()