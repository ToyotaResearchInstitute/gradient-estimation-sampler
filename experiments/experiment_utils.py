import torch
import argparse
import os
import sys
import shutil
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.nn.functional import cosine_similarity
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image
from diffusers.models.unet_2d import UNet2DOutput
from pytorch_fid.fid_score import calculate_fid_given_paths

def show(x, nrow=4, padding=2):
    return TF.to_pil_image(make_grid(x, nrow=nrow, padding=padding).add(1).div(2).clamp(0, 1))

def batched_t(batchsize, t, device):
    return torch.ones(batchsize, dtype=int, device=device) * t

def norm(x):
    return torch.linalg.vector_norm(x, dim=(1,2,3))

def normalize(x):
    return x / norm(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def angle(x, y):
    x, y = torch.flatten(x, start_dim=1), torch.flatten(y, start_dim=1)
    return cosine_similarity(x, y, dim=1)

class Plotter:
    def __init__(self, auto_plot=True, save_to=None, hist=False, hist_kwargs=None):
        self.prev = False
        self.series = defaultdict(list)
        self.auto_plot = auto_plot
        self.save_to = save_to
        self.hist = hist
        self.hist_kwargs = hist_kwargs or dict(bins=100, cmap='plasma_r', cmin=1)

    def add(self, name, t, value):
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
        self.series[name].append((t, value))

    def plot(self):
        for name, xs, ys in self.iterplots():
            if self.hist:
                X, Y = map(np.concatenate, zip(*((np.ones_like(y) * x, y) for x, y in zip(xs, ys))))
                plt.hist2d(X, Y, **self.hist_kwargs)
            else:
                plt.plot(xs, ys, label=name)
        if self.save_to is not None:
            plt.tight_layout()
            plt.savefig(self.save_to)
        else:
            plt.show()

    def iterplots(self):
        for name, series in self.series.items():
            x, y = zip(*series)
            yield name, x, y

    def __enter__(self):
        plt.clf()
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)
        if self.auto_plot:
            self.plot()

class Saver:
    def __init__(self, images_dir, target=None, overwrite=True,):
        if overwrite and os.path.exists(images_dir):
            print(f'Removing directory: {images_dir}')
            shutil.rmtree(images_dir)
        if not os.path.exists(images_dir):
            print(f'Making new directory: {images_dir}')
            os.makedirs(images_dir)
        self.images_dir = images_dir
        self.prev = False
        self.batch_counter = 0
        self.target = target
        self.fid_score = None

    def save(self, sample): # sample.shape = (B, C, H, W)
        sample = sample.add(1).div(2).clamp(0, 1)
        for i, img in enumerate(sample):
            path = os.path.join(self.images_dir, f'{self.batch_counter:05}-{i:03}.png')
            save_image(img, path)
        self.batch_counter += 1

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)
        if self.target is not None:
            self.fid_score = fid(self.images_dir, self.target)
            print('FID score: ', self.fid_score)

def samples(model, scheduler, sample, timesteps=100, display_every=1, verbose=True):
    scheduler.set_timesteps(timesteps)
    steps = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for i, t in enumerate(steps):
        with torch.no_grad():
            residual = model(sample, t).sample
        res = scheduler.step(residual, t, sample)
        sample = res.prev_sample

        if (i + 1) % display_every == 0:
            yield sample, residual, res.pred_original_sample

class DDIMModel:
    def __init__(self,
                 config_path='./config/ddim_cifar10.yml',
                 model_path='./models/ddim_cifar10.ckpt',
                 device='cuda'):
        with open(os.path.join(os.getcwd(), config_path), 'r') as f:
            config = yaml.safe_load(f)
        self.device = device
        print("Loading model...")
        from ddim_model import Model
        self.model = Model(None, config['Model']).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        self.model.eval()
        print("...done.")
        self.in_channels = self.model.in_channels
        self.sample_size = self.model.resolution

    def __call__(self, x, t):
        return UNet2DOutput(sample=self.model(x, t.expand(x.shape[0]).to(self.device)))

def generate_images(model, scheduler,
        timesteps=100,
        batchsize=64,
        batches=256,
        seed=0,
        device='cuda',
        save_to='images/fid_results'):
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    torch.manual_seed(seed)
    shape = (batchsize, model.in_channels, model.sample_size, model.sample_size)

    for i in tqdm(range(batches)):
        init = torch.randn(shape).to(device)
        for _, _, sample in samples(model, scheduler, init,
                                    timesteps=timesteps,
                                    display_every=timesteps,
                                    verbose=False):
            sample = sample.add(1).div(2).clamp(0, 1)
            for j, img in enumerate(sample):
                path = os.path.join(save_to, f'{i:05}-{j:03}.png')
                save_image(img, path)


def fid(source,
        target,
        device='cuda:0',
        batch_size=64,
        dims=2048):
    return calculate_fid_given_paths((source, target), batch_size, device, dims)
