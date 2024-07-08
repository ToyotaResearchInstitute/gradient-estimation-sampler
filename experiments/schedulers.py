import torch
import numpy as np

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils import BaseOutput
from typing import List, Optional, Tuple, Union

# Implementation Based on https://huggingface.co/docs/diffusers/using-diffusers/schedulers
class Scheduler:
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = 'linear',
            set_alpha_to_one: bool = True
    ):
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        else:
            raise NotImplementedError(f'{beta_schedule} does is not implemented for {self.__class__}')

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def sigma_to_t(self, sigma):
        '''Returns t so that self.nsr(t-1) <= sigma < self.nsr(t)'''
        alpha_bar = 1/(sigma**2+1)
        return self.num_train_timesteps - torch.searchsorted(reversed(self.alphas_cumprod), alpha_bar)

    def get_end_sigma(self, factor=2):
        sigma_1 = self.sigma(self.timesteps[-2])
        sigma_2 = self.sigma(self.timesteps[-1])
        return ((sigma_1.log() + sigma_2.log())/factor).exp()

    def set_timesteps_sigma(self,
                            start: float,
                            end: float,
                            num_inference_steps: int,
                            style: str = 'DDIM'):
        if style == 'DDIM':
            start_t, end_t = map(self.sigma_to_t, (start, end))
            self.timesteps = torch.tensor(np.linspace(start_t, end_t, num_inference_steps + 1),
                                          dtype=torch.long)
        elif style == 'EDM':
            rho = 7
            N = num_inference_steps + 1
            self.timesteps = [self.sigma_to_t(
                (start**(1/rho) + i/(N-1)*(end**(1/rho) - start**(1/rho)))**rho
            ) for i in range(N)]
        elif style == 'Linear':
            sigmas = torch.tensor(np.exp(np.linspace(np.log(start), np.log(end), num_inference_steps)))
            self.timesteps = torch.cat([self.sigma_to_t(sigmas), torch.tensor([0])])
        else:
            raise ValueError('Invalid style!')

    @property
    def timesteps_sigma(self):
        return torch.tensor([self.sigma(t) for t in self.timesteps])

    def set_timesteps(self, num_inference_steps: int, steps_offset: int):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f'`num_inference_steps` ({num_inference_steps}) must be smaller than'
                f'`self.num_train_timesteps` ({self.num_train_timesteps})'
            )
        if steps_offset >= self.num_train_timesteps:
            raise ValueError(
                f'`steps_offset` ({steps_offset}) must be smaller than '
                f'`self.num_train_timesteps` ({self.num_train_timesteps})'
            )
        start = self.num_train_timesteps - steps_offset
        self.timesteps = torch.tensor(np.linspace(start, 0, num_inference_steps + 1), dtype=torch.long)

    def ap(self, timestep: int):
        return self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod

    def nsr(self, timestep: int):
        return (1/self.ap(timestep) - 1).sqrt()

    def sigma(self, timestep: int):
        return self.nsr(timestep)
