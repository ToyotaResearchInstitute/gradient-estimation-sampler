import math
from dataclasses import dataclass
from itertools import pairwise
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->GE
class GESchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class GEScheduler(SchedulerMixin, ConfigMixin):
    """
    Gradient estimation scheduler that improves upon DDIM in two ways:
     1. Uses previous denoiser outputs to correct for errors in estimated nosie
     2. Uses a log-linear sigma schedule in the denoising process

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2306.04848

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample for numerical stability.
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, must be `epsilon` (predicting the noise of the diffusion
            process) for now
        start (`float`):
            the sigma value to start the diffusion process, default 14.6 for StableDiffusion
        end (`float`):
            the sigma value to end the diffusion process, default 0.02 for StableDiffusion
        style (`str`):
            what kind of timestep spacings to use, can be
             - 'DDIM' for equal spacings in t
             - 'EDM'  for the spacing used in the Karras et.al. paper
             - 'Linear' for log-linear spacings in sigma
        gamma (`torch.float`):
            value of gamma to use for the gradient estimation,
            see equation (12) of https://arxiv.org/abs/2306.04848
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = True,
        prediction_type: str = "epsilon",
        start: float = 8.35,
        end: float = 0.02,
        style: str = 'Linear',
        gamma: float = 2.,
        factor: float = 0.,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.sigmas = torch.tensor([self.sigma(t) for t in range(num_train_timesteps)])

        # setable values
        self.num_inference_steps = None
        self.timesteps = None       # Not including last step
        self._timesteps_full = None # Full timesteps including last step

        self.init_noise_sigma = self.sigmas[-1]

        assert 0 <= factor <= 1, '`factor` should lie in [0,1]!'

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample * self._ap(timestep).sqrt()

    def sigma_to_t(self, sigma: float) -> int:
        ''' Conversion from noise-to-signal ratio sigma to discrete timestep t
        Args:
            sigma (`float`): input sigma
        Returns
            `int`: t so that self.sigma(t-1) <= sigma < self.sigma(t)
        '''
        return torch.searchsorted(self.sigmas, sigma)

    def _get_end_sigma(self, num_inference_steps, end):
        factor = self.config.factor
        sigma_1 = self.sigma(self.sigma_to_t(end))
        sigma_2 = self.sigma(self.config.num_train_timesteps // num_inference_steps)
        return (sigma_1.log() * factor + sigma_2.log()*(1-factor)).exp()

    def _ap(self, timestep: int) -> float:
        '''Convenience function to get alpha_cumprod'''
        return self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod

    def sigma(self, timestep: int) -> float:
        ''' Conversion from discrete timestep t to noise-to-signal ratio sigma
        Args:
            timestep (`int`): input sigma
        Returns
            `float`: sigma = sqrt(1-alpha_bar)/sqrt(alpha_bar)
        '''
        return (1/self._ap(timestep) - 1).sqrt()

    @property
    def timesteps_sigma(self) -> torch.FloatTensor:
        ''' The sigmas to use in timesteps
        '''
        return torch.tensor([self.sigma(t) for t in self._timesteps_full])

    def set_timesteps(self,
                      num_inference_steps: int,
                      device: Union[str, torch.device] = None,
                      ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )
        self.num_inference_steps = num_inference_steps

        start, end, style = self.config.start, self.config.end, self.config.style
        # self._timesteps_full includes last step
        if style == 'DDIM':
            start_t, end_t = map(self.sigma_to_t, (start, end))
            self._timesteps_full = torch.tensor(np.linspace(start_t, end_t, num_inference_steps + 1),
                                                 dtype=torch.long).to(device)
        elif style == 'DDIMPrecise':
            start_t, end_t = map(self.sigma_to_t, (start, end))
            self._timesteps_full = torch.cat([
                torch.tensor(np.linspace(start_t, end_t, num_inference_steps),dtype=torch.long),
                torch.tensor([0])
            ]).to(device)
        elif style == 'EDM':
            rho = 7
            N = num_inference_steps + 1
            self._timesteps_full = torch.tensor([self.sigma_to_t(
                (start**(1/rho) + i/(N-1)*(end**(1/rho) - start**(1/rho)))**rho
            ) for i in range(N)], dtype=torch.long).to(device)
        elif style == 'Linear':
            linear_end = self._get_end_sigma(num_inference_steps, end)
            sigmas = torch.tensor(np.exp(np.linspace(np.log(start), np.log(linear_end), num_inference_steps)))
            self._timesteps_full = torch.cat([
                self.sigma_to_t(sigmas), torch.tensor([0])
            ]).to(device)
        elif style == 'LinearPrecise':
            linear_end = self._get_end_sigma(num_inference_steps, end)
            sigmas = torch.tensor(np.exp(np.linspace(np.log(start), np.log(linear_end), num_inference_steps-1)))
            self._timesteps_full = torch.cat([
                self.sigma_to_t(sigmas), torch.tensor([1, 0])
            ]).to(device)
        else:
            raise NotImplementedError('Invalid style!')

        # self.timesteps does not include last step,
        # so that num_inference_steps == len(self.timesteps)
        self.timesteps = self._timesteps_full[:-1]
        self.init_noise_sigma = 1 / self._ap(self.timesteps[0]).sqrt()
        assert len(self.timesteps) + 1 == len(self._timesteps_full), \
            f'self._timesteps_full need to include last timestep'

        # Erase history of eps for predictions
        self.eps_prev = None

        # Helps determine next step given current step
        self._timesteps_map = dict(pairwise(self._timesteps_full.cpu().numpy()))

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[GESchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than GESchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.GESchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.GESchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.config.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be `epsilon`"
            )

        # 0. Define some variables in notation of paper
        first_iter = self.eps_prev is None
        t, t_prev = timestep, self._timesteps_map[int(timestep)]
        eps = model_output
        xt = sample
        gamma = self.config.gamma

        # 1.1 Predict x0, clip if necessary, then recompute eps with clipped x0
        if self.config.clip_sample:
            x0_pred = (xt - self.sigma(t) * eps).clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
            eps = (xt - x0_pred) / self.sigma(t)

        # 1.2 Gradient/projection estimate
        eps_bar = eps if first_iter else eps * gamma + self.eps_prev * (1 - gamma)

        # 2. DDIM step
        prev_sample = xt_prev = xt + (self.sigma(t_prev) - self.sigma(t)) * eps_bar

        # 3. Use more accurate estimate of gradient/projection (eps_bar) to predict original sample
        pred_original_sample = x0_pred = xt - self.sigma(t) * eps_bar

        # 4. Save current eps for next iteration
        self.eps_prev = eps

        if not return_dict:
            return (prev_sample,)

        return GESchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
    def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
