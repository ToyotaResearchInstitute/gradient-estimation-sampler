import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from PIL import Image
from itertools import pairwise

from schedulers import Scheduler

class StableDiffuser:
    def __init__(self,
                 model_key,
                 scheduler='ddim',
                 device='cuda',
                 float_dtype=torch.float32):
        self.model_key = model_key
        self.device = device
        self.float_dtype = float_dtype
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder='vae',
                                                 torch_dtype=float_dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_key,
            subfolder='tokenizer',
            torch_dtype=float_dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key,
            subfolder='text_encoder',
            torch_dtype=float_dtype)

        self.unet = UNet2DConditionModel.from_pretrained(
            model_key,
            subfolder='unet',
            torch_dtype=float_dtype)
        if scheduler == 'pndm':
            self.scheduler = PNDMScheduler.from_pretrained(
                model_key,
                subfolder='scheduler')
        else:
            assert(scheduler == 'ddim')
            self.scheduler = DDIMScheduler.from_pretrained(
                model_key,
                subfolder='scheduler',
                torch_dtype=float_dtype)

        self.vae = self.vae.to(self.device)
        self.vae.enable_slicing()
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    def embed_text(self, prompt):
        text_input = self.tokenizer([prompt], padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embs = self.text_encoder(
                text_input.input_ids.to(self.device))[0]

        # Include unconditional text input for classifier-free guidance.
        uncond_input = self.tokenizer([''], padding='max_length',
                                      max_length=text_input.input_ids.shape[-1],
                                      return_tensors='pt')
        with torch.no_grad():
            uncond_embs = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embs, text_embs])

    def encode_imgs(self, imgs):
        '''
        Args:
          imgs: [B, 3, H, W], with values in [0, 1]
        '''
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        # TODO: should we use mode here or sample (also pass a generator)?
        latents = posterior.sample() * 0.18215
        # latents = posterior.mode()
        return latents

    def decode_latents(self, latents):
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs # (B, 3, H, W)

    def predict_noise(self, text_embs, latents_noisy, t,
                      guidance_scale=7.5):
        with torch.no_grad():
            B = latents_noisy.shape[0]
            latents_noisy = torch.cat([latents_noisy] * 2)
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            noise_pred = self.unet(
                latents_noisy,
                t,
                encoder_hidden_states=text_embs.repeat(B, 1, 1)
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond)
        return noise_pred

    def prompt_to_img(self, prompt, num_inference_steps=50, guidance_scale=7.5,
                      seed=42, custom_sampler=False):
        text_embs = self.embed_text(prompt)
        latents = torch.randn([1, self.unet.in_channels, 512 // 8, 512 // 8],
                              dtype=self.float_dtype,
                              generator=torch.manual_seed(seed)).to(self.device)

        if custom_sampler:
            gam=2
            sc = Scheduler(beta_start= 0.00085, beta_end= 0.012, beta_schedule='scaled_linear')
            sc.set_timesteps_sigma(start=6.57, end=0.1195, num_inference_steps=num_inference_steps, style='Linear')
            eps = None
            for i, (t, t_prev) in enumerate(pairwise(sc.timesteps)):
                eps, eps_prev = self.predict_noise(text_embs, latents, t,guidance_scale=guidance_scale), eps
                eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                latents += (sc.sigma(t_prev) - sc.sigma(t)) * eps_av
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            for t in self.scheduler.timesteps: # t is in [0, num_train_steps]
                latents = self.scheduler.step(
                    self.predict_noise(text_embs, latents, t,
                                       guidance_scale=guidance_scale),
                    t, latents).prev_sample
        return self.decode_latents(latents)[0]

def tensor_to_pil(img):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype("uint8")
    return Image.fromarray(img)

def sample(sd, prompt, seed=0, num_inference_steps=10, guidance_scale=7.5, style='2nd'):
    if style in ('ddim', 'pndm', 'dpm', 'unipc'):
        if style == 'ddim':
            sd.scheduler = DDIMScheduler.from_pretrained(sd.model_key, subfolder='scheduler')
        elif style == 'pndm':
            sd.scheduler = PNDMScheduler.from_pretrained(sd.model_key, subfolder='scheduler')
        elif style == 'dpm':
            sd.scheduler = DPMSolverMultistepScheduler.from_pretrained(sd.model_key, subfolder='scheduler')
        elif style == 'unipc':
            sd.scheduler = UniPCMultistepScheduler.from_pretrained(sd.model_key, subfolder='scheduler')
        return sd.prompt_to_img(prompt, seed=seed, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    text_embs = sd.embed_text(prompt)
    latents = torch.randn([1, sd.unet.in_channels, 64, 64],
                        dtype=sd.float_dtype,
                        generator=torch.manual_seed(seed)).to(sd.device)
    sc = Scheduler(beta_start= 0.00085, beta_end=0.012, beta_schedule='scaled_linear')
    #sc.set_timesteps_sigma(start=14.6, end=0.02, num_inference_steps=num_inference_steps, style='DDIM')
    #sigma_min = sc.sigma(sc.timesteps[-2])
    sc.set_timesteps_sigma(start=8.35, end=0.02, num_inference_steps=num_inference_steps, style='DDIM')
    print(sc.timesteps)
    latents = latents / sc.ap(sc.timesteps[0]).sqrt()
    if style == 'ddim':
        for t, t_prev in pairwise(sc.timesteps):
            eps = sd.predict_noise(text_embs, latents * sc.ap(t).sqrt(), t,guidance_scale=guidance_scale)
            latents += (sc.sigma(t_prev) - sc.sigma(t)) * eps
    elif style == '2nd':
        eps = None
        for i, (t, t_prev) in enumerate(pairwise(sc.timesteps)):
            eps, eps_prev = sd.predict_noise(text_embs, latents * sc.ap(t).sqrt(), t,guidance_scale=guidance_scale), eps
            eps_av = eps * 2 - eps_prev if i > 0 else eps
            latents += (sc.sigma(t_prev) - sc.sigma(t)) * eps_av
    return sd.decode_latents(latents)[0]

def experiment(sd, prompt, seed=0, N=10, saveto=None):
    for style in ('2nd', 'unipc', 'dpm', 'pndm', 'ddim'):
        img = tensor_to_pil(sample(sd, prompt, seed=seed, num_inference_steps=N, style=style))
        if saveto is None:
            display(img)
        else:
            img.save(f'{saveto}_{style}.png')

if __name__ == '__main__':
    # Save pictures used in paper
    sd = StableDiffuser(model_key='stabilityai/stable-diffusion-2-1-base')
    experiment(sd, "A digital Illustration of the Babel tower, 4k, detailed, trending in artstation, fantasy vivid colors",
               seed=4, N=10, saveto='figures/sd_1')
    experiment(sd, "london luxurious interior living-room, light walls",
               seed=3, N=10, saveto='figures/sd_2')
    experiment(sd, "Cluttered house in the woods, anime, oil painting, high resolution, cottagecore, ghibli inspired, 4k",
               seed=0, N=10, saveto='figures/sd_3')
