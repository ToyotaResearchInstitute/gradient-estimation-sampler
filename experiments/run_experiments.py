import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, VQModel
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from torchvision.transforms import functional as TF
from torchvision.datasets import CIFAR10
from pytorch_fid.inception import InceptionV3
from torch.utils.tensorboard import SummaryWriter
from itertools import islice, pairwise

from experiment_utils import *
from schedulers import *

device = 'cuda:0'

class ExperimentMixin:
    def __init__(self, seed, B, device, save_folder):
        assert self.model is not None, 'Model needs to be initialized!'
        self.device = device
        self.seed = seed
        self.B = B
        C = self.model.in_channels
        H = W = self.model.sample_size
        self.shape = (B, C, H, W)
        self.dim = np.sqrt(C * H * W)
        self.save_folder = save_folder

    def bt(self, t):
        return batched_t(self.B, t, self.device)

    def new_gen(self, seed=None):
        return torch.manual_seed(self.seed if seed is None else seed)

    def get_noise(self, gen=None):
        return torch.randn(self.shape, generator=gen).to(self.device)

    def get_save_to(self, filename):
        return f'{self.save_folder}/{filename}' if self.save_folder is not None else None

class Experiment(ExperimentMixin):
    def __init__(self,
                 repo_id,
                 config_path=None,
                 device='cuda',
                 ddim=False,
                 B=32, # Batch size
                 seed=0,
                 num_inference_steps=100,
                 save_folder=None,
                 ):
        if ddim:
            self.model = DDIMModel(
                config_path=config_path,
                model_path=repo_id,
                device=device,
            )
            self.scheduler = DDIMScheduler.from_config('google/ddpm-cifar10-32')
        else:
            self.model = UNet2DModel.from_pretrained(repo_id).to(device)
            self.scheduler = DDIMScheduler.from_config(repo_id)
        self.T = num_inference_steps
        self.sc = Scheduler()
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        ExperimentMixin.__init__(self, seed, B, device, save_folder)

    def get_dataset_imgs(self, dataset, nbatches):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.B)
        return [img.to(self.device) for img, _ in islice(loader, nbatches)]

    def samples_second_order(self, zt, start=40, end=0.06, steps=10, gam=2, style='Linear', display_every=10):
        sc_kwargs = dict(start=start, end=end, num_inference_steps=steps, style=style)
        self.sc.set_timesteps_sigma(**sc_kwargs)
        xt = zt / self.sc.ap(self.sc.timesteps[0]).sqrt()
        eps = None
        for i, (t, t_prev) in enumerate(pairwise(self.sc.timesteps)):
            zt = xt * self.sc.ap(t).sqrt()
            eps, eps_prev = self.model(zt, t).sample, eps
            eps_av = eps * gam + eps_prev * (1 - gam)  if i > 0 else eps
            xt_prev, xt = xt, xt + (self.sc.sigma(t_prev) - self.sc.sigma(t)) * eps_av
            if (i + 1) % display_every == 0:
                z0_pred = (xt_prev - self.sc.sigma(t) * eps_av)
                yield xt, eps, z0_pred

    def samples(self, sample, custom_timesteps=None, display_every=10):
        timesteps = self.scheduler.timesteps if custom_timesteps is None else custom_timesteps
        for i, t in enumerate(timesteps):
            # 1. predict noise residual
            with torch.no_grad():
               residual = self.model(sample, t).sample

            # 2. compute previous image and set x_t -> x_t-1
            res = self.scheduler.step(residual, t, sample)
            sample = res.prev_sample

            # 3. optionally look at image
            if (i + 1) % display_every == 0:
                yield sample, residual, res.pred_original_sample

    def plot_denoising_norm(self, nbatches=100, gen=None):
        imgs = []
        with Plotter(hist=True, save_to=self.get_save_to('denoising_norm.png')) as p:
            for _ in tqdm(range(nbatches)):
                noise = self.get_noise(gen)
                res = self.samples(noise, display_every=1)
                for t, (x0, residual, _) in zip(self.scheduler.timesteps, res):
                    p.add('eps_hat norm', t.item(), norm(residual)/self.dim)
                    imgs.append(x0)
            plt.xlabel('t')
        return imgs

    def plot_denoiser_error(self, imgs, step_every=10, gen=None):
        with Plotter(hist=True, save_to=self.get_save_to('denoiser_error.png')) as p:
            for img in tqdm(imgs):
                noise = self.get_noise(gen)
                for t in range(0, self.scheduler.num_train_timesteps, step_every):
                    pred = self.model(self.scheduler.add_noise(img, noise, self.bt(t)), self.bt(t)).sample
                    p.add('Norm of noise - eps', t, norm(pred - noise)/self.dim)
            plt.axhline(y=1)
            plt.xlabel('t')

class SamplerExperiment(ExperimentMixin):
    def __init__(self,
                 config_path='./config/ddim_cifar10.yml',
                 model_path='./models/ddim_cifar10.ckpt',
                 fid_target='./fid/fid_cifar10_train.npz',
                 device='cuda',
                 B=64, # Batch size
                 seed=0,
                 save_folder='./',
                 name='',
                 ):

        self.model = DDIMModel(
            config_path=config_path,
            model_path=model_path,
            device=device,
        )
        self.fid_target = fid_target
        self.sc = Scheduler()
        self.logger = SummaryWriter(os.path.join(save_folder, name))
        ExperimentMixin.__init__(self, seed, B, device, save_folder)

    def get_batches(self, nimages):
        return tqdm(range(nimages // self.B))

    def model_x(self, xt, t):
        zt = xt * self.sc.ap(t).sqrt()
        return self.model(zt, t).sample

    def log_fid(self, saver, params):
        self.logger.add_hparams(params, dict(fid=saver.fid_score))

    def get_noise_x(self, gen=None):
        return self.get_noise(gen=gen) / self.sc.ap(self.sc.timesteps[0]).sqrt()

    def sampler_ddim(self, nimages, start=20, end=0.1, steps=10, style='Linear', gen=None):
        gen = gen or self.new_gen()
        if end is None:
            start_sigma_ddim = self.sc.sigma(self.sc.num_train_timesteps-1)
            self.sc.set_timesteps_sigma(start=start_sigma_ddim, end=0, num_inference_steps=steps, style='DDIM')
            end = self.sc.get_end_sigma().item()
        sc_kwargs = dict(start=start, end=end, num_inference_steps=steps, style=style)
        self.sc.set_timesteps_sigma(**sc_kwargs)
        with Saver('images/fid_results_ddim', target=self.fid_target) as s:
            for _ in self.get_batches(nimages):
                xt = self.get_noise_x(gen=gen)
                for t, t_prev in pairwise(self.sc.timesteps):
                    xt += (self.sc.sigma(t_prev) - self.sc.sigma(t)) * self.model_x(xt, t)
                s.save(xt)
        self.log_fid(s, params=(sc_kwargs | dict(seed=self.seed)))

    def sampler_second_order(self, nimages, start=40, end=None, steps=10, gam=2, mu=0,
                             style='Linear', gen=None, gen_w=None):
        gen = gen or self.new_gen()
        gen_w = gen_w or self.new_gen()
        if end is None:
            start_sigma_ddim = self.sc.sigma(self.sc.num_train_timesteps-1)
            self.sc.set_timesteps_sigma(start=start_sigma_ddim, end=0, num_inference_steps=steps, style='DDIM')
            end = self.sc.get_end_sigma().item()
        sc_kwargs = dict(start=start, end=end, num_inference_steps=steps, style=style)
        self.sc.set_timesteps_sigma(**sc_kwargs)
        with Saver('images/fid_results_secondorder', target=self.fid_target) as s:
            for _ in self.get_batches(nimages):
                xt = self.get_noise_x(gen=gen)
                eps = None
                for i, (t, t_prev) in enumerate(pairwise(self.sc.timesteps)):
                    sig_prev, sig = self.sc.sigma(t_prev), self.sc.sigma(t)
                    eps, eps_prev = self.model_x(xt, t), eps
                    eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                    sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                    eta = (sig_prev**2 - sig_p**2).sqrt()
                    xt = xt - (sig - sig_p) * eps_av + eta * self.get_noise(gen=gen_w)
                s.save(xt)
        self.log_fid(s, params=(sc_kwargs | dict(gam=gam, seed=self.seed, nimages=nimages, mu=mu)))

def get_experiment_cifar10(B=128, seed=0, name='', save_folder='results'):
    return SamplerExperiment(
        config_path = './config/ddim_cifar10.yml',
        model_path  = './models/ddim_cifar10.ckpt',
        fid_target  = './fid/fid_cifar10_train.npz',
        save_folder = save_folder,
        name=name,
        B = B,
        seed = seed,
    )

def get_experiment_celeba(B=128, seed=0, name='', save_folder='results'):
    return SamplerExperiment(
        config_path = './config/ddim_celeba.yml',
        model_path  = './models/ddim_celeba.ckpt',
        fid_target  = './fid/fid_celeba_train.npz',
        save_folder = save_folder,
        name=name,
        B = B,
        seed=seed,
    )

def get_cifar_test():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return CIFAR10('cifar', download=True, train=False, transform=transform)

def denoiser_exp(nbatches=100):
    exp = Experiment('google/ddpm-cifar10-32', B=128, save_folder='figures')
    exp.plot_denoising_norm(nbatches=nbatches, gen=exp.new_gen())
    cifar_test_images = exp.get_dataset_imgs(get_cifar_test(), nbatches)
    exp.plot_denoiser_error(cifar_test_images, step_every=10, gen=exp.new_gen())

def schedule_exp(nimages=50000):
    # Vary schedules for DDIM sampler as well as second-order sampler
    exp_cifar = get_experiment_cifar10(name='sigma_cifar_fid')
    exp_celeba = get_experiment_celeba(name='sigma_celeba_fid')
    styles = [
        dict(style='DDIM', start=157, end=0.002),
        dict(style='DDIM', start=40, end=0.002),
        dict(style='Linear', start=40, end=None),
        dict(style='EDM', start=80, end=0.002),
    ]
    for steps in (5, 10, 20, 50):
        for kwargs in styles:
            kwargs['steps'] = steps
            print(kwargs)
            exp_cifar.sampler_ddim(nimages, gen=exp_cifar.new_gen(), **kwargs)
            exp_cifar.sampler_second_order(
                nimages, gen=exp_cifar.new_gen(), gam=2.0, **kwargs
            )
            exp_celeba.sampler_ddim(nimages, gen=exp_celeba.new_gen(), **kwargs)
            exp_celeba.sampler_second_order(
                nimages, gen=exp_celeba.new_gen(), gam=2.0, **kwargs
            )

def plot_sigma_schedule(save_folder='figures', N=10):
    # Plots sigma schedule on a log-scale for N=10
    sc = Scheduler()
    sc.set_timesteps_sigma(start=157, end=0.002, num_inference_steps=N, style='DDIM')

    plt.clf()
    plt.plot(sc.timesteps_sigma.log().numpy(), label='DDIM')

    sc.set_timesteps_sigma(start=40, end=0.002, num_inference_steps=N, style='DDIM')
    plt.plot(sc.timesteps_sigma.log().numpy(), label='DDIM offset')

    sc.set_timesteps_sigma(start=80, end=0.002, num_inference_steps=N, style='EDM')
    plt.plot(sc.timesteps_sigma.log().numpy(), label='EDM')

    sc.set_timesteps_sigma(start=sc.sigma(sc.num_train_timesteps-1), end=0,
                           num_inference_steps=N, style='DDIM')
    end = sc.get_end_sigma().item()
    sc.set_timesteps_sigma(start=40, end=end, num_inference_steps=N, style='Linear')
    plt.plot(sc.timesteps_sigma.log().numpy(), label='Ours')

    plt.xlabel('$t$')
    plt.ylabel('$log(\sigma_t)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_folder}/sigma_schedule.png')

def ddim_full_exp(nimages=50000, seed=0, start=157, end=0.002):
    # Reproduce DDIM baseline
    exp_cifar = get_experiment_cifar10(seed=seed, name='ddim_celeba_fid')
    exp_celeba = get_experiment_celeba(seed=seed, name='ddim_cifar_fid')
    for steps in (5, 10, 20, 50):
        exp_cifar.sampler_ddim(
            nimages, start=start, end=end, steps=steps, style='DDIM',
        )
        exp_celeba.sampler_ddim(
            nimages, start=start, end=end, steps=steps, style='DDIM',
        )

def second_order_full_experiment(nimages=50000, gam=2, seed=0, start=40):
    # Our gradient-estimation sampler with best parameters
    exp_cifar = get_experiment_cifar10(seed=seed, name='second_order_cifar_fid')
    for steps in (5, 10, 20, 50):
        exp_cifar.sampler_second_order(
            nimages, start=start, steps=steps, style='Linear', gam=gam
        )
    exp_celeba = get_experiment_celeba(seed=seed, name='second_order_celeba_fid')
    for steps, gam, start in [(5,  2.0, 40 ),
                              (10, 2.0, 80 ),
                              (20, 2.4, 100),
                              (50, 2.8, 120)]:
        exp_celeba.sampler_second_order(
            nimages, start=start, steps=steps, style='Linear', gam=gam
        )

def sampler_gamma_exp(nimages=50000, seed=0, start=40):
    # Varying gamma for gradient-estimation sampler
    exp_cifar = get_experiment_cifar10(seed=seed, name='second_order_cifar_gamma')
    exp_celeba = get_experiment_celeba(seed=seed, name='second_order_celeba_gamma')

    for steps in (5, 10, 20):
        for gam in np.linspace(1, 3, 21):
            exp_cifar.sampler_second_order(
                nimages, start=start, steps=steps, style='Linear', gam=gam
            )
            exp_celeba.sampler_second_order(
                nimages, start=start, steps=steps, style='Linear', gam=gam
            )

if __name__ == '__main__':
    denoiser_exp(50)
    schedule_exp()
    plot_sigma_schedule(N=10)
    ddim_full_exp()
    second_order_full_experiment()
    sampler_gamma_exp()
