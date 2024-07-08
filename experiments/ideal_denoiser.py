import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from torchvision import transforms
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import pairwise
from smalldiffusion import IdealDenoiser, MappedDataset, ScheduleDDPM

from schedulers import Scheduler

DATASET_PATH = 'cifar'

def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

def projections(dataset, loader, x):
    # x: b x c x w x h
    # Returns p, a tensor with same shape as x, so that p[i] is projection of x[i] onto dataset
    x_flat = x.flatten(start_dim=1)
    xb, xr = x_flat.shape
    diffs = []
    for (data, _) in loader:
        d_flat = data.flatten(start_dim=1).to(x)
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        # ||x - x0|| ,shape xb x db
        diffs.append((sq_norm(x_flat, db) + sq_norm(d_flat, xb).T - 2 * x_flat @ d_flat.T).sqrt())
    dists = torch.cat(diffs, dim=1)
    min_indices = dists.min(dim=1).indices
    min_dists = dists.min(dim=1).values
    projs = torch.stack([dataset[i][0] for i in min_indices]).to(x)
    return min_dists, projs, dists

def samples_ddim(model, sigmas, batchsize):
    accelerator = Accelerator()
    xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0]
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps = model(xt, sig.to(xt))
        x0_pred = xt - sig * eps
        xt = xt - (sig - sig_prev) * eps
        yield xt, x0_pred

def get_ideal_model_sigmas(N):
    accelerator = Accelerator()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    cifar_train = CIFAR10(DATASET_PATH, train=True, transform=transform)
    dataset  = accelerator.prepare(MappedDataset(cifar_train, lambda x: x[0]))
    uloader  = DataLoader(cifar_train, batch_size=512, shuffle=False)
    model    = IdealDenoiser(dataset)
    sigmas   = ScheduleDDPM(1000).sample_sigmas(N)
    return model, sigmas, cifar_train, uloader

def save(saveto, iters=20, batchsize=500, N=50):
    sch = Scheduler()
    model, sigmas, unmapped_dataset, unmapped_loader = get_ideal_model_sigmas(N)

    def t_to_sigma(t):
        return (1/sch.alphas_cumprod[torch.tensor(t, dtype=torch.long)]-1).sqrt().numpy()

    series = defaultdict(list)

    for _ in range(iters):
        errs = []
        nus = []
        for sig, (xt, x0_pred) in tqdm(zip(sigmas, samples_ddim(model, sigmas, batchsize))):
            min_dists, x0_proj, dists = projections(unmapped_dataset, unmapped_loader, xt)
            errs.append((x0_pred-x0_proj).flatten(start_dim=1).norm(dim=1)/min_dists)
            n = torch.prod(torch.tensor(xt.shape[1:]))
            nus.append((sig * n.sqrt()) / min_dists)
        xy = torch.stack(errs).cpu().numpy()

        for x, nu, sigma in zip(xy, nus, sigmas):
            t = sch.sigma_to_t(sigma).item()
            series['Relative error'].append((t, x))
            series['nu'].append((t, nu))

    with open(saveto, 'wb') as f:
        pickle.dump(series, f, pickle.HIGHEST_PROTOCOL)

def plot(data_file, figure_file, N=50):
    sch = Scheduler()
    fwd = np.vectorize(lambda t: sch.sigma(int(t)-1).item(), otypes=[float])
    inv = np.vectorize(lambda s: sch.sigma_to_t(s).item()+1, otypes=[float])

    model, sigmas, unmapped_dataset, unmapped_loader = get_ideal_model_sigmas(N)
    torch.manual_seed(8) # 28: Horse on white background
    res = list(samples_ddim(model, sigmas, 1))
    x0s = torch.cat([r[1] for r in res])
    xts = torch.cat([r[0] for r in res])
    example_img = TF.to_pil_image(make_grid(reversed(x0s[::7]), nrow=10, padding=0).add(1).div(2).clamp(0, 1))

    min_dists, x0_proj, dists = projections(unmapped_dataset, unmapped_loader, xts)
    ex_y = ((x0s-x0_proj).flatten(start_dim=1).norm(dim=1)/min_dists).cpu().numpy()
    ex_x = [sch.sigma_to_t(sig).item() for sig in sigmas[:-1]]

    with open(data_file, 'rb') as f:
        series = pickle.load(f)['Relative error']
    cmap  = 'plasma_r'
    cmin  = 15

    xs, ys = zip(*series)
    X_t, Y = map(np.concatenate, zip(*((np.ones_like(y) * x, y) for x, y in zip(xs, ys))))

    nbins = len(set(X_t))
    cmax  = len(Y) // nbins

    fig, ax = plt.subplots(1,1, figsize=(6, 5))

    # Auto histogram
    # hist = ax.hist2d(X_t, Y, cmap=cmap, bins=nbins, cmin=1)
    # Manual histogram
    xbins = np.array(sorted(set(X_t)))
    ybins = np.linspace(min(Y), max(Y), nbins+1)
    h, _, _ = np.histogram2d(X_t, Y, bins=(xbins, ybins))
    h[h < cmin] = None
    h[h > cmax] = cmax
    ax.pcolormesh(xbins, ybins, h.T, cmap=cmap)
    ax.set_xlabel('t')
    ax.set_ylabel('Relative error (ideal denoiser vs projection)')

    ax2 = ax.secondary_xaxis('top', functions=(fwd, inv))
    ax2.set_xticks(np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax2.set_xlabel('$\\sigma$')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=0, vmax=1)
    cb = fig.colorbar(sm, ax=ax)
    cb.ax.set_xlabel('Density', loc='left')

    im = OffsetImage(example_img, zoom=1.1)
    ab = AnnotationBbox(im, (0.5, 0.88), xycoords='axes fraction', frameon=False, pad=0)
    ax.add_artist(ab)

    ax.plot(ex_x, ex_y, c='w', linestyle='dashed')

    plt.tight_layout()
    plt.savefig(figure_file)

if __name__ == '__main__':
    save('results/ideal_exp.pkl')
    plot('results/ideal_exp.pkl', 'figures/ideal_denoiser_error.png')
