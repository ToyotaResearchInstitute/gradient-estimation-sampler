# Environment setup

1. Install requirements (requires python 3.10 or above)

```bash
pip install -r requirements.txt
```

2. Download model checkpoints and FID statistics from
   [https://github.com/luping-liu/PNDM/](https://github.com/luping-liu/PNDM/).
   Requires following files before running the code:
     - `models/ddim_celeba.ckpt`
     - `models/ddim_cifar10.ckpt`
     - `fid/fid_celeba_train.npz`
     - `fid/fid_cifar10_train.npz`

3. Download COCO dataset and modify constants in `coco_experiment.py` to point to dataset

# To reproduce results

1. For FID evaluations and plots, run

```bash
python run_experiments.py
```

   Plots will be generated in `figures`, FID scores will be logged in `results`
   (viewable with tensorboard), and Images will be generated in `images`.

2. For ideal denoiser experiments, run

```bash
python ideal_denoiser.py
```

   Plots will be generated in `figures`.

3. For MS-COCO experiments, run

```bash
python coco_experiment.py
```

4. For text-to-image generation with Stable Diffusion, run

```bash
python stable_diffusion_exp.py
```

   Images used in the paper will be generated in `figures`.
