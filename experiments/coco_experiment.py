import os
import json
import torch
import sys
import shutil
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from scheduling_gradient_estimation import GEScheduler
from experiment_utils import fid

# Edit paths to COCO dataset before running
COCO_ANNO_FILE = '/path/to/captions_val2014.json'
FID_REFERENCE = '/path/to/coco_val.npz'

def main(scheduler, folder, start=0, end=30000, steps=10, sch_config=None):
    sch_config = sch_config or {}
    if not os.path.exists(folder):
        print(f'Making new directory: {folder}')
        os.makedirs(folder)
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base').to('cuda')
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **sch_config)
    pipe.scheduler.set_timesteps(steps)
    print('Timesteps:', pipe.scheduler.timesteps)
    captions = json.load(open(COCO_ANNO_FILE))
    anno_dict = {a['image_id']: a['caption'] for a in captions['annotations']}
    pipe.set_progress_bar_config(disable=True)

    gen = torch.manual_seed(0)
    for img_info in tqdm(captions['images'][start:end]):
        img_id = img_info["id"]
        img = pipe(anno_dict[img_id], num_inference_steps=steps, generator=gen).images[0]
        img.save(f'{folder}/{img_id}.png')

def copy_subset(folder_from, folder_to, start=0, end=30000):
    if not os.path.exists(folder_to):
        print(f'Making new directory: {folder_to}')
        os.makedirs(folder_to)
    captions = json.load(open(COCO_ANNO_FILE))
    anno_dict = {a['image_id']: a['caption'] for a in captions['annotations']}

    for img_info in tqdm(captions['images'][start:end]):
        img_id = img_info["id"]
        shutil.copy(f'{folder_from}/{img_id}.png', folder_to)

if __name__ == '__main__':
    schedulers = [GEScheduler, DDIMScheduler, PNDMScheduler,
                  DPMSolverMultistepScheduler, UniPCMultistepScheduler]
    outputs = ['coco_results_ge', 'coco_results_ddim', 'coco_results_pndm', 'coco_results_dpm', 'coco_results_unipc']

    ## Generate images
    for scheduler, output in zip(schedulers, outputs):
        main(scheduler, output, start=0, end=30000, steps=10)

    ## Compute FID
    for output in outputs:
        print(output, fid(output, FID_REFERENCE, batch_size=128))
