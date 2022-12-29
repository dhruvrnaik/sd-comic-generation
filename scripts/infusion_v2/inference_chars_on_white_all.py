import jax
import numpy as np
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard


#from diffusers import FlaxStableDiffusionPipeline
from infusion_models.flax_infusion_pipeline import FlaxInfusingStableDiffusionPipeline
from infusion_models.flax_infusion_pipeline import FlaxInfusionUNetModel
#from infusion_models.pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
from PIL import Image

from torchvision import transforms

import argparse
import sys

import time
from loguru import logger
import os


def load_image(image_path):
    urls = [f"{image_path}/{i}.png" for i in range(8)]
    images = list(Image.open(file_path).convert('RGB') for file_path in urls)
    bias_images = images

    size = 512
    image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            )

    bias_instance_images = [image_transforms(i).numpy() for i in bias_images]
    return bias_instance_images

def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid



# bias_pixel_values = [img.to(memory_format=torch.contiguous_format).float() for img in bias_instance_images]



### LOAD MODEL
# unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
#     "/home/andrew/model/infusion_coco", subfolder="unet", dtype=jax.numpy.bfloat16
# )
unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", subfolder="unet", dtype=jax.numpy.bfloat16
)
pipeline, params = FlaxInfusingStableDiffusionPipeline.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", dtype=jax.numpy.bfloat16
)
pipeline.unet = unet
params['unet'] = unet_params
params = replicate(params)


prompt = "Person walking"

root_folder = "/home/andrew/data/imgs_on_white"
img_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

def gen_image(character_name, prompt = " ",weights = [.2,.2,.2,.2], bias_decay = .99):
    only_char_name = character_name.split("/")[-1]

    layer_bias_factors = weights

    image_path = f"{character_name}"
    bias_instance_images = load_image(image_path)

    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 1000

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    prng_seed = jax.random.split(prng_seed, jax.device_count())
    prompt_ids = shard(prompt_ids)

    logger.info("Generating images...")
    images = pipeline(prompt_ids, params, prng_seed, bias_instance_images, layer_bias_factors, bias_decay,  num_inference_steps, guidance_scale = 7.5, jit=True).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
    prng_seed = jax.random.PRNGKey(1)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    logger.info("Done")
    g = image_grid(images,2,4)
    
    save_dir = f"experiments/generated/{only_char_name}_{num_inference_steps}_{layer_bias_factors}_{0:02d}{prompt[0]}.jpg"
    print("Saving to file...", save_dir)
    g.save(save_dir)
    return g

weights = [.4,.4,.4,.4]
for prompt_idx, character_name in enumerate(img_folders):
    folder = character_name.split("/")[-1]
    prompt = " "
    if("tree" in character_name):
        prompt = "tree"
    if("pers" in character_name):
        prompt = "person"
    if("house" in character_name):
        prompt = "house"
    if("animal" in character_name):
        prompt = "animal"
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}', prompt = prompt, weights = weights, bias_decay = .999)