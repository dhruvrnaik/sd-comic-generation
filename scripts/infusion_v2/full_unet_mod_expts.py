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

from hardcoded_stuff.clip_captions_dict import img_to_caption_dict


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

def gen_image(character_name, prompt = None ,weights = [.2,.2,.2,.2], bias_decay = .99, guidance_scale = 7.5):
    only_char_name = character_name.split("/")[-1]
    
    if(prompt == None):
        prompt = " "
        if("_base" in only_char_name and only_char_name[:-5] in img_to_caption_dict):
            prompt = img_to_caption_dict[only_char_name[:-5]]
        elif(only_char_name in img_to_caption_dict):
            prompt = img_to_caption_dict[only_char_name]
        else:
            logger.info("Image Name not found in list of prompts. Going with empty prompt")
    print(prompt)
    
    

    layer_bias_factors = weights

    image_path = f"{character_name}"
    bias_instance_images = load_image(image_path)

    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 50

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    prng_seed = jax.random.split(prng_seed, jax.device_count())
    prompt_ids = shard(prompt_ids)

    logger.info("Generating images...")
    images = pipeline(prompt_ids, params, prng_seed, bias_instance_images, \
                      layer_bias_factors, bias_decay,  num_inference_steps, guidance_scale = guidance_scale, jit=True).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
    prng_seed = jax.random.PRNGKey(1)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    logger.info("Done")
    g = image_grid(images,2,4)
    
    if not os.path.exists(f"experiments/disappearing/{only_char_name}"):
        os.makedirs(f"experiments/disappearing/{only_char_name}")
    save_dir = f"experiments/disappearing/{only_char_name}/{num_inference_steps}_5_exp_reduc_in_time_{layer_bias_factors[:2]}_{0:02d}.jpg"
    print("Saving to file...", save_dir)
    g.save(save_dir)
    return g

trees = ["green_c_tree", "xmas_c_tree", "red_r_tree", "green_r_tree", \
         "xmas_r_tree", "palm_r_tree", "red_c_tree", "palm_c_tree" ]

trees += ["fox_c_animal", "doc_c_animal", "sqrl_c_animal", "elep_c_animal", \
         "doc_r_animal", "rabbit_r_animal", "cat_r_animal", "elep_r_animal" ]

trees += ["girl_c_pers", "teach_c_pers", "white_c_pers", "profile_c_pers", \
         "man_r_pers", "lady_r_pers", "old_r_pers", "profile_r_pers" ]
         
trees += ["red_c_house", "sketch_house", "wood_c_house", "brick_c_house", \
         "big_r_house", "white_house", "big_c_house", "style_r_house" ]




# trees = ["b_cowboy", "b_cowboy_2", "b_krosh_full"]



for folder in trees:
    weights = [0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, -0.0158691, \
                   -0.03125, -0.03125, 0.03125, 0.03125, 0.03125]
    #weights = [5*i for i in weights]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=1)


    #weights = [1e-5]*13
    #gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = "tree", weights = weights, bias_decay=.5)
import pdb; pdb.set_trace()


cats



for folder in trees:
    weights = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.15, .15, .15, .15, .15, .15, .15, .15, .15, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .15, .15, .15, .15, .15, .15, .15, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .1, .2, .2, .2, .2, .2, .1, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .1, .1, .1, .1, .1, .1, .1, .1, -.001, -.1, -.1, -.1]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.3, .1, .1, .1, .1, .1, .1, .1, .3, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .3, .1, .1, .1, .1, .1, .3, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .1, .3, .1, .1, .1, .3, .1, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    weights = [.1, .1, .1, .3, .1, .3, .1, .1, .1, .0001, .0001, .0001, .0001]
    gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)
    
    for i in range(13):
        weights = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .0001, .0001, .0001, .0001]
        weights[i] = .5
        gen_image(f'/home/andrew/data/imgs_on_white/{folder}',prompt = None, weights = weights, bias_decay=.99)