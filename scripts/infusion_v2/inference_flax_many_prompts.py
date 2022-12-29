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


weight_num = int(sys.argv[1])   #Changes slowly (i)
character_num = int(sys.argv[2]) 


def load_image(image_path):
    urls = [f"{image_path}/00{i}.png" for i in range(33,41)]
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

image_path = "/home/andrew/data/b_cowboy"
bias_instance_images = load_image(image_path)


# bias_pixel_values = [img.to(memory_format=torch.contiguous_format).float() for img in bias_instance_images]



unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
    "/home/andrew/model/infusion_coco", subfolder="unet", dtype=jax.numpy.bfloat16
)

# unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
#     "duongna/stable-diffusion-v1-4-flax", subfolder="unet", dtype=jax.numpy.bfloat16
# )


pipeline, params = FlaxInfusingStableDiffusionPipeline.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", dtype=jax.numpy.bfloat16
)

pipeline.unet = unet
params['unet'] = unet_params
params = replicate(params)


prompt = "Person walking"

# image_path = "/home/andrew/data/b_cowboy_2"
character_names = ["b_shed2", "b_tree4", "b_house", "b_tower", "b_krosh_full", "b_cowboy_3","b_krosh_blankbackground", \
                     "b_fred", "b_fred2",  "b_krosh_dots", "b_krosh_winning"]


weights_list  = [[.3,.3,.3,.3],[.03,.1,.12,.15],[.4,.4,.4,.4],[.2,.2,.2,.2],[.25,.35,.45,.55],[.3,.3,.3,.3]]
prompts  = ["Person", "Person with a tree in the background", "Person in front of a tree", "Tree, Person", "Person in a field"]
#prompts  = [" " , "Tree", "Green Tree", "Person with a tree in the background", "Person in front of a tree", "Tree, Person", "Person in a field", "Person in front of a tree", "Person" , "Person dancing by a tree", "Person sees friends by a tree", "Person hugs friends by a tree"]
prompts  = [" ", " Shed", " House", "Mansion", \
            "A picture of john walking next to a house", \
            "John walking next to a house", \
            "john meeting Jane next to a house", \
            "john talking to Jane next to a house", \
            "John arguinng with Jane next to a house", \
            "A picture of john walking in front of a mansion", \
            "John walking in front of a mansion", \
            "john meeting Jane in front of a mansion", \
            "john talking to Jane in front of a mansion", \
            "John arguinng with Jane in front of a mansion", \
            "A picture of john walking in front of a shed", \
            "John walking in front of a shed", \
            "john meeting Jane in front of a shed", \
            "john talking to Jane in front of a shed", \
            "John arguinng with Jane in front of a shed", \
            "A picture of john walking in front of a house", \
            "John walking in front of a house", \
            "john meeting Jane in front of a house", \
            "john talking to Jane in front of a house", \
            "John arguinng with Jane in front of a house"] #, \
            #"People talking. Background: Small green tree", "People talking in a field", "People talking, Scene: Field with a green tree", "People talking in a field. Background: Tree", "Foreground: People talking. Background: Green Tree", "People talking in a summer field"]

for prompt_idx, character_name in enumerate(character_names):

    prompt = " "
    #character_name = character_names[character_num]
    layer_bias_factors = weights_list[weight_num]

    image_path = f"/home/andrew/data/{character_name}"
    bias_instance_images = load_image(image_path)
    #bias_instance_images[0].save(f"generated/no_text/{character_name}.jpg")
    #prompt = prompts[weights_num]

    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 100 # CHANGE:ITERSTEP IF YOU CHANGE THIS, ALSO CHAGNE THE BIAS FACTOR DEGENERATION

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    # shard inputs and rng
    #params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    prompt_ids = shard(prompt_ids)

    logger.info("Generating images...")
    images = pipeline(prompt_ids, params, prng_seed, bias_instance_images, layer_bias_factors, num_inference_steps, guidance_scale = 7.5, jit=True).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
    prng_seed = jax.random.PRNGKey(1)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    #images2 = pipeline(prompt_ids, params, prng_seed, bias_instance_images, layer_bias_factors, num_inference_steps, guidance_scale = 7.5, jit=True).images
    #images2 = pipeline.numpy_to_pil(np.asarray(images2.reshape((num_samples,) + images2.shape[-3:])))
    logger.info("Done")

    #images = images + images2

    #params = unreplicate(params)

    # for idx, i in enumerate(images):
    #     im = Image.fromarray(i)
    #     i.save(f"{idx}.jpg")

    g = image_grid(images,2,4)
    save_dir = f"generated/chars/3weight_tune_chars/{character_name}_{num_inference_steps}_{layer_bias_factors}_{prompt_idx:02d}{prompt[0]}.jpg"
    g.save(save_dir)
    print("Saving to file...", save_dir)
# character_names = ['b_fred', 'b_fred2', 'b_cowboy', 'b_cowboy', 'b_krosh', 'b_krosh_dots']
# layer_bias_factors_list = [[.01,.03,.04,.05], [.02,.06,.08,.1],[.02,.05,.01,.15]]

# for layer_bias_factors in layer_bias_factors_list:
#     for character_name in character_names:
#         image_path = f"/home/andrew/data/{character_name}"
#         bias_instance_images = load_image(image_path)
#         bias_instance_images[0].save(f"generated/no_text/{character_name}.jpg")
#         prompt = " "

#         prng_seed = jax.random.PRNGKey(0)
#         num_inference_steps = 10

#         num_samples = jax.device_count()
#         prompt = num_samples * [prompt]
#         prompt_ids = pipeline.prepare_inputs(prompt)

#         # shard inputs and rng
#         params = replicate(params)
#         prng_seed = jax.random.split(prng_seed, jax.device_count())
#         prompt_ids = shard(prompt_ids)

#         images = pipeline(prompt_ids, params, prng_seed, bias_instance_images, layer_bias_factors, num_inference_steps, jit=True).images
#         images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
#         params = unreplicate(params)

#         # for idx, i in enumerate(images):
#         #     im = Image.fromarray(i)
#         #     i.save(f"{idx}.jpg")
#         def image_grid(imgs, rows, cols):
#             w,h = imgs[0].size
#             grid = Image.new('RGB', size=(cols*w, rows*h))
#             for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
#             return grid
#         g = image_grid(images,4,2)
#         #g.save("generated/krosh_winning_a_game_01_03_04_05.jpg")
#         g.save(f"generated/no_text/{character_name}_{layer_bias_factors}.jpg")