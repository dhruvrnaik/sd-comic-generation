import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

#from diffusers import FlaxStableDiffusionPipeline
from infusion_models.flax_infusion_pipeline import FlaxInfusingStableDiffusionPipeline, FlaxInfusionUNetModel
#from infusion_models.pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
from PIL import Image

from torchvision import transforms

urls = [f"/home/andrew/data/cowboy/Cowboy{i}.png" for i in range(1,15)]
images = list(Image.open(file_path).convert('RGB') for file_path in urls)
bias_images = images[11:]

size = 512
image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size)
                #transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5])
            ]
        )

bias_instance_images = [image_transforms(i) for i in bias_images]


# bias_pixel_values = [img.to(memory_format=torch.contiguous_format).float() for img in bias_instance_images]



unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", subfolder="unet", dtype=jax.numpy.bfloat16
)

pipeline, params = FlaxInfusingStableDiffusionPipeline.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", dtype=jax.numpy.bfloat16
)

pipeline.unet = unet
params['unet'] = unet_params







prompt = "Cowboy riding a horse"

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, bias_instance_images, [.05,.05,.05,.05], num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))

# for idx, i in enumerate(images):
#     im = Image.fromarray(i)
#     i.save(f"{idx}.jpg")
def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
g = image_grid(images,4,2)
g.save("cowboy_riding_a_horse.jpg")