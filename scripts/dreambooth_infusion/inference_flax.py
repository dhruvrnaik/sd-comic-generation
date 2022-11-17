import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline
from infusion_models.flax_infusion_pipeline import FlaxInfusingStableDiffusionPipeline
from PIL import Image

pipeline, params = FlaxInfusingStableDiffusionPipeline.from_pretrained(
    "duongna/stable-diffusion-v1-4-flax", dtype=jax.numpy.bfloat16
)

prompt = "Cowboy with a hat"

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
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
g.save("test_after_tuning.jpg")