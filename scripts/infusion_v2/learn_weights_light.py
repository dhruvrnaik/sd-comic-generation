import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional
import random
import json 

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from coco_dataset import COCOCropDataset

import jax
import jax.numpy as jnp
import optax
import transformers
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler
    #FlaxUNet2DConditionModel
    #FlaxStableDiffusionPipeline
)
from jax import lax

from infusion_models.flax_infusion_pipeline import FlaxInfusingStableDiffusionPipeline
from infusion_models.flax_infusion_unet_light import FlaxInfusionUNetModel
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel, set_seed
from jax_smi import initialise_tracking



logger = logging.getLogger(__name__)
DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="/mnt/disks/persist/COCO14",
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args

size = 512
image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        vids_path = "flintstones_dataset/video_frames"
        json_name = "flintstones_annotations_v1-0.json"

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        data_root = "/mnt/disks/persist"
        vid_dataset = "flintstones_dataset/video_frames"

        jsons = json.load(open(os.path.join(data_root, json_name)))
        id_to_description = {i['globalID']:i['description'] for i in jsons}

        vid_keys = list(id_to_description.keys())
        vid_keys.sort()

        self.image_paths = []
        self.descriptions = []

        for key in vid_keys:
            vid_path = os.path.join(data_root, vids_path , key+".npy")
            self.image_paths.append(vid_path)
            self.descriptions.append(id_to_description[key])
            # np_vid = np.load(vid_path)
            # images.append(Image.fromarray(np_vid[40]))
        self._length = len(self.image_paths)

        print(self._length)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length -1

    def __getitem__(self, index):
        example = {}
        index = index + 1

        np_vid = np.load(self.image_paths[index])
        instance_image = Image.fromarray(np_vid[40])
        instance_prompt = self.descriptions[index]

        # instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)


        np_vid = np.load(self.image_paths[index-1])
        instance_image = Image.fromarray(np_vid[40])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["bias_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example


def LoadBiasingImages(biassing_data_root):
    """
    Loads a list of images form the dir
    Returns a list of images to match "Instance Images" in a batch
    """
    biassing_imgs = []
    bias_img_paths = list(Path(biassing_data_root).iterdir())
    for bias_img_path in bias_img_paths:
        instance_image = Image.open(bias_img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        biassing_imgs.append(image_transforms(instance_image))
    return biassing_imgs

# def combine_BiassingImges(biassing_imgs):
#     bias_pixel_values = torch.stack(biassing_imgs)
#     bias_pixel_values = bias_pixel_values.to(memory_format=torch.contiguous_format).float()
#     return bias_pixel_values.numpy().astype(dtype = np.float16)


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = parse_args()
    initialise_tracking()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline, params = FlaxInfusingStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, safety_checker=None
            )
            pipeline.unet = unet
            params['unet'] = unet_params
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            total_sample_batch_size = args.sample_batch_size * jax.local_device_count()
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not jax.process_index() == 0
            ):
                prompt_ids = pipeline.prepare_inputs(example["prompt"])
                if(not DEBUG):
                    prompt_ids = shard(prompt_ids)
                    p_params = jax_utils.replicate(params)
                else:
                    p_params = params
                rng = jax.random.split(rng)[0]
                sample_rng = jax.random.split(rng, jax.device_count())
                images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
                images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
                images = pipeline.numpy_to_pil(np.array(images))

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    train_dataset = COCOCropDataset(
        data_root="/mnt/disks/persist/COCO14",
        tokenizer=tokenizer,
        size=args.resolution,
    )



    bias_img_path = "/home/andrew/data/b_krosh"
    b_img_path = LoadBiasingImages(bias_img_path)
    biassing_images =  LoadBiasingImages(bias_img_path)
    #biassing_pixel_values = combine_BiassingImges(biassing_images)

    print(f"dataset: {len(train_dataset)}")
    print(f"dir: {args.instance_data_dir}")

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        bias_pixel_values = [example["bias_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        bias_pixel_values = torch.stack(bias_pixel_values)
        bias_pixel_values = bias_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "bias_pixel_values": bias_pixel_values,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=total_train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    print(f"weight type: {weight_dtype}")
    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", dtype=weight_dtype
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", dtype=weight_dtype
    )


    UNET_PATH="/home/andrew/model/infusion_coco"
    unet, unet_params = FlaxInfusionUNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", dtype=weight_dtype
        # UNET_PATH, subfolder="unet", dtype=weight_dtype
    )
    # [-0.0106812 -0.0546875 -0.480469 0.0947266 -0.00107574 0.160156 0.243164
    # -0.00665283 0.0388184 -0.135742 -0.53125 -0.0688477 -0.118652]
    all_unet_params = unet.init_weights(rng)
    unet_params['layer_biases'] = all_unet_params['layer_biases'].astype(weight_dtype)
    del all_unet_params

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    def create_mask(params, label_fn):
        def _map(params, mask, label_fn):
            for k in params:
                if label_fn(k):
                    mask[k] = "layer_biases"
                else:
                    if isinstance(params[k], dict):
                        mask[k] = {}
                        _map(params[k], mask[k], label_fn)
                    else:
                        mask[k] = "zero"

        mask = {}
        _map(params, mask, label_fn)
        return mask
        
    def zero_grads():
        # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.tree_util.tree_map(jnp.zeros_like, updates), ()

        return optax.GradientTransformation(init_fn, update_fn)

    # Zero out gradients of layers other than the token embedding layer
    tx = optax.multi_transform(
        {"layer_biases": optimizer, "zero": zero_grads()},
        create_mask(unet_params, lambda s: s == "layer_biases"),
    )
    #import pdb; pdb.set_trace()

    #logger.info(unet_params['layer_biases'])

    #optimizer = optax.masked(optax.sgd(0.1), {"params": {"Dense_0": False, "Dense_1": False, "Dense_2": True}})


    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=tx)
    text_encoder_state = train_state.TrainState.create(
        apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=50
    )

    # Initialize our training
    if(not DEBUG):
        train_rngs = jax.random.split(rng, jax.local_device_count())
    else:
        train_rngs = rng
    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        if args.train_text_encoder:
            params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}
        else:
            params = {"unet": unet_state.params}


        def breakpoint_if_nonfinite(x):
            is_finite = jnp.isfinite(x).all()
            def true_fn(x):
                pass
            def false_fn(x):
                jax.debug.breakpoint()
            lax.cond(is_finite, true_fn, false_fn, x)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * 0.18215

            #biasing_latent_dists = [self.vae.apply({"params": params["vae"]}, b_image, method=self.vae.encode).latent_dist.mean \
            # 
            #import pdb; pdb.set_trace()
            # jax.debug.breakpoint()

            #bias_idx = random.randrange(0, biassing_pixel_values.shape[0])                        
            biassing_vae_outs = vae.apply(
                {"params": vae_params}, batch["bias_pixel_values"], deterministic=True, method=vae.encode
            )
            biassing_latents = biassing_vae_outs.latent_dist.sample(sample_rng)
            biassing_latents = jnp.transpose(biassing_latents, (0, 3, 1, 2))
            biassing_latents = biassing_latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #VERIFY SHAPE STAYS THE SAME

            #Prev 4 lines but for biassing latents
            #latent_model_biases = []
            #for b_latents in biasing_latents:
            #  latent_model_biases.append(jnp.concatenate([b_latents] * 2))
            #latent_model_biases = [self.scheduler.scale_model_input(scheduler_state, bias, t) for bias in latent_model_biases]

            # Get the text embedding for conditioning
            if args.train_text_encoder:
                encoder_hidden_states = text_encoder_state.apply_fn(
                    batch["input_ids"], params=params["text_encoder"], dropout_rng=dropout_rng, train=True
                )[0]
            else:
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], params=text_encoder_state.params, train=False
                )[0]

            # Predict the noise residual
            unet_outputs = unet.apply(
                {"params": params["unet"]}, biassing_latents.astype(np.float16), timesteps, encoder_hidden_states,
                train=True
            )

            #import pdb; pdb.set_trace()

            biasList = []
            for bias in unet_outputs.biasList:
                if(type(bias) == tuple):
                    biasList.append((b.primal for b in bias))
                    #biasList.append((bias[0].primal, bias[1].primal))
                else:
                    biasList.append(bias.primal)
            #biasList = [bias.primal for bias in unet_outputs.biasList]
            #biassing_list = unet_return_dict['biasList']
                        # Predict the noise residual
            unet_outputs = unet.apply(
                {"params": params["unet"]}, noisy_latents.astype(np.float16), timesteps, encoder_hidden_states,
                biasList=biasList, #unet_outputs.biasList,
                train=True
            )
            #unet_return_dict = unet.apply({"params": params["unet"]}, noisy_latents.astype(np.float16), timesteps, encoder_hidden_states,biasList=biassing_list,train=True)
            #unet_outputs = unet_return_dict['sample']
            #unet_outputs = unet_return_dict[0]

            #breakpoint_if_nonfinite(unet_outputs)

            noise_pred = unet_outputs.sample

            # is_finite = jnp.isfinite(noise_pred).all()
            # if(not is_finite):
                
            #     import pdb; pdb.set_trace()

            #jax.debug.breakpoint()

            if args.with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                noise_pred, noise_pred_prior = jnp.split(noise_pred, 2, axis=0)
                noise, noise_prior = jnp.split(noise, 2, axis=0)

                # Compute instance loss
                loss = (noise - noise_pred) ** 2
                loss = loss.mean()

                # Compute prior loss
                prior_loss = (noise_prior - noise_pred_prior) ** 2
                prior_loss = prior_loss.mean()

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = (noise - noise_pred) ** 2
                loss = loss.mean()

            # jax.debug.print("LOSS 🤯 {x} 🤯 LOSS", x=loss)


            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params)
        if(not DEBUG):
            grad = jax.lax.pmean(grad, "batch")

        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        if args.train_text_encoder:
            new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
        else:
            new_text_encoder_state = text_encoder_state

        metrics = {"loss": loss}
        if(not DEBUG):
            metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1, 2))

    # Replicate the train state on each device
    if(not DEBUG):
        unet_state = jax_utils.replicate(unet_state)
        text_encoder_state = jax_utils.replicate(text_encoder_state)
        vae_params = jax_utils.replicate(vae_params)

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:

            #import pdb; pdb.set_trace()
            #jax.jit(train_step)(unet_state, text_encoder_state, vae_params, batch, train_rngs)

            if(not DEBUG):
                batch = shard(batch)
                unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
                    unet_state, text_encoder_state, vae_params, batch, train_rngs
                )
            else:
                unet_state, text_encoder_state, train_metric, train_rngs = train_step(
                    unet_state, text_encoder_state, vae_params, batch, train_rngs
                )

            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= args.max_train_steps:
                break
            
            
            if(global_step %100 == 1):
                losses = [jax_utils.unreplicate(metric)['loss'] for metric in train_metrics[-100:]]
                print(f"Loss: {np.mean(losses)}")
                
                print(jax_utils.unreplicate(unet_state.params['layer_biases']))
            #import pdb; pdb.set_trace()

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        scheduler = FlaxPNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        )
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        pipeline = FlaxInfusingStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        pipeline.save_pretrained(
            args.output_dir,
            params={
                "text_encoder": get_params_to_save(text_encoder_state.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(unet_state.params),
                "safety_checker": safety_checker.params,
            },
        )

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

def load_images(urls):
    images = list(Image.open(file_path).convert('RGB') for file_path in urls)
    bias_images = images[11:]

    size = 512
    image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            )

    return [image_transforms(i) for i in bias_images]


if __name__ == "__main__":
    #with jax.disable_jit():
    if(not DEBUG):
        main()
    else:
        with jax.disable_jit():
            main()
