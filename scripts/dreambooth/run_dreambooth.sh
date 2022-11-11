export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="/home/dhruvnaik/sd-comic-generation/data/dog/"
export OUTPUT_DIR="/home/dhruvnaik/sd-comic-generation/models/test_dreambooth_dog"

python3 train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400 \
  --mixed_precision="bf16"