export OUTPUT_DIR="/home/andrew/model"
export INSTANCE_DIR="/home/andrew/data/cowboy"
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"



python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400
