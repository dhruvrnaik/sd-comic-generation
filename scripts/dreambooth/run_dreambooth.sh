export OUTPUT_DIR="/home/andrew/model/dreambooth"
export INSTANCE_DIR="/home/andrew/data/krosh"
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"



python3 train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a krosh" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=4000 \
  --mixed_precision="bf16"
    
# Some change