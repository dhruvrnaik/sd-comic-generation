export OUTPUT_DIR="/home/andrew/model/infusion"
export INSTANCE_DIR="/mnt/disks/persist"
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
#"/home/andrew/model/dreambooth"


python3 train_infusion_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a picture of krosh" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=40000 \
  --mixed_precision="bf16"
    
# Some change
#alias python=python3

# For running inference
# Need to specify model dir & image path
#python inference_flax.py  

#For running training
