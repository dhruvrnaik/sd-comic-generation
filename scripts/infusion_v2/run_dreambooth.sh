export OUTPUT_DIR="/home/andrew/model/infusion_coco"
export INSTANCE_DIR="/mnt/disks/persist"
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
#"/home/andrew/model/dreambooth"

#For trainingon dreambooth 
# NOTE: You need to modify to use Flintstoens dataset
# python3 train_infusion_flax.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a picture of krosh" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --learning_rate=5e-6 \
#   --max_train_steps=40000 \
#   --mixed_precision="bf16"

#For trainingon dreambooth 
# NOTE: You need to modify to use Flintstoens dataset
# python3 train_infusion_flax.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --output_dir=$OUTPUT_DIR \
#   --train_batch_size=1 \
#   --max_train_steps=4000 \
#   --resolution=512 \
#   --num_train_epochs=4 \
#   --mixed_precision="bf16"

#For trainingon infusion first extracting features then adding them 
python3 learn_weights_light.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=4000 \
  --resolution=512 \
  --num_train_epochs=4 \
  --mixed_precision="bf16"

#Learned weights:
#LR 5e-3: 
#    weights = [-0.0106812, -0.0546875, -0.480469, 0.0947266, -0.00107574, 0.160156, \
#         0.243164, -0.00665283, 0.0388184, -0.135742, -0.53125, -0.0688477, -0.118652]

#LR 1e-4: 
#    weights = [-0.03125, -0.0230713, -0.00628662, -0.0106812, 0.015625, 0.0285645, 0.03125, \
#               -0.0169678, 0.0177002, -0.03125, -0.03125, -0.0211182, 0.0159912]


#For trainingon dreambooth 
# NOTE: You need to modify to use Flintstoens dataset

# python3 learn_weights_light.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --output_dir=$OUTPUT_DIR \
#   --max_train_steps=4000 \
#   --num_train_epochs=4 \
#   --mixed_precision="bf16"

    
# Some change
#alias python=python3

# For running inference
# Need to specify model dir & image path
#python inference_flax.py  

#For running training
