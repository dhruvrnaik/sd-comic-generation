export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export DATA_DIR="/home/dhruvnaik/sample_dataset"

python3 textual_inversion_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --output_dir="textual_inversion_cat_test"