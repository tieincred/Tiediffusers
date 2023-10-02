# Import necessary libraries
from accelerate.utils import write_basic_config
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from slack_notify import send_slack_notification
import torch
import os
import sys
os.environ['TRANSFORMERS_CACHE'] = './'

# Set up the environment
write_basic_config()

training_type = "text_and_unet"
os.environ['MODEL_NAME'] = "runwayml/stable-diffusion-v1-5"
os.environ['INSTANCE_DIR'] = "segioperez"
os.environ['OUTPUT_DIR'] = f"{os.environ['INSTANCE_DIR']}_{training_type}"
os.environ['CLASS_DIR'] = "./man"
os.environ['HF_HOME'] = '/media/pixis-ubuntu-20/pixis/tausif_workspace/laion'
os.environ['INSTANCE_PROMPT'] = "a photo of sksjeff man"
os.environ['CLASS_PROMPT'] = "a photo of man"
os.environ['VALIDATION_PROMPT'] = "a photo of sksjeff man in a suit sitting on chair, upper body, 4K, HD"

learning_rate = 1e-6
num_photos = len(os.listdir(os.environ['INSTANCE_DIR']))
step_guess = (num_photos * 300) / (learning_rate * 1e6)
print(step_guess)

# sys.exit()

# Define the directories
output_dir = os.environ.get('OUTPUT_DIR', 'JL_basic')  # Default to 'JL_basic' if 'OUTPUT_DIR' is not set in the environment
class_dir = os.environ.get('CLASS_DIR', './woman')  # Default to './woman' if 'CLASS_DIR' is not set in the environment

# Check and create OUTPUT_DIR if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Check and create CLASS_DIR if it doesn't exist
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

if training_type == 'basic':

  # Command to launch the training script
  os.system(f"""
  accelerate launch examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps={str(int(step_guess))}
  """)

  print(f"Weights are saved in: {os.environ['OUTPUT_DIR']}")

elif training_type == 'prior_preserving':

  # Command to launch the training script
  os.system("""
  accelerate launch examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1200 \
  """)

elif training_type == 'text_and_unet':
  
  # Command to launch the training script
  os.system("""
  accelerate launch examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
 		--train_text_encoder \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_checkpointing \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=1500 \
    --max_train_steps=4000 \
    --validation_prompt="$VALIDATION_PROMPT" \
    --num_validation_images=4 \
    --checkpointing_steps=250 \
    --validation_steps=250 \
    --use_8bit_adam
  """)

send_slack_notification()