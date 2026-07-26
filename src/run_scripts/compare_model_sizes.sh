#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=0:30:00
#SBATCH --output=logs/compare_model_sizes.out.txt

# 1. Path Setup
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs

# 2. Activate the venv
source $GPU_ENV/bin/activate
pip install -e .

# 3. Environment variables
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 4. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

cd $PROJECT_DIR

echo "Comparing model sizes and memory usage..."
echo "==========================================="

python << 'EOF'
import torch
from transformers import AutoModel, AutoConfig
import os

os.environ['HF_HOME'] = '/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache'

print("\nLoading DINOv2-base...")
dinov2_config = AutoConfig.from_pretrained('facebook/dinov2-base', output_hidden_states=True, return_dict=True)
dinov2 = AutoModel.from_pretrained('facebook/dinov2-base', config=dinov2_config)

print("Loading vit-large...")
vitlarge_config = AutoConfig.from_pretrained('google/vit-large-patch16-224', output_hidden_states=True, return_dict=True)
vitlarge = AutoModel.from_pretrained('google/vit-large-patch16-224', config=vitlarge_config)

print("Loading rad-dino...")
rad_dino_config = AutoConfig.from_pretrained('microsoft/rad-dino', output_hidden_states=True, return_dict=True)
rad_dino = AutoModel.from_pretrained('microsoft/rad-dino', config=rad_dino_config)

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Parameters
dinov2_params = sum(p.numel() for p in dinov2.parameters())
vitlarge_params = sum(p.numel() for p in vitlarge.parameters())
rad_dino_params = sum(p.numel() for p in rad_dino.parameters())

print(f"\nParameters:")
print(f"  DINOv2-base:  {dinov2_params:,}")
print(f"  ViT-Large:    {vitlarge_params:,}")
print(f"  rad-dino:     {rad_dino_params:,}")

# Model size (FP32)
dinov2_size_mb = sum(p.numel() * 4 / 1e6 for p in dinov2.parameters())
vitlarge_size_mb = sum(p.numel() * 4 / 1e6 for p in vitlarge.parameters())
rad_dino_size_mb = sum(p.numel() * 4 / 1e6 for p in rad_dino.parameters())

print(f"\nModel Size (FP32):")
print(f"  DINOv2-base:  {dinov2_size_mb:.1f} MB")
print(f"  ViT-Large:    {vitlarge_size_mb:.1f} MB")
print(f"  rad-dino:     {rad_dino_size_mb:.1f} MB")

# Config comparison
print(f"\nConfig:")
print(f"  DINOv2-base hidden size: {dinov2_config.hidden_size}, layers: {dinov2_config.num_hidden_layers}")
print(f"  ViT-Large hidden size:   {vitlarge_config.hidden_size}, layers: {vitlarge_config.num_hidden_layers}")
print(f"  rad-dino hidden size:    {rad_dino_config.hidden_size}, layers: {rad_dino_config.num_hidden_layers}")

# GPU memory usage (move to GPU and check)
print(f"\nGPU Memory Usage (loading model):")
torch.cuda.reset_peak_memory_stats()
dinov2_gpu = dinov2.cuda()
dinov2_load_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"  DINOv2-base: {dinov2_load_gb:.2f} GB")

torch.cuda.reset_peak_memory_stats()
vitlarge_gpu = vitlarge.cuda()
vitlarge_load_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"  ViT-Large:   {vitlarge_load_gb:.2f} GB")

torch.cuda.reset_peak_memory_stats()
rad_dino_gpu = rad_dino.cuda()
rad_dino_load_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"  rad-dino:    {rad_dino_load_gb:.2f} GB")

print("\n" + "="*60)
print("Testing forward pass (batch of 8, 224x224 images with hidden states)...")
print("="*60)

batch_size = 8
img_size = (batch_size, 3, 224, 224)
dummy_input = torch.randn(img_size, device='cuda')

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = dinov2_gpu(dummy_input, output_hidden_states=True)
dinov2_forward_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"DINOv2-base: {dinov2_forward_gb:.2f} GB")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = vitlarge_gpu(dummy_input, output_hidden_states=True)
vitlarge_forward_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"ViT-Large:   {vitlarge_forward_gb:.2f} GB")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = rad_dino_gpu(dummy_input, output_hidden_states=True)
rad_dino_forward_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"rad-dino:    {rad_dino_forward_gb:.2f} GB")

print("\n" + "="*60)
EOF

echo "Done."