#!/bin/bash
set -euxo pipefail

ROOT=$(pwd)

# This can be either fp16 or main for float32 weights.
BRANCH=fp16

wget_vocab() {
   wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
   gunzip bpe_simple_vocab_16e6.txt.gz
}

wget_clip_weights() {
  wget -c https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
  LD_LIBRARY_PATH= python3 -c "
import torch
from safetensors.torch import save_file

model = torch.load('./pytorch_model.bin')
tensors = {k: v.clone().detach() for k, v in model.items() if 'text_model' in k}
save_file(tensors, 'pytorch_model.safetensors')
"
}

wget_vae_unet_weights() {
  # download weights for vae
  header="Authorization: Bearer $1"
  wget --header="$header" https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/$BRANCH/vae/diffusion_pytorch_model.bin -O vae.bin
  # download weights for unet
  wget --header="$header" https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/$BRANCH/unet/diffusion_pytorch_model.bin -O unet.bin	
  
  # convert to npz
  LD_LIBRARY_PATH= python3 -c "
import torch
from safetensors.torch import save_file
  
model = torch.load('./vae.bin')
save_file(dict(model), './vae.safetensors')
  
model = torch.load('./unet.bin')
save_file(dict(model), './unet.safetensors')
"
}

if [ $# -ne 1 ]; then
    echo 'Usage: ./download_weights.sh <HUGGINGFACE_TOKEN>' >&2
    exit 1
fi

echo "Setting up for diffusers-rs..."

mkdir -p data
cd data

echo "Getting the Weights and the Vocab File"
# get the weights
wget_vocab 
wget_clip_weights 
wget_vae_unet_weights $1 

echo "Cleaning ..."
rm -rf $ROOT/data/*.bin

echo "Done."
