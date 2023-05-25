#!/bin/bash
set -euxo pipefail

ROOT=$(pwd)

# This can be either fp16 or main for float32 weights.
BRANCH=fp16

wget_vocab() {
   wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
   gunzip bpe_simple_vocab_16e6.txt.gz
}

wget_weights() {
  header="Authorization: Bearer $1"
  # download weights for clip
  wget --header="$header" https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/$BRANCH/text_encoder/pytorch_model.bin -O clip.bin
  # download weights for vae
  wget --header="$header" https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/$BRANCH/vae/diffusion_pytorch_model.bin -O vae.bin
  # download weights for unet
  wget --header="$header" https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/$BRANCH/unet/diffusion_pytorch_model.bin -O unet.bin	
  
  # convert to npz
  LD_LIBRARY_PATH= python3 -c "
import torch
from safetensors.torch import save_file
  
model = torch.load('./clip.bin')
save_file({k: v for k, v in model.items() if 'text_model' in k}, './clip_v2.1.safetensors')
  
model = torch.load('./vae.bin')
save_file(dict(model), './vae_v2.1.safetensors')
  
model = torch.load('./unet.bin')
save_file(dict(model), './unet_v2.1.safetensors')
"
}

if [ $# -ne 1 ]; then
    echo 'Usage: ./download_weights_2.1.sh <HUGGINGFACE_TOKEN>' >&2
    exit 1
fi

echo "Setting up for diffusers-rs..."

mkdir -p data
cd data

echo "Getting the Weights and the Vocab File"
# get the weights
wget_vocab 
wget_weights $1 

echo "Cleaning ..."
rm -rf $ROOT/data/*.bin

echo "Done."
