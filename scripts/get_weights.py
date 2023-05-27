import torch
from safetensors.torch import save_file
import urllib.request
import os
import sys
import argparse

data_path = os.path.join(os.path.dirname(__file__), "../data/")
vocab_filename = "bpe_simple_vocab_16e6.txt"

def ensure_data_dir(safetensors):
    print("Ensuring empty data directory...")

    if os.path.exists(data_path):
        # Fail if conflicting files exist
        files = os.listdir(data_path)
        newfiles = [x for name in safetensors for x in (f"{name}.bin", f"{name}.safetensors")]
        newfiles += [vocab_filename, f"{vocab_filename}.gz"]
        conflicts = set(files) & set(newfiles)
        if len(conflicts) != 0:
            print("Error: please remove the following files from data directory:")
            print(conflicts)
            sys.exit("Found conflicting files in data directory.")
    else:
        os.mkdir(data_path)

    print("Found no conflicts!")

def get_safetensors(safetensors, weight_bits):
    for name, url in safetensors.items():
        print(f"Getting {name} {weight_bits} bit tensors...")

        # Download bin file
        urllib.request.urlretrieve(url, os.path.join(data_path, f"{name}.bin"))

        # Make safetensors file
        model = torch.load(os.path.join(data_path, f"{name}.bin"), map_location=torch.device("cpu"))
        tensors = {k: v.clone().detach() for k, v in model.items() if 'text_model' in k} if name in ["clip_v2.1", "pytorch_model"] else dict(model)
        save_file(tensors, os.path.join(data_path, f"{name}.safetensors"))

        # Remove bin file
        os.remove(os.path.join(data_path, f"{name}.bin"))

def get_vocab(vocab_url):
    print("Getting vocab...")
    urllib.request.urlretrieve(vocab_url, os.path.join(data_path, f"{vocab_filename}.gz"))
    import gzip
    with gzip.open(os.path.join(data_path, f"{vocab_filename}.gz"), 'rb') as g:
        with open(os.path.join(data_path, vocab_filename), "xb") as f:
            f.write(g.read())
    os.remove(os.path.join(data_path, f"{vocab_filename}.gz"))

def get_urls(sd_version, weight_bits):
    branch = "main"
    if weight_bits == "16":
        branch = "fp16" # fp16 for float16 weights or main for float32 weights

    safetensors_v1_5 = {
        "vae": f"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/{branch}/vae/diffusion_pytorch_model.bin",
        "unet": f"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/{branch}/unet/diffusion_pytorch_model.bin",
        "pytorch_model": f"https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"
    }
    safetensors_v2_1 = {
        "vae_v2.1": f"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/{branch}/vae/diffusion_pytorch_model.bin",
        "unet_v2.1": f"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/{branch}/unet/diffusion_pytorch_model.bin",
        "clip_v2.1": f"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/{branch}/text_encoder/pytorch_model.bin"
    }
    vocab_url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"

    return safetensors_v1_5 if sd_version == "1.5" else safetensors_v2_1, vocab_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download weights for diffusers-rs.")
    parser.add_argument("--sd_version", "-v", choices=["2.1", "1.5"], default="2.1")
    parser.add_argument("--weight_bits", "-w", choices=["16", "32"], default="16")
    args = parser.parse_args()

    print("Setting up model weights for diffusers-rs...")

    safetensors, vocab_url = get_urls(args.sd_version, args.weight_bits)
    ensure_data_dir(safetensors)
    get_vocab(vocab_url)
    get_safetensors(safetensors, args.weight_bits)

    print("Finished!")
