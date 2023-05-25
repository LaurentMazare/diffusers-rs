import torch
from safetensors.torch import save_file
import urllib.request
import os
import sys
import argparse

data_path = os.path.join(os.path.dirname(__file__), "../data/")

def ensure_data_dir():
    print("Ensuring empty data directory...")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print("Created empty data directory")
        return
    else:
        print("Found data directory")

    if os.listdir(data_path):
        while (response := input("Data directory isn't empty. Would you like to overwrite its contents? [(y)/n]")) not in ["y","n",""]:
            pass

        if response == "n":
            print("Aborting.")
            sys.exit()
        elif response == "y" or response == "":
            print("Removing existing files...")
            for i in os.listdir(data_path):
                try:
                    os.remove(os.path.join(data_path, i))
                except:
                    pass
            return

    print("Data directory is empty!")

def get_safetensors(safetensors, weight_bits, use_cpu):
    for name, url in safetensors.items():
        print(f"Getting {name} {weight_bits} bit tensors...")

        # Download bin file
        urllib.request.urlretrieve(url, os.path.join(data_path, f"{name}.bin"))

        # Make safetensors file
        model = torch.load(os.path.join(data_path, f"{name}.bin"), map_location=torch.device("cpu") if use_cpu else None)
        tensors = {k: v.clone().detach() for k, v in model.items() if 'text_model' in k} if name in ["clip_v2.1", "pytorch_model"] else dict(model)
        save_file(tensors, os.path.join(data_path, f"{name}.safetensors"))

        # Remove bin file
        os.remove(os.path.join(data_path, f"{name}.bin"))

def get_vocab(vocab_url):
    print("Getting vocab...")
    urllib.request.urlretrieve(vocab_url, os.path.join(data_path, "bpe_simple_vocab_16e6.txt.gz"))
    import gzip
    with gzip.open(os.path.join(data_path, "bpe_simple_vocab_16e6.txt.gz"), 'rb') as g:
        with open(os.path.join(data_path, "bpe_simple_vocab_16e6.txt"), "xb") as f:
            f.write(g.read())
    os.remove(os.path.join(data_path, "bpe_simple_vocab_16e6.txt.gz"))

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
    print("Setting up model weights for diffusers-rs...")

    parser = argparse.ArgumentParser(description="Setting up model weights for diffusers-rs...")
    parser.add_argument("--sd_version", "-v", choices=["2.1", "1.5"], default="2.1")
    parser.add_argument("--weight_bits", "-w", choices=["16", "32"], default="16")
    parser.add_argument("--use_cpu", "-c", action="store_true", default=False)
    args = parser.parse_args()

    safetensors, vocab_url = get_urls(args.sd_version, args.weight_bits)
    ensure_data_dir()
    get_vocab(vocab_url)
    get_safetensors(safetensors, args.weight_bits, args.use_cpu)

    print("Finished!")
