import os, sys, argparse, glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import cv2
import numpy as np
from tqdm import tqdm
from src.models.gan import Generator

def read_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (m>127).astype(np.float32)

def read_image(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)/255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g., ./data/train")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--variants_per_image", type=int, default=2)
    ap.add_argument("--noise_channels", type=int, default=2)
    ap.add_argument("--in_channels", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(in_image_channels=args.in_channels, noise_channels=args.noise_channels).to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device))
    G.eval()

    img_dir = os.path.join(args.data_root, "images")
    msk_dir = os.path.join(args.data_root, "masks")
    out_img = os.path.join(args.out_dir, "images")
    out_msk = os.path.join(args.out_dir, "masks")
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_msk, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))

    for ip in tqdm(img_paths, desc="Augmenting"):
        name = os.path.basename(ip)
        mp = os.path.join(msk_dir, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(msk_dir, name)
            if not os.path.exists(mp): 
                continue

        img = read_image(ip)
        msk = read_mask(mp)
        h, w = msk.shape
        img = img[:,:,:3]
        img_t = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).float().to(device)
        msk_t = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).float().to(device)

        for k in range(args.variants_per_image):
            noise = torch.randn(1, args.noise_channels, h, w, device=device)
            with torch.no_grad():
                fake = G(img_t*0, msk_t, noise).clamp(-1,1)
            fake = (fake*0.5 + 0.5)
            fake_np = fake.squeeze(0).cpu().numpy().transpose(1,2,0)
            fake_np = (fake_np*255.0).astype(np.uint8)
            fake_np = cv2.cvtColor(fake_np, cv2.COLOR_RGB2BGR)
            out_name = os.path.splitext(name)[0] + f"_gan{k}.png"
            cv2.imwrite(os.path.join(out_img, out_name), fake_np)
            cv2.imwrite(os.path.join(out_msk, out_name), (msk*255).astype(np.uint8))

if __name__ == "__main__":
    main()
