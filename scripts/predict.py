import os, sys, argparse, glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import cv2
import numpy as np
from src.models.unet_transformer import UNetTransformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="segmentation checkpoint (.pt)")
    p.add_argument("--in_dir", type=str, required=True, help="folder with images")
    p.add_argument("--out_dir", type=str, required=True, help="output folder for predicted masks")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--resize", type=int, nargs=2, default=[512,512])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetTransformer(in_channels=args.in_channels, num_classes=1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for ip in sorted(glob.glob(os.path.join(args.in_dir, "*"))):
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None: continue
        h0, w0 = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_res = cv2.resize(img_rgb, tuple(args.resize[::-1]), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(np.transpose(img_res, (2,0,1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logit = model(t)
            pred = torch.sigmoid(logit)
            pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
        pred = (pred>0.5).astype(np.uint8)*255
        out_path = os.path.join(args.out_dir, os.path.basename(ip))
        cv2.imwrite(out_path, pred)

if __name__ == "__main__":
    main()
