import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.functional import interpolate
from tqdm import tqdm

from src.data.landslide4sense import LandslideDataset
from src.models.unet_transformer import UNetTransformer
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import iou_score, dice_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="./data")
    p.add_argument("--extra_train_dir", type=str, default=None, help="GAN-augmented train dir (images/masks)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resize", type=int, nargs=2, default=[512,512])
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--transformer_blocks", type=int, default=1)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="./checkpoints/seg")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_real = LandslideDataset(os.path.join(args.data_root, "train"), augment=True, resize=tuple(args.resize))
    ds = ds_real
    if args.extra_train_dir:
        ds_gan = LandslideDataset(args.extra_train_dir, augment=True, resize=tuple(args.resize))
        ds = ConcatDataset([ds_real, ds_gan])

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_val = DataLoader(LandslideDataset(os.path.join(args.data_root, "val"), augment=False, resize=tuple(args.resize)),
                        batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNetTransformer(in_channels=args.in_channels, num_classes=1,
                            base_ch=args.base_ch, transformer_blocks=args.transformer_blocks,
                            transformer_heads=args.transformer_heads).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = BCEDiceLoss(0.5)

    best_iou = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl, desc=f"Train {epoch}/{args.epochs}")
        for batch in pbar:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            logits = model(img)
            if logits.shape[-2:] != msk.shape[-2:]:
                logits = interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, msk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # validation
        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for batch in dl_val:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                logits = model(img)
                if logits.shape[-2:] != msk.shape[-2:]:
                    logits = interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
                ious.append(iou_score(logits, msk))
                dices.append(dice_score(logits, msk))
        miou, mdice = sum(ious)/len(ious), sum(dices)/len(dices)
        print(f"[Val] IoU: {miou:.4f}  Dice: {mdice:.4f}")

        # checkpoint
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.out_dir, "latest.pt"))
        if miou > best_iou:
            best_iou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.out_dir, "best.pt"))

if __name__ == "__main__":
    main()
