import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.functional import interpolate
from tqdm import tqdm
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.landslide4sense import LandslideDataset
from src.models.HSC_HENet import HSC_HENet, get_hsc_henet_loss_function
from src.utils.metrics import iou_score, dice_score

def main():
    p = argparse.ArgumentParser(description="Train HSC_HENet for Landslide Detection")
    p.add_argument("--data_root", type=str, required=True, help="Path to data root containing images/ and annotations/")
    p.add_argument("--extra_train_dir", type=str, default=None, help="Additional data root for training (e.g. GAN augmented)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--resize", type=int, nargs=2, default=[256, 256])
    p.add_argument("--in_channels", type=int, default=14, help="Number of input channels (14 for Landslide4Sense)")
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--deep_supervision", action="store_true", help="Enable deep supervision")
    p.add_argument("--out_dir", type=str, default="./checkpoints/hsc_henet")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define transforms
    # LandslideDataset applies transforms at the end. 
    # We use nearest neighbor for masks to preserve class values (0/1).
    transform = transforms.Compose([
        transforms.Resize(tuple(args.resize))
    ])
    target_transform = transforms.Compose([
        transforms.Resize(tuple(args.resize), interpolation=transforms.InterpolationMode.NEAREST)
    ])

    # Load Datasets
    # Note: LandslideDataset expects data_root to contain "images/{split}" and "annotations/{split}"
    print(f"Loading training data from {args.data_root}...")
    ds_real = LandslideDataset(
        data_root=args.data_root, 
        split="train", 
        transform=transform, 
        target_transform=target_transform
    )
    ds = ds_real
    
    if args.extra_train_dir:
        print(f"Loading extra training data from {args.extra_train_dir}...")
        ds_gan = LandslideDataset(
            data_root=args.extra_train_dir, 
            split="train", 
            transform=transform, 
            target_transform=target_transform
        )
        ds = ConcatDataset([ds_real, ds_gan])

    print(f"Total training samples: {len(ds)}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    print(f"Loading validation data from {args.data_root}...")
    ds_val = LandslideDataset(
        data_root=args.data_root, 
        split="validation", # Assuming 'validation' folder exists, or use 'test' if that's the convention
        transform=transform,
        target_transform=target_transform
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Total validation samples: {len(ds_val)}")

    # Initialize Model
    model = HSC_HENet(
        n_channels=args.in_channels,
        n_classes=1,
        base_channels=args.base_ch,
        deep_supervision=args.deep_supervision
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler (Optional, but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss Function
    loss_fn = get_hsc_henet_loss_function(deep_supervision=args.deep_supervision)

    best_iou = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl, desc=f"Train {epoch}/{args.epochs}")
        epoch_loss = 0.0
        
        for batch in pbar:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            
            if args.deep_supervision:
                main_out, aux_outs = model(img, return_aux=True)
                loss = loss_fn((main_out, aux_outs), msk)
                logits = main_out # For metrics/logging if needed
            else:
                logits = model(img)
                # Ensure output size matches mask (though Resize transform should handle it)
                if logits.shape[-2:] != msk.shape[-2:]:
                     logits = interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(logits, msk)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for batch in dl_val:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                
                # Inference always without deep supervision return
                logits = model(img, return_aux=False)
                
                if logits.shape[-2:] != msk.shape[-2:]:
                    logits = interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
                
                ious.append(iou_score(logits, msk))
                dices.append(dice_score(logits, msk))
        
        miou = sum(ious)/len(ious) if ious else 0
        mdice = sum(dices)/len(dices) if dices else 0
        
        print(f"[Val] Epoch {epoch} - IoU: {miou:.4f}  Dice: {mdice:.4f}")

        # Checkpoint
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args
        }
        torch.save(save_dict, os.path.join(args.out_dir, "latest.pt"))
        
        if miou > best_iou:
            best_iou = miou
            torch.save(save_dict, os.path.join(args.out_dir, "best.pt"))
            print(f"New best IoU: {best_iou:.4f} saved.")

if __name__ == "__main__":
    main()
