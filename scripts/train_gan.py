import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from src.data.landslide4sense import LandslideDataset
from src.models.gan import Generator, PatchDiscriminator
from src.utils.wgan_gp import gradient_penalty


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def visualize_generation(G, real_image, mask, device, output_path, use_zero_image=True, noise_channels=2):
    """
    可视化生成结果并保存
    Args:
        G: 生成器模型
        real_image: 真实图像 (B, C, H, W)
        mask: 掩码 (B, 1, H, W)
        device: 设备
        output_path: 保存路径
        use_zero_image: 是否使用零图像作为输入
        noise_channels: 噪声通道数
    """
    # 选择第一个样本进行可视化
    real = real_image[0:1].to(device)  # 取第一个样本，保持batch维度
    mask_sample = mask[0:1].to(device)
    
    # 生成噪声
    b, _, h, w = real.shape
    noise = torch.randn(b, noise_channels, h, w, device=device)
    
    # 生成伪造图像
    with torch.no_grad():
        # 根据参数选择输入图像
        image_input = real * 0 if use_zero_image else real
        fake = G(image_input, mask_sample, noise)
    
    # 转换为numpy数组并处理
    real_np = real.squeeze().cpu().numpy()  # (C, H, W)
    mask_np = mask_sample.squeeze().cpu().numpy()  # (H, W)
    fake_np = fake.squeeze().cpu().numpy()  # (C, H, W)
    
    # 创建可视化目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存RGB合成图 (选择合适的3个波段)
    # Landslide4Sense数据通常使用波段组合：红(3), 绿(2), 蓝(1) 或 假彩色合成
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 真实图像RGB合成
    real_rgb = np.stack([real_np[3], real_np[2], real_np[1]], axis=-1)  # BGR to RGB
    real_rgb = (real_rgb - real_rgb.min()) / (real_rgb.max() - real_rgb.min())  # 归一化到[0,1]
    axes[0].imshow(real_rgb)
    axes[0].set_title('Real Image (RGB Composite)')
    axes[0].axis('off')
    
    # 掩码图像
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    # 伪造图像RGB合成
    fake_rgb = np.stack([fake_np[3], fake_np[2], fake_np[1]], axis=-1)
    fake_rgb = (fake_rgb - fake_rgb.min()) / (fake_rgb.max() - fake_rgb.min())
    axes[2].imshow(fake_rgb)
    axes[2].set_title('Fake Image (RGB Composite)')
    axes[2].axis('off')
    
    # 真实图像与伪造图像的差异
    diff = np.abs(real_rgb - fake_rgb)
    axes[3].imshow(diff)
    axes[3].set_title('Difference')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存单个波段图像（前6个波段）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        ax = axes[i//3, i%3]
        band_data = fake_np[i]
        ax.imshow(band_data, cmap='viridis')
        ax.set_title(f'Band {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_bands.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default='C:\Users\hongbo_0\Desktop\U_net_new\Landslide4Sense-2022-main\landslide4sense', help="root with train/images & train/masks")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--in_channels", type=int, default=14, help="输入图像通道数")
    p.add_argument("--noise_channels", type=int, default=2)
    p.add_argument("--resize", type=int, nargs=2, default=[256,256])
    p.add_argument("--n_critic", type=int, default=5, help="D steps per G step")
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--out_dir", type=str, default="./checkpoints/gan")
    p.add_argument("--viz_dir", type=str, default="./visualizations/gan", help="可视化结果保存目录")
    p.add_argument("--use_zero_image", action="store_true", default=True, help="使用零图像作为生成器的图像输入")
    p.add_argument("--viz_freq", type=int, default=5, help="每N轮保存一次可视化结果")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"数据根目录: {args.data_root}")
    print(f"使用零图像输入: {args.use_zero_image}")
    print(f"可视化频率: 每{args.viz_freq}轮")
    print(f"可视化保存目录: {args.viz_dir}")
    
    # 打印数据集信息
    ds = LandslideDataset(data_root=args.data_root, 
                          split="train")
    print(f"数据集大小: {len(ds)}")
    # 获取一个样本查看形状
    if len(ds) > 0:
        sample = ds[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"掩码形状: {sample['mask'].shape}")
    
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"数据加载器批次大小: {args.batch_size}")
    
    # 打印模型配置
    print(f"生成器输入通道: {args.in_channels}, 噪声通道: {args.noise_channels}")
    G = Generator(in_image_channels=args.in_channels, noise_channels=args.noise_channels).to(device)
    D = PatchDiscriminator(in_image_channels=args.in_channels).to(device)
    print("模型初始化完成")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    
    # 保存初始状态的可视化
    if len(ds) > 0:
        visualize_generation(G, real_image=sample['image'].unsqueeze(0), 
                            mask=sample['mask'].unsqueeze(0), 
                            device=device, 
                            output_path=os.path.join(args.viz_dir, "init_generation.png"),
                            use_zero_image=args.use_zero_image,
                            noise_channels=args.noise_channels)
    
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            real = batch["image"].to(device)
            mask = batch["mask"].to(device)
            b, c, h, w = real.shape

            # Train D
            for _ in range(args.n_critic):
                noise = torch.randn(b, args.noise_channels, h, w, device=device)
                
                # 根据参数选择输入图像
                image_input = real * 0 if args.use_zero_image else real
                
                with torch.no_grad():
                    fake = G(image_input, mask, noise)
                D_real = D(real, mask).mean()
                D_fake = D(fake, mask).mean()
                gp = gradient_penalty(D, real, fake, mask, device, lambda_gp=args.lambda_gp)
                loss_d = (D_fake - D_real) + gp

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # Train G
            noise = torch.randn(b, args.noise_channels, h, w, device=device)
            
            # 根据参数选择输入图像
            image_input = real * 0 if args.use_zero_image else real
            
            fake = G(image_input, mask, noise)
            D_fake = D(fake, mask).mean()
            loss_g = -D_fake

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            pbar.set_postfix({"loss_d": f"{loss_d.item():.3f}", "loss_g": f"{loss_g.item():.3f}"})
        
        # 定期保存可视化结果
        if epoch % args.viz_freq == 0:
            visualize_generation(G, real_image=real, 
                                mask=mask, 
                                device=device, 
                                output_path=os.path.join(args.viz_dir, f"epoch_{epoch}_generation.png"),
                                use_zero_image=args.use_zero_image,
                                noise_channels=args.noise_channels)
        
        save_ckpt(G, os.path.join(args.out_dir, "latest.pt"))
        save_ckpt(D, os.path.join(args.out_dir, "discriminator_latest.pt"))

if __name__ == "__main__":
    main()