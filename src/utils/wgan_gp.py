import torch, torch.autograd as autograd

def gradient_penalty(discriminator, real_imgs, fake_imgs, masks, device, lambda_gp=10.0):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates, masks)
    d_interpolates = d_interpolates.mean()
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp
