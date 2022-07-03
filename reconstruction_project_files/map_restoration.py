import torch
import torch.nn as nn
import torch.optim as optim
from utils import total_variation


def map_restoration_vae_type1(inpimg, decmu, vae_prior, device='cuda', riter=100, tvweight=1, alpha=0.01):
    """
    :param inpimg: [batch, channels, width, height] Input image (image with lesion)
    :param decmu:  [batch, channels, width, height] Decoded mean for l2 loss calculation
    :param vae_prior: trained VAE prior
    :param device: computation device
    :param riter: reconstruction iters
    :param tvweight: tv_norm weight
    :param alpha: learning rate
    :return: Restored image
    """
    inpimg = nn.Parameter(inpimg, requires_grad=False)
    decmu = nn.Parameter(decmu.float(), requires_grad=False)
    X_i = nn.Parameter(inpimg.clone(), requires_grad=True)

    inpimg = inpimg.to(device)
    decmu = decmu.to(device)
    X_i = X_i.to(device)
    vae_prior = vae_prior.to(device)

    optimizer = optim.Adam([X_i], lr=alpha)
    for i in range(riter):
        _, z_mean, z_cov, _ = vae_prior(X_i.double())
        l2_loss = (decmu.view(-1, decmu.numel()) - X_i.view(-1, X_i.numel())).pow(2)
        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        gradfn = torch.sum(l2_loss) + kl_loss + tvweight * total_variation((X_i - inpimg).squeeze(1))
        gradfn.backward()
        torch.clamp(X_i, -100, 100)
        optimizer.step()
        optimizer.zero_grad()
    return X_i


def map_restoration_vae_type2(Y, decoded_mu, vae_model, recon_iters, tv_weight, step_size, device='cuda'):
    img_ano = nn.Parameter(Y.clone().to(device), requires_grad=True)
    Y = Y.to(device)
    decoded_mu = decoded_mu.to(device)
    for i in range(recon_iters):
        _, z_mean, z_cov, _ = vae_model(img_ano.double())
        kl_loss = -0.5*torch.sum(1+z_cov-z_mean.pow(2)-z_cov.exp())
        l2_loss = (decoded_mu.view(-1, decoded_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2)
        elbo = l2_loss + kl_loss
        tv_loss = total_variation((img_ano - Y).squeeze(1))
        loss = (-1)*tv_weight*tv_loss + elbo
        loss_grad = torch.autograd.grad(loss, img_ano, grad_outputs=loss.data.new(loss.shape).fill_(1), create_graph=True)
        loss_grad = torch.clip(loss_grad, -100, 100)
        img_ano_update = img_ano + step_size*loss_grad.detach()
        img_ano = img_ano_update.to(device)
        img_ano.requires_grad = True
    return img_ano


