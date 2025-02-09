
from torch import Tensor

import torch


class Diffusion:
    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps = 1500):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps

        """
        if
            betas = [0.1, 0.2, 0.3, ...]
        then
            alphas = [0.9, 0.8, 0.7, ...]
            alphas_cumprod = [0.9, 0.9 * 0.8, 0.9 * 0.8, * 0.7, ...]


        """
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, images: Tensor, time_steps: Tensor, device: torch.device):
        """
        x_0: (B, C, H, W)
        t: (B,)
        """
        noise = torch.randn_like(images)
        # print(f"Noise = {noise.shape}")
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), time_steps, images.shape)
        # print(f"TSteps Sqrt of CumProd of Alpha = {sqrt_alphas_cumprod_t.shape}")
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), time_steps, images.shape)
        # print(f"TSteps Sqrt of NegCumProd of Alpha = {sqrt_alphas_cumprod_t.shape}")

        mean = sqrt_alphas_cumprod_t.to(device) * images.to(device)
        # print(f"Mean = {mean.shape}")
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        # print(f"Variance = {variance.shape}")
        noisy_images = mean + variance
        # print(f"Noise Imeages = {noisy_images.shape}")
        # noisy_x = (noisy_x - noisy_x.min()) / (noisy_x.max() - noisy_x.min())

        return noisy_images, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values from vals
        according to the indices stored in `t`
        """
        result = values.gather(-1, t.cpu())
        """
        if
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3

        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)

        """
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)