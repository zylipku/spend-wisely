import copy
from typing import Callable, Any
import math

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import lightning as L

from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from torch.distributed import barrier

from ..frozen_strategy import FrozenStrategy


class IterativeImaging(L.LightningModule):

    def __init__(self,
                 pretrained_model: DDPMPipeline,
                 optimizer: torch.optim.Optimizer,
                 frozen_strategy: FrozenStrategy,
                 train_type: str,
                 experiment_setting,
                 buffer_size: int = 160,
                 compile: bool = True) -> None:
        super().__init__()

        self.training_model = pretrained_model.unet
        self.scheduler = pretrained_model.scheduler

        self.sampling_model = self.freeze_denoising()

        # self.print(frozen_strategy)
        # self.print(experiment_setting)
        self.save_hyperparameters(logger=False)
        # self.print(self.hparams)

        self.Nk_func = frozen_strategy
        self.k_step = 0
        self.frozen_at_nsteps = self.Nk_func(self.k_step)
        self.current_acc_nsteps = 0

        self.noise_idx = -math.ceil(len(self.scheduler.timesteps) * experiment_setting.noise_ratio)

        self.train_type = train_type

        self.sample_buffer = []
        self.sample_buffer_size = 0
        self.max_buffer_size = buffer_size

        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

        self.psnr_min, self.psnr_max = experiment_setting.psnr_range

        self.training_cost = 0
        self.sampling_cost = 0

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.training_model = torch.compile(self.training_model)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.training_model.parameters())
        return optimizer
        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}

    def psnr2prob(self, psnr: torch.Tensor) -> torch.Tensor:
        return torch.clip((psnr - self.psnr_min) / (self.psnr_max - self.psnr_min), min=0., max=1.)

    def filter_sample(self,
                      samples: torch.Tensor,
                      samples_gt: torch.Tensor) -> torch.Tensor:
        psnrs = [
            self.psnr(sample, sample_gt).item()
            for sample, sample_gt in zip(samples, samples_gt)
        ]
        weights = self.psnr2prob(torch.tensor(psnrs))
        sample_ftd = []
        for weight, sample in zip(weights, samples):
            if torch.rand(1) < weight:
                sample_ftd.append(sample)
        if len(sample_ftd) == 0:
            return None
        return torch.stack(sample_ftd)

    def denoising(self,
                  module: nn.Module,
                  imgs_noisy: torch.Tensor,
                  noise_idx: int) -> tuple[torch.Tensor, int]:

        num_inference_steps = 0

        x = imgs_noisy.clone()
        for t in self.scheduler.timesteps[noise_idx:]:
            # Prepare model input
            model_input = self.scheduler.scale_model_input(x, t)
            noise_pred = module(model_input, t)['sample']
            num_inference_steps += 1
            # Calculate what the updated sample should look like with the scheduler
            scheduler_output = self.scheduler.step(noise_pred, t, x)
            # Update x
            x = scheduler_output.prev_sample
        imgs_recov = x.clone()

        return imgs_recov, num_inference_steps

    def freeze_denoising(self) -> nn.Module:
        module = copy.deepcopy(self.training_model)
        module = module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def on_train_start(self):
        self.training_inference_nsteps = 0

    def training_step(self, batch, batch_idx):
        images, labels = batch
        imgs_gt: torch.Tensor = images

        if self.train_type == 'finetune-gt':
            imgs_ftred = imgs_gt.clone()
        else:
            noise = torch.randn_like(imgs_gt)
            imgs_noisy = self.scheduler.add_noise(imgs_gt, noise, timesteps=self.scheduler.timesteps[self.noise_idx])

            # sample from the model
            if self.current_acc_nsteps == self.frozen_at_nsteps:
                self.sampling_model = self.freeze_denoising()
                self.print(f'Frozen at Nk = N{self.k_step} = {self.frozen_at_nsteps} steps.')
                self.k_step += 1
                self.frozen_at_nsteps = self.Nk_func(self.k_step)
                self.current_acc_nsteps = 0

            with torch.no_grad():
                imgs_recov, steps = self.denoising(self.sampling_model, imgs_noisy, self.noise_idx)
                self.sampling_cost += imgs_recov.shape[0]
                self.training_inference_nsteps += steps * imgs_gt.shape[0]
                # self.log('train/recovered_psnr', self.psnr(imgs_recov, imgs_gt), sync_dist=True)
            if self.train_type == 'finetune-filtered':
                imgs_ftred = self.filter_sample(imgs_recov, imgs_gt)
            elif self.train_type == 'finetune-no-filtered':
                imgs_ftred = imgs_recov
            else:
                raise ValueError

        if imgs_ftred is not None:
            self.sample_buffer.append(imgs_ftred)
            self.sample_buffer_size += imgs_ftred.shape[0]

        self.log('train/sample_buffer_size', self.sample_buffer_size)

        if self.sample_buffer_size < self.max_buffer_size:
            return torch.tensor(0.0, requires_grad=True)

        # reset sample buffer
        imgs_ftred = torch.cat(self.sample_buffer, dim=0)[:self.max_buffer_size]
        self.sample_buffer.clear()
        self.sample_buffer_size = 0

        # update the model
        self.training_cost += imgs_ftred.shape[0]
        noise = torch.randn_like(imgs_ftred)
        timesteps_idx = torch.randint(self.noise_idx, 0, (len(imgs_ftred),))
        timesteps = self.scheduler.timesteps[timesteps_idx].to(imgs_ftred.device)
        imgs_noisy = self.scheduler.add_noise(imgs_ftred, noise, timesteps=timesteps)
        noise_pred = self.training_model(imgs_noisy, timesteps)['sample']
        self.training_inference_nsteps += imgs_noisy.shape[0]

        loss = F.mse_loss(noise_pred, noise)
        self.log('train/loss', loss, sync_dist=True)
        self.log('train/inference_steps', self.training_inference_nsteps, sync_dist=True)

        self.log('train/training_cost', self.training_cost, sync_dist=True)
        self.log('train/sampling_cost', self.sampling_cost, sync_dist=True)

        self.current_acc_nsteps += 1

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        imgs_gt: torch.Tensor = images

        noise = torch.randn_like(imgs_gt)
        timesteps = self.scheduler.timesteps[self.noise_idx]
        imgs_noisy = self.scheduler.add_noise(imgs_gt, noise=noise, timesteps=timesteps)

        # sample from the model
        with torch.no_grad():
            imgs_recov, _ = self.denoising(self.training_model, imgs_noisy, self.noise_idx)
            psnr = self.psnr(imgs_recov, imgs_gt)
            ssim = self.ssim(imgs_recov, imgs_gt)
            self.log('valid/psnr', psnr, sync_dist=True)
            self.log('valid/ssim', ssim, sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        imgs_gt: torch.Tensor = images

        noise = torch.randn_like(imgs_gt)
        timesteps = self.scheduler.timesteps[self.noise_idx]
        imgs_noisy = self.scheduler.add_noise(imgs_gt, noise=noise, timesteps=timesteps)

        # sample from the model
        with torch.no_grad():
            imgs_recov, _ = self.denoising(self.training_model, imgs_noisy, self.noise_idx)

            psnr_raw = self.psnr(imgs_noisy, imgs_gt)
            ssim_raw = self.ssim(imgs_noisy, imgs_gt)

            psnr = self.psnr(imgs_recov, imgs_gt)
            ssim = self.ssim(imgs_recov, imgs_gt)
            self.log('test/psnr_raw', psnr_raw, sync_dist=True)
            self.log('test/ssim_raw', ssim_raw, sync_dist=True)
            self.log('test/psnr', psnr, sync_dist=True)
            self.log('test/ssim', ssim, sync_dist=True)
