import os
import io
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import webdataset as wds

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import logging
import wandb
import json

# from https://github.com/cloneofsimo/minSDXL/blob/master/sdxl_rewrite.py

from torch import nn

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)

class LatentDownscaler(LightningModule):
    def __init__(self,
                 input_channels: int = 4,
                 output_channels: int = 4,
                 hidden_channels: int = 64,
                 num_layers: int = 6,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 weight_decay: float = 0,
                 batch_size: int = 64, 
                 num_workers: int = 0,
                 log_every_n_steps: int = 100,
                 train_url: str = None,
                 val_url: str = None,
                 cosine_scheduler: bool = False,
                 max_steps: int = -1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.train_url = train_url
        self.val_url = val_url
        
        self.cosine_scheduler = cosine_scheduler
        self.max_steps = max_steps
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        
        # Build the model
        self.build_model()
        
    def build_model(self):
        # Create a 6-layer CNN for downscaling
        layers = []
        # upscale to hidden_channels
        layers.append(nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1))
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range((self.num_layers - 2)//2 - 1):
            layers.append(ResnetBlock2D(self.hidden_channels, self.hidden_channels, conv_shortcut=False))

        layers.append(Downsample2D(self.hidden_channels, self.hidden_channels))
        
        for _ in range((self.num_layers - 2)//2):
            layers.append(ResnetBlock2D(self.hidden_channels, self.hidden_channels, conv_shortcut=False))
        
        # Final layer: hidden_channels -> output_channels
        layers.append(nn.GroupNorm(32, self.hidden_channels, eps=1e-05, affine=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3, stride=1, padding=1))
        
        # Create the model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input: [B, C, 128, 128]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        large_latents, small_latents = batch
        # Forward pass
        predicted_small_latents = self(large_latents)
        
        # Calculate loss (MSE)
        loss = F.mse_loss(predicted_small_latents, small_latents)
        
        # Log metrics
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr)
        self.log('epoch', float(self.current_epoch), on_step=True, on_epoch=True)
        self.log('loss', loss.item(), on_step=True, on_epoch=True)

        # Calculate and log additional metrics
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            # Calculate PSNR
            mse = F.mse_loss(predicted_small_latents, small_latents)
            psnr = 10 * torch.log10(1.0 / mse)
            self.log('psnr', psnr.item(), on_step=True, on_epoch=True)
            
            # Log histograms
            self.logger.experiment.log({
                'predictions': wandb.Histogram(predicted_small_latents.detach().cpu()),
                'targets': wandb.Histogram(small_latents.detach().cpu())
            })
            
            # Log gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2, norm_type=2)
            self.log("grad_norm", grad_norm)
            
            # Log parameter gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.log({f"grad_{name}": wandb.Histogram(param.grad.detach().cpu())})

        return loss
    
    def validation_step(self, batch, batch_idx):
        large_latents, small_latents = batch
        # Forward pass
        predicted_small_latents = self(large_latents)
        
        # Calculate loss (MSE)
        loss = F.mse_loss(predicted_small_latents, small_latents)
        
        # Calculate PSNR
        mse = F.mse_loss(predicted_small_latents, small_latents)
        psnr = 10 * torch.log10(1.0 / mse)
        
        # Log metrics
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('val_psnr', psnr.item(), on_step=True, on_epoch=True)
        
        return loss
       
    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        weight_decay = self.weight_decay
        
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        
        if self.cosine_scheduler:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_steps),
                    "interval": "step"
                }
            }
        else:
            return opt
    
    def url_to_dataloader(self, url):
        def log_and_continue(exn):
            """Call in an exception handler to ignore any exception, issue a warning, and continue."""
            logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
            return True
        
        def load_tensor(z):
            return torch.load(io.BytesIO(z), map_location='cpu').to(torch.float32)
        
        pipeline = [
            wds.SimpleShardList(url),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(bufsize=5000, initial=1000),
            wds.rename(large="latent_big.pt", small="latent_small.pt"),
            wds.map_dict(large=load_tensor, small=load_tensor),
            wds.to_tuple("large", "small"),
            wds.batched(self.batch_size, partial=False),
        ]

        dataset = wds.DataPipeline(*pipeline)

        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False, num_workers=self.num_workers,
        )
        return loader

    def train_dataloader(self):
        return self.url_to_dataloader(self.train_url)

    def val_dataloader(self):
        return self.url_to_dataloader(self.val_url)
    

def main(args: Namespace) -> None:
    os.environ['WANDB_DIR'] = args.wandb_dir
    torch.set_float32_matmul_precision('high')
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LatentDownscaler(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    
    if args.devices is not None:
        trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator=args.accelerator, devices=args.devices, 
                        strategy=args.strategy, gradient_clip_val=args.gradient_clip_val, log_every_n_steps=args.log_every_n_steps)
    else:
        trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator=args.accelerator, strategy=args.strategy,
                        gradient_clip_val=args.gradient_clip_val, log_every_n_steps=args.log_every_n_steps)

    if trainer.is_global_zero:
        # Wandb logging
        os.makedirs(args.checkpoint_path, exist_ok=True)
        wandb_logger = WandbLogger(project=args.wandb_project, 
                                log_model=False, 
                                save_dir=args.checkpoint_path,
                                config=vars(args))
        run_name = wandb_logger.experiment.name

        # Configure the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, run_name),  # Define the path where checkpoints will be saved
            save_top_k=args.checkpoint_top_k,  # Set to -1 to save all epochs
            verbose=True,  # If you want to see a message for each checkpoint
            monitor='loss',  # Quantity to monitor
            mode='min',  # Mode of the monitored quantity
            every_n_train_steps=args.checkpoint_every_n_examples//(args.batch_size * trainer.world_size),
        )

        trainer.callbacks.append(checkpoint_callback)
        trainer.logger = wandb_logger


    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    # model parameters
    parser.add_argument("--input_channels", type=int, default=4, help="number of input channels")
    parser.add_argument("--output_channels", type=int, default=4, help="number of output channels")
    parser.add_argument("--hidden_channels", type=int, default=64, help="number of hidden channels")
    parser.add_argument("--num_layers", type=int, default=6, help="number of convolutional layers")
    
    # data parameters
    parser.add_argument("--train_url", type=str, default='/share/datasets/datasets/laicoyo_latent_pairs/{000000..000080}.tar')
    parser.add_argument("--val_url", type=str, default='/share/datasets/datasets/laicoyo_latent_pairs/{000081..000090}.tar')
    
    # training parameters
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--cosine_scheduler", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--max_steps", type=int, default=-1, help="number of steps of training")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay for optimizer")
    
    # checkpoint and logging parameters
    parser.add_argument("--checkpoint_path", type=str, default="models/latent_downscaler")
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--checkpoint_every_n_examples", type=int, default=50000)
    parser.add_argument("--checkpoint_top_k", type=int, default=5)
    
    # hardware parameters
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="ddp", help="ddp, ddp2, ddp_spawn, etc.")
    parser.add_argument("--accelerator", type=str, default="auto", help="auto, gpu, tpu, mpu, cpu, etc.")
    
    # logging parameters
    parser.add_argument("--wandb_project", type=str, default="latent_downscaler")
    parser.add_argument("--wandb_dir", type=str, default=".")
    
    hparams = parser.parse_args()

    main(hparams)