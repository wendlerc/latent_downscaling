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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms

import logging
import wandb
import json

from muon import Muon
from torch import nn


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, patch_size=8, num_heads=16):
        super(AttentionBlock, self).__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
        self.patch_unembed = nn.Conv2d(in_channels, patch_size**2 * in_channels, kernel_size=1)
        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, in_channels, 64//patch_size, 64//patch_size))
        self.num_heads = num_heads
        self.dim_heads = in_channels // num_heads
        self.in_channels = in_channels
    
    def forward(self, x):
        # make patches
        patches = self.patch_embed(x) + self.pos
        # turn patches into q, k, v
        qkv = self.qkv(patches)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(x.shape[0], self.num_heads, -1, patches.shape[2]*patches.shape[3], self.dim_heads)
        k = k.reshape(x.shape[0], self.num_heads, -1, patches.shape[2]*patches.shape[3], self.dim_heads)
        v = v.reshape(x.shape[0], self.num_heads, -1, patches.shape[2]*patches.shape[3], self.dim_heads)
        mixed = nn.functional.scaled_dot_product_attention(q, k, v)
        # turn patches back into feature map
        mixed = mixed.reshape(x.shape[0], self.in_channels, patches.shape[2], patches.shape[3])
        feature_map_ = self.patch_unembed(mixed) # this has too many channels--> reshape those into patches
        feature_map = feature_map_.reshape(x.shape[0], -1, x.shape[2], x.shape[3])
        return feature_map + x

    
class SimpleResnetBlock2D(nn.Module):
    def __init__(self, in_channels):
        super(SimpleResnetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.leaky_relu2(out)
        
        return out


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
                 lr_adamw: float = 0.0002,
                 lr_muon: float = 0.02,
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
                 use_resizing: bool = False,
                 datapoints_per_epoch: int = None,
                 max_epochs: int = None,
                 use_resnet: bool = False,
                 use_attention: bool = False,
                 simple: bool = False,
                 gradient_clip_val: float = 0.0,
                 use_ema: bool = False,
                 ema_decay: float = 0.9999,
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
        self.datapoints_per_epoch = datapoints_per_epoch
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        if self.datapoints_per_epoch is not None:
            self.max_steps = self.max_epochs * self.datapoints_per_epoch // self.batch_size 
        self.lr_adamw = lr_adamw
        self.lr_muon = lr_muon
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        self.use_resizing = use_resizing
        self.scheduler = None
        self.optimizer = None
        self.use_resnet = use_resnet
        self.use_attention = use_attention
        self.automatic_optimization = False
        self.gradient_clip_val = gradient_clip_val
        self.simple = simple
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        # Build the model
        if self.use_resnet and self.simple:
            self.build_simple_resnet_model()
        elif self.use_resnet:
            self.build_resnet_model()
        else:
            self.build_convnet_model()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Initialize EMA if enabled
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)

    def build_convnet_model(self):
        # Create a 6-layer CNN for downscaling
        layers = []
        
        # First layer: input_channels -> hidden_channels
        layers.append(nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range((self.num_layers - 2)//2 - 1):
            layers.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, stride=2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        n_small = (self.num_layers - 2)//2
        if self.use_attention:
            layers.append(AttentionBlock(self.hidden_channels))
            n_small -= 1
        for _ in range(n_small):
            layers.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer: hidden_channels -> output_channels
        layers.append(nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3, padding=1))
        
        # Create the model
        self.model = nn.Sequential(*layers)
    

    def build_simple_resnet_model(self):
        # Create a 6-layer CNN for downscaling
        layers = []
        
        # First layer: input_channels -> hidden_channels
        layers.append(nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range((self.num_layers - 2)//2):
            layers.append(SimpleResnetBlock2D(self.hidden_channels))

        layers.append(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, stride=2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        n_small = (self.num_layers - 2)//2
        for _ in range(n_small):
            layers.append(SimpleResnetBlock2D(self.hidden_channels))
        
        # Final layer: hidden_channels -> output_channels
        layers.append(nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3, padding=1))
        
        # Create the model
        self.model = nn.Sequential(*layers)


    def build_resnet_model(self):
        # Create a 6-layer CNN for downscaling
        layers = []
        # upscale to hidden_channels
        layers.append(nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1))
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range((self.num_layers - 2)//2 - 1):
            layers.append(ResnetBlock2D(self.hidden_channels, self.hidden_channels, conv_shortcut=False))

        layers.append(Downsample2D(self.hidden_channels, self.hidden_channels))
        
        n_small = (self.num_layers - 2)//2
        if self.use_attention:
            layers.append(AttentionBlock(self.hidden_channels))
            n_small -= 1
        for _ in range(n_small):
            layers.append(ResnetBlock2D(self.hidden_channels, self.hidden_channels, conv_shortcut=False))
        
        # Final layer: hidden_channels -> output_channels
        layers.append(nn.GroupNorm(32, self.hidden_channels, eps=1e-05, affine=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3, stride=1, padding=1))
        
        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input: [B, C, 128, 128]
        if self.use_resizing:
            helper = transforms.Resize((x.shape[2]//2, x.shape[3]//2), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
            return self.model(x) + helper(x)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):            
        large_latents, small_latents = batch
        # Forward pass
        predicted_small_latents = self(large_latents)

        # Calculate loss (MSE)
        loss = F.mse_loss(predicted_small_latents, small_latents)
        
        # Log metrics
        self.log('epoch', float(self.current_epoch), on_step=True, on_epoch=True)
        self.log('loss', loss.item(), on_step=True, on_epoch=True)
        self.log('lr-adamw', self.adamw.param_groups[0]['lr'], on_step=True, on_epoch=True)
        self.log('lr-muon', self.muon.param_groups[0]['lr'], on_step=True, on_epoch=True)

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
            
            
            # Log parameter gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.log({f"grad_{name}": wandb.Histogram(param.grad.detach().cpu())})
        self.manual_backward(loss)
        # Log gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val, norm_type=2)
        self.log("grad_norm", grad_norm)
        self.muon.step()
        self.muon.zero_grad()
        adamw = self.optimizers()
        adamw.step()
        adamw.zero_grad()
        
        # Update EMA after optimizer step if enabled
        if self.use_ema:
            self.ema.update()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        large_latents, small_latents = batch
        
        # Apply EMA shadow weights for validation if enabled
        if self.use_ema:
            self.ema.apply_shadow()
        
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
        x = large_latents
        helper = transforms.Resize((x.shape[2]//2, x.shape[3]//2), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        helper_bilinear = transforms.Resize((x.shape[2]//2, x.shape[3]//2), interpolation=transforms.InterpolationMode.BILINEAR)
        helper_bicubic = transforms.Resize((x.shape[2]//2, x.shape[3]//2), interpolation=transforms.InterpolationMode.BICUBIC)
        helper_lanczos = transforms.Resize((x.shape[2]//2, x.shape[3]//2), interpolation=transforms.InterpolationMode.LANCZOS)
        baseline_mean = F.avg_pool2d(large_latents, kernel_size=2)
        self.log('baseline_nearest_exact', F.mse_loss(helper(large_latents), small_latents), on_step=True, on_epoch=True)
        self.log('baseline_zeros', F.mse_loss(torch.zeros_like(small_latents), small_latents), on_step=True, on_epoch=True)
        self.log('baseline_bilinear', F.mse_loss(helper_bilinear(large_latents), small_latents), on_step=True, on_epoch=True)
        self.log('baseline_bicubic', F.mse_loss(helper_bicubic(large_latents), small_latents), on_step=True, on_epoch=True)
        self.log('baseline_lanczos', F.mse_loss(helper_lanczos(large_latents), small_latents), on_step=True, on_epoch=True)
        self.log('baseline_mean', F.mse_loss(baseline_mean, small_latents), on_step=True, on_epoch=True)
        
        # Restore original weights after validation if EMA is enabled
        if self.use_ema:
            self.ema.restore()
        
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        # Apply EMA shadow weights before saving checkpoint if enabled
        if self.use_ema:
            self.ema.apply_shadow()
        # The model state will be saved with EMA weights if enabled
    
    def on_load_checkpoint(self, checkpoint):
        # If EMA is enabled, re-initialize EMA with the loaded model
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema.register()
       
    def configure_optimizers(self):
        lr_adamw = self.lr_adamw
        lr_muon = self.lr_muon
        b1 = self.b1
        b2 = self.b2
        weight_decay = self.weight_decay

        # Find ≥2D parameters in the body of the network -- these should be optimized by Muon
        muon_params = [p for p in self.parameters() if p.ndim >= 2]
        # Find everything else -- these should be optimized by AdamW
        adamw_params = [p for p in self.parameters() if p.ndim < 2]
        # Create the optimizer
        self.muon = Muon(muon_params, lr=lr_muon, momentum=0.95, rank=0, world_size=1)
        self.adamw = torch.optim.AdamW(adamw_params, lr=lr_adamw, betas=(b1, b2), weight_decay=weight_decay)
        
        if self.cosine_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.adamw, T_max=self.max_steps)
            return {
                "optimizer": self.adamw,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step"
                }
            }
        else:
            return self.adamw
    
    def url_to_dataloader(self, url, train=True):
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
        if self.datapoints_per_epoch is not None and train:
            loader = loader.with_epoch(self.datapoints_per_epoch//self.batch_size)
        return loader

    def train_dataloader(self):
        return self.url_to_dataloader(self.train_url, train=True)

    def val_dataloader(self):
        return self.url_to_dataloader(self.val_url, train=False)
    

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
                        strategy=args.strategy, log_every_n_steps=args.log_every_n_steps,
                        val_check_interval=args.checkpoint_every_n_examples//(args.batch_size))
    else:
        trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator=args.accelerator, strategy=args.strategy, 
                          log_every_n_steps=args.log_every_n_steps, val_check_interval=args.checkpoint_every_n_examples//(args.batch_size))

    if trainer.is_global_zero:
        # Wandb logging
        os.makedirs(args.checkpoint_path, exist_ok=True)
        wandb_logger = WandbLogger(project=args.wandb_project, 
                                log_model=False, 
                                save_dir=args.checkpoint_path,
                                config=vars(args),
                                name=args.experiment_name)
        run_name = wandb_logger.experiment.name

        # Configure the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, run_name),  # Define the path where checkpoints will be saved
            save_top_k=args.checkpoint_top_k,  # Set to -1 to save all epochs
            verbose=True,  # If you want to see a message for each checkpoint
            monitor='val_loss',  # Quantity to monitor
            mode='min',  # Mode of the monitored quantity
            every_n_train_steps=args.checkpoint_every_n_examples//(args.batch_size * trainer.world_size)+1,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer.callbacks.append(checkpoint_callback)
        trainer.callbacks.append(lr_monitor)
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
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--cosine_scheduler", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--max_steps", type=int, default=-1, help="number of steps of training")
    parser.add_argument("--lr_adamw", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lr_muon", type=float, default=0.02, help="muon: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay for optimizer")
    parser.add_argument("--datapoints_per_epoch", type=int, default=None, help="number of data points per epoch")
    parser.add_argument("--use_ema", default=False, action="store_true", help="use exponential moving average")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for exponential moving average")
    
    # checkpoint and logging parameters
    parser.add_argument("--checkpoint_path", type=str, default="models/latent_downscaler")
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--checkpoint_every_n_examples", type=int, default=100000)
    parser.add_argument("--checkpoint_top_k", type=int, default=5)
    
    # hardware parameters
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="ddp", help="ddp, ddp2, ddp_spawn, etc.")
    parser.add_argument("--accelerator", type=str, default="auto", help="auto, gpu, tpu, mpu, cpu, etc.")
    
    # logging parameters
    parser.add_argument("--wandb_project", type=str, default="latent_downscaler")
    parser.add_argument("--wandb_dir", type=str, default=".")
    parser.add_argument("--use_resizing", default=False, action="store_true")
    parser.add_argument("--experiment_name", type=str, default=None, help="optional name for the experiment")
    parser.add_argument("--use_resnet", default=False, action="store_true")
    parser.add_argument("--use_attention", default=False, action="store_true")
    parser.add_argument("--simple", default=False, action="store_true")
    parser.add_argument("--bn", default=False, action="store_true")
    
    hparams = parser.parse_args()

    main(hparams)