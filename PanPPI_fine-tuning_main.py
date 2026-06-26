##### run PLM_fine-tune
import argparse
import json
import yaml
import re
import sys
import lightning as pl
import torch
from collections import OrderedDict
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from model.class_dataset import Class_CSVDataset,Class_CollateFn, Alphabet
from model.class_wrapper import ESMWrapper,upgrade_state_dict,ESM_Inference_Wrapper
from transformers import EsmForSequenceClassification
import os
import random
import numpy as np
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # ?????[11,12](@ref)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,roundup_power2_divisions:4,garbage_collection_threshold:0.95"
torch.set_float32_matmul_precision('medium')
os.environ["WANDB_MODE"] = "offline"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

cfg = argparse.Namespace()
with open("raw_MINT_param/esm2_t33_650M_UR50D.json") as f:
    cfg.__dict__.update(json.load(f))

args = argparse.Namespace()
with open("args_parameter_PanPPI_fine-tuning.yaml") as f:
    args.__dict__.update(yaml.safe_load(f))

# in-cluster
train_dataset = Class_CSVDataset('dataset/linear_in-cluster_train_1-30.csv',\
                                  'source','target', 'species', 'class','dataset/Eukaryotes_protein_len50_reportset_fasta.h5') 
train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=False, num_workers=4, batch_size=args.batch_size, collate_fn=Class_CollateFn(args.crop_length), shuffle=True)

valid_dataset = Class_CSVDataset('dataset/linear_in-cluster_valid_1-30.csv',\
                                  'source','target', 'species', 'class','dataset/Eukaryotes_protein_len50_reportset_fasta.h5')
valid_loader = torch.utils.data.DataLoader(valid_dataset, pin_memory=False, num_workers=4, batch_size=args.batch_size, collate_fn=Class_CollateFn(args.crop_length), shuffle=False)

model = ESMWrapper(cfg, args)

import torch
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from datetime import timedelta

class NCCLMonitor(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_rank == 0:  
            print("\n" + "="*30 + f" Batch {batch_idx} Start " + "="*30)
            print(torch.cuda.memory_summary())
            
            if hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_push("Communication_Phase")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            if hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
            
            print(f"Peak Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            print("="*30 + f" Batch {batch_idx} End " + "="*30 + "\n")


trainer = pl.Trainer(
    default_root_dir=f"./{args.run_name}",
    accelerator="gpu",
    #strategy='ddp',
    devices=[0,1],
    max_steps=args.max_steps,
    num_sanity_val_steps=0,
    enable_progress_bar=not args.wandb,
    gradient_clip_val=args.grad_clip,
    enable_checkpointing=True,
    callbacks=[ModelCheckpoint(dirpath=f"./{args.run_name}"), NCCLMonitor()],
    #callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_check_interval,
    precision="bf16-mixed",
    strategy=DDPStrategy(
             find_unused_parameters=True,
             static_graph=False,
             gradient_as_bucket_view=True,
             timeout=timedelta(minutes=30)), 
)

if args.validate:
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.validate(model, valid_loader)
else:
    #trainer.fit(model, train_loader, valid_loader, ckpt_path="650M_freeze_classify/epoch=1-step=13210.ckpt") # continue train
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None) # init training or linear_contrast

