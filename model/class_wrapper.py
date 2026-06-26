import json
import time
from collections import defaultdict
import re, gc
import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import wandb
import math
import sys
import argparse
from typing import Union
from collections import OrderedDict

from .class_dataset import Alphabet
from .class_ESM2 import ESM2
from .utils import get_logger
from .class_loss import PPIMemoryInfoNCE, SimularityLoss, SupervisedContrastiveLoss, NewPPIMemoryInfoNCE, SupervisedTripletLoss, get_alpha

logger = get_logger(__name__)

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log

def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.mean(log[key])
        except:
            pass
    if log:
        out["entries"] = len(log[key])
    return out

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict

def load_config(path):
    cfg = argparse.Namespace()
    with open(path) as f:
        cfg.__dict__.update(json.load(f))
    return cfg

#####################################################################################

class ESMWrapper(pl.LightningModule):
    def __init__(self, cfg, args):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.args = args
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=not args.no_multimer,
        )
        
        checkpoint = torch.load('raw_MINT_param/mint.ckpt')
        if self.args.linearconstract_freeze:
            checkpoint = torch.load('650M_freeze_classify/PPLM-PPI.ckpt')
        new_checkpoint = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items())
        self.model.load_state_dict(new_checkpoint, strict=False)

        # self.constract_loss = NewPPIMemoryInfoNCE(hidden_size=2560, mem_size=256, temp=0.1, alpha=0.25, margin=0.5)
        self.constract_loss = SupervisedTripletLoss(margin=0.2, strategy='batch_all_triplets')
        # self.constract_loss = SupervisedContrastiveLoss(margin=1.0, gamma=2.0)
        self.iter_step = -1
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        # self._intermediate_buffers = []

        if args.wandb and self.global_rank == 0:  #
            wandb.init(
                settings=wandb.Settings(init_timeout=120),
                project="PLM_sequence_train",
                name="ESM_fine-tune",
                config={
                    "learning_rate": cfg.lr,
                    "batch_size": args.batch_size,
                    "model_arch": "ESM2"
                })

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = self.forward(batch)
        # if self.iter_step % 15000 == 0:
        #     self.model.save_pretrained(self.args.run_name)

        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        self.constract_loss.reset_stats()
        self.forward(batch)
        if self.args.validate:
            self.try_print_log()

    def forward(self, batch):
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

        self.iter_step += 1
        try:
            tokens, chains_id, class_id = batch
            mask = ((~tokens.eq(self.model.cls_idx)) \
                  & (~tokens.eq(self.model.eos_idx)) \
                  & (~tokens.eq(self.model.padding_idx)))

            mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask
            inp = torch.where(mask, self.model.mask_idx, tokens)
            # self._intermediate_buffers.append(mask) 
            output = self.model(inp, chains_id)
            # self._release_intermediates()

            logits = output['logits']
            embedding = output['linear_feature']

            loss_fct = torch.nn.CrossEntropyLoss()
            entro_loss = loss_fct(logits, class_id[:, 0])
            cons_loss = self.constract_loss(embedding, class_id[:, 0])

            alpha = get_alpha(self.trainer.current_epoch, total_epochs=10, max_alpha=0.8, min_alpha=0.2, scheduler_type="cosine")

            if self.args.linearconstract_freeze:
                loss = 0.8 * entro_loss
            else:
                loss = (1 - alpha) * entro_loss + alpha * cons_loss

            self.log("tokens", mask.sum())
            self.log("entro_loss", entro_loss)
            self.log("cons_loss", cons_loss)
            self.log("loss", loss)
            self.log("perplexity", torch.exp(loss))
            self.log("dur", time.time() - self.last_log_time)
            self.last_log_time = time.time()

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError("==============No GPU memory")

        finally:
            if torch.cuda.memory_reserved() > 0.75 * 40e9:
                print("====================free memory")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        return loss

    def try_print_log(self):
        step = self.iter_step if self.args.validate else self.trainer.global_step
        if (step + 1) % self.args.print_freq == 0:
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = gather_log(log, self.trainer.world_size)
            mean_log = get_log_mean(log)
            mean_log.update(
                {
                    "epoch": self.trainer.current_epoch,
                    "step": self.trainer.global_step,
                    "iter_step": self.iter_step,
                }
            )
            if self.trainer.is_global_zero:
                logger.info(str(mean_log))
                if self.args.wandb:
                    wandb.log(mean_log)
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().item()
        log = self._log
        log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def on_train_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        mean_log.update(
            {
                "epoch": self.trainer.current_epoch,
                "step": self.trainer.global_step,
                "iter_step": self.iter_step,
            }
        )

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        if self.trainer.is_global_zero:
            logger.info(str(get_log_mean(log)))
            if self.args.wandb:
                wandb.log(get_log_mean(log))

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    def on_before_optimizer_step(self, optimizer):
        self.try_print_log()
        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is None:
                    print(name)

    def configure_optimizers(self):
        # For model training optimization, we used Adam with β1 = 0.9, β2 = 0.98, ε = 10⁻⁸ and L2 weight decay of
        # 0.01 for all models except the 15 billion parameter model, where we used a weight decay of 0.1. The learning rate is
        # warmed up over the first 2,000 steps to a peak value of 4e-4 (1.6e-4 for the 15B parameter model), and then linearly
        # decayed to one tenth of its peak value over the 90% of training duration
        self.model.requires_grad_(False)
        total_layers = self.cfg.encoder_layers  # 33

        for name, param in self.model.named_parameters():
            if "linearconstract" in name or "classifier" in name:
                param.requires_grad = True

            if "linearconstract" in name and self.args.linearconstract_freeze:
                param.requires_grad = False

            if "emb_layer_norm_after" in name and (self.args.pair_training_percent > 0 or self.args.single_training_percent > 0):
                param.requires_grad = True

            elif "layers" in name and "multimer_attn" in name:
                layer_num = name.split(".")[1]
                if int(layer_num) > (total_layers - math.floor(total_layers * self.args.pair_training_percent)):
                    param.requires_grad = True

            elif "layers" in name and "multimer_attn" not in name:
                layer_num = name.split(".")[1]
                if int(layer_num) > (total_layers - math.floor(total_layers * self.args.single_training_percent)):
                    param.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr[0],
            betas=json.loads(self.cfg.adam_betas),
            eps=self.cfg.adam_eps,
            weight_decay=self.cfg.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-12, end_factor=1.0, total_iters=self.cfg.warmup_updates
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.end_learning_rate / self.cfg.lr[0],
            total_iters=int(0.9 * int(self.cfg.total_num_update)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[self.cfg.warmup_updates]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _release_intermediates(self):
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        if mem_allocated > 0.8 * 40:  # Release when GPU memory exceeds 80%
            while len(self._intermediate_buffers) > 1:  # Keep only the most recent result
                buf = self._intermediate_buffers.pop(0)
                if isinstance(buf, torch.Tensor):
                    del buf
            gc.collect()
            torch.cuda.empty_cache()

#############################################################################################################

class ESM_Inference_Wrapper(nn.Module):
    def __init__(
        self,
        cfg,
        checkpoint_path,
        use_multimer=True,
        sep_chains=False,
        device="cuda:0",
    ):
        super().__init__()
        self.cfg = cfg
        self.sep_chains = sep_chains
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=use_multimer,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if use_multimer:
            new_checkpoint = OrderedDict(
                (key.replace("model.", ""), value)
                for key, value in checkpoint["state_dict"].items())
            self.model.load_state_dict(new_checkpoint)

        else:
            new_checkpoint = upgrade_state_dict(checkpoint["model"])
            self.model.load_state_dict(new_checkpoint)
        total_layers = cfg.encoder_layers
        for name, param in self.model.named_parameters():
            if "embed_tokens.weight" in name or "layers" in name or "lm_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = False
                    
        self.model.to(device)

    def forward(self, test_dataloader, device, threshold=0.5):
        test_predicted_list = []
        test_label_list = []
        prob_list = []

        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    
        for tokens, chains_id, class_id in test_dataloader:
            mask = ((~tokens.eq(self.model.cls_idx)) \
                  & (~tokens.eq(self.model.eos_idx)) \
                  & (~tokens.eq(self.model.padding_idx)))

            mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask

            inp = tokens[mask]
            logits = self.model(inp, chains_id)['logits']

            probs = torch.nn.functional.softmax(logits, dim=1)
            prob_list.append(probs.cpu().numpy())
          
            class1_probs = probs[:, 1]
            predicted = torch.where(class1_probs >= threshold, 
                                    torch.ones_like(class1_probs), 
                                    torch.zeros_like(class1_probs))
            
            test_predicted_list.append(predicted.cpu().numpy())
            test_label_list.append(class_id[:, 0].view(-1).cpu().numpy())

        # Convert to numpy arrays
        y_true = np.hstack(test_label_list)
        y_pred = np.hstack(test_predicted_list)
        y_probs = np.vstack(prob_list)[:, 1]  # Get probability of positive class
    
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        auc = roc_auc_score((1 - y_true), y_probs)
    
        print(classification_report(y_true, y_pred, target_names=['class_0(neg)', 'class_1(posi)']))
        print(f"AUC: {auc:.4f}")
    
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'y_probs': y_probs,
            'y_pred': y_pred,
            'y_true': y_true
        }
