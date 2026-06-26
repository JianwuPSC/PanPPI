import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple
import logging
import os
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, classification_report

class SimularityLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=2, temp=0.2):
        super().__init__()
        self.alpha = alpha  # positive sample weight
        self.gamma = gamma  # hard sample focusing parameter
        self.temp = temp

    def forward(self, z1, labels):
        z1 = F.normalize(z1, p=2, dim=-1)
        sim = torch.mm(z1, z1.T) / self.temp

        pos_mask = (labels.unsqueeze(1) == 1) & (labels.unsqueeze(0) == 1)
        pos_mask.fill_diagonal_(False)
        pos_sim = sim[pos_mask]
        pos_loss = -self.alpha * F.logsigmoid(pos_sim) * (1 - torch.sigmoid(pos_sim + 1e-8)).pow(self.gamma)

        neg_mask = ((labels.unsqueeze(1) == 0) & (labels.unsqueeze(0) == 0))
        neg_mask.fill_diagonal_(False)
        neg_sim = sim[neg_mask]

        batch_size = sim.size(0)
        valid_cols = neg_sim.size(0) // batch_size
        neg_sim = neg_sim[:batch_size * valid_cols].view(batch_size, valid_cols)
        topk = min(1024, neg_sim.size(1))  # Select Top-K hard negative samples
        hard_neg = torch.topk(neg_sim, k=topk, dim=1)[0]
        neg_loss = -(1 - self.alpha) * F.logsigmoid(-hard_neg) * torch.sigmoid(hard_neg + 1e-8).pow(self.gamma)

        return pos_loss.mean() + neg_loss.mean()

####################################################################################

class PPIMemoryInfoNCE(nn.Module):
    """PPI-specific InfoNCE loss with memory bank (optimized for 1:3 imbalance)"""
    def __init__(self, hidden_size=512, mem_size=4096, temp=0.1, alpha=0.25, margin=0.5):
        """
        Args:
            mem_size: total size of memory bank
            temp: temperature parameter
            alpha: weight for interacting pairs
            margin: decision boundary for interacting/non-interacting
        """
        super().__init__()
        self.temp = temp
        self.alpha = alpha
        self.margin = margin
        
        # Separate memory banks: interacting and non-interacting pairs
        ppi_bank = torch.randn(mem_size, hidden_size)
        nn.init.xavier_normal_(ppi_bank, gain=1.0)
        self.register_buffer("ppi_bank", F.normalize(ppi_bank, p=2, dim=-1))
        self.register_buffer("label_bank", torch.zeros(mem_size, dtype=torch.long))  # labels
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        
        # Ratio statistics
        self.register_buffer("ppi_count", torch.tensor(0))  # count of interacting pairs
        self.register_buffer("non_ppi_count", torch.tensor(0))  # count of non-interacting pairs

    def update_memory(self, pair_features, labels):
        """Update memory bank (separate interacting and non-interacting protein pairs)"""
        # Normalize features
        pair_features = F.normalize(pair_features, p=2, dim=-1)
        # Separate interacting and non-interacting pairs
        ppi_mask = (labels == 1)
        non_ppi_mask = (labels == 0)
        # Update global counts
        self.ppi_count += ppi_mask.sum().item()
        self.non_ppi_count += non_ppi_mask.sum().item()
        # Update memory bank
        ptr = int(self.ptr)
        batch_size = pair_features.size(0)
        # Compute memory space
        end_idx = min(ptr + batch_size, self.ppi_bank.size(0))
        replace_len = end_idx - ptr
        
        if replace_len > 0:
            # Update front part of memory bank
            self.ppi_bank[ptr:end_idx] = pair_features[:replace_len]
            self.label_bank[ptr:end_idx] = labels[:replace_len]
        
        # Handle overflow samples
        if replace_len < batch_size:
            remaining = batch_size - replace_len
            # Update beginning of memory bank
            self.ppi_bank[:remaining] = pair_features[replace_len:]
            self.label_bank[:remaining] = labels[replace_len:]
        # Update pointer
        self.ptr[0] = (ptr + batch_size) % self.ppi_bank.size(0)
    
    def check_memory_health(self):
        """Check memory state"""
        # Check NaN/Inf
        if torch.isnan(self.ppi_bank).any() or torch.isinf(self.ppi_bank).any():
            print("inf or nan exist; reset !!!")
            self._reset_memory()
            return False
    
        # Label distribution
        ppi_count = (self.label_bank == 1).sum().item()
        non_ppi_count = (self.label_bank == 0).sum().item()
    
        if ppi_count == 0 or non_ppi_count == 0:
            print("no positive or negative label exist; reset !!!")
            self._reset_memory()
            return False
    
        return True

    def _reset_memory(self):
        """Safely reset memory bank"""
        # Save original size
        mem_size = self.ppi_bank.size(0)
        hidden_size = self.ppi_bank.size(1)
    
        # Create new memory bank
        new_bank = torch.randn(mem_size, hidden_size)
        nn.init.xavier_normal_(new_bank)
        new_bank = F.normalize(new_bank, p=2, dim=-1)
    
        # Create new label bank (balanced distribution)
        new_labels = torch.zeros(mem_size, dtype=torch.long)
        new_labels[:mem_size//2] = 0  # 50% positive samples
        new_labels[mem_size//2:] = 1  # 50% negative samples
    
        # Reset pointer and counters
        self.ppi_bank.copy_(new_bank)
        self.label_bank.copy_(new_labels)
        self.ptr.zero_()
        self.ppi_count.fill_(mem_size//2)
        self.non_ppi_count.fill_(mem_size//2)
        print("memory reset")

    def forward(self, pair_features, labels):
        """Compute PPI-specific loss"""
        if not self.check_memory_health():
            return torch.tensor(0.0, device=pair_features.device)
        # Update memory bank
        self.update_memory(pair_features.detach(), labels.detach()) 
        # Normalize features
        query = F.normalize(pair_features, p=2, dim=-1)
        # Retrieve memory bank data
        memory = self.ppi_bank.to(query.device)
        mem_labels = self.label_bank.to(query.device)
        # Compute similarity matrix
        sim_matrix = query @ memory.T

        # Dynamic temperature adjustment
        sim_std = sim_matrix.std().detach()
        adaptive_temp = self.temp * (1 + 0.5 * torch.sigmoid(sim_std - 1.0))
        ppi_count = max(1, (mem_labels == 1).sum().item())
        non_ppi_count = max(1, (mem_labels == 0).sum().item())
        balance_factor = torch.clamp(torch.tensor(non_ppi_count / ppi_count), 0.33, 3.0)
    
        same_label = labels.unsqueeze(1) == mem_labels.unsqueeze(0)

        ppi_mask = same_label & (mem_labels == 1).unsqueeze(0)
        ppi_sim = torch.where(ppi_mask, sim_matrix, -1e8)  # Use a large negative number instead of -inf
        ppi_logits = torch.logsumexp(ppi_sim / adaptive_temp, dim=1)
    
        non_ppi_mask = same_label & (mem_labels == 0).unsqueeze(0)
        non_ppi_sim = torch.where(non_ppi_mask, sim_matrix, -1e8)
        non_ppi_logits = torch.logsumexp(non_ppi_sim / adaptive_temp, dim=1)

        # Filter queries without positive samples
        valid_mask = ~torch.isnan(ppi_logits) & ~torch.isinf(ppi_logits)
        if valid_mask.any():
            ppi_loss = -ppi_logits[valid_mask].mean() * balance_factor
        else:
            ppi_loss = torch.tensor(0.0, device=sim_matrix.device)

        valid_mask = ~torch.isnan(non_ppi_logits) & ~torch.isinf(non_ppi_logits)
        if valid_mask.any():
            non_ppi_loss = -non_ppi_logits[valid_mask].mean() / balance_factor
        else:
            non_ppi_loss = torch.tensor(0.0, device=sim_matrix.device)

        # Max absolute math
        total_loss = (100 - ppi_loss) / 100 + (100 - non_ppi_loss) / 100

        total_loss = torch.clamp(total_loss, min=0.0)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"error ppi_loss={ppi_loss.item()}, non_ppi_loss={non_ppi_loss.item()}")
            return torch.tensor(0.0, device=pair_features.device)

        return total_loss

################################################################################################

class NewPPIMemoryInfoNCE(nn.Module):

    def __init__(self, hidden_size=512, mem_size=4096, temp=0.1, alpha=0.25, margin=0.5):
        """
        Args:
            mem_size: total size of memory bank
            temp: temperature parameter
            alpha: weight for interacting pairs
            margin: decision boundary for interacting/non-interacting
        """
        super().__init__()
        self.temp = temp
        self.alpha = alpha
        self.margin = margin
        
        # Separate memory banks: interacting and non-interacting pairs
        ppi_bank = torch.randn(mem_size, hidden_size)
        nn.init.xavier_normal_(ppi_bank, gain=1.0)
        self.register_buffer("ppi_bank", F.normalize(ppi_bank, p=2, dim=-1))
        self.register_buffer("label_bank", torch.zeros(mem_size, dtype=torch.long))  # labels
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long)) 
        # Ratio statistics
        self.register_buffer("ppi_count", torch.tensor(0))  # count of interacting pairs
        self.register_buffer("non_ppi_count", torch.tensor(0))  # count of non-interacting pairs

    def update_memory(self, pair_features, labels):
        """Update memory bank (separate interacting and non-interacting protein pairs)"""
        # Normalize features
        pair_features = F.normalize(pair_features, p=2, dim=-1)
        # Separate interacting and non-interacting pairs
        ppi_mask = (labels == 1)
        non_ppi_mask = (labels == 0)
        # Update global counts
        self.ppi_count += ppi_mask.sum().item()
        self.non_ppi_count += non_ppi_mask.sum().item()
        # Update memory bank
        ptr = int(self.ptr)
        batch_size = pair_features.size(0)
        # Compute memory space
        end_idx = min(ptr + batch_size, self.ppi_bank.size(0))
        replace_len = end_idx - ptr
        
        if replace_len > 0:
            # Update front part of memory bank
            self.ppi_bank[ptr:end_idx] = pair_features[:replace_len]
            self.label_bank[ptr:end_idx] = labels[:replace_len]
        # Handle overflow samples
        if replace_len < batch_size:
            remaining = batch_size - replace_len
            # Update beginning of memory bank
            self.ppi_bank[:remaining] = pair_features[replace_len:]
            self.label_bank[:remaining] = labels[replace_len:]
        # Update pointer
        self.ptr[0] = (ptr + batch_size) % self.ppi_bank.size(0)
    
    def check_memory_health(self):
        """Check memory state"""
        # Check NaN/Inf
        if torch.isnan(self.ppi_bank).any() or torch.isinf(self.ppi_bank).any():
            print("inf or nan exist; reset !!!")
            self._reset_memory()
            return False
    
        # Label distribution
        ppi_count = (self.label_bank == 1).sum().item()
        non_ppi_count = (self.label_bank == 0).sum().item()
    
        if ppi_count == 0 or non_ppi_count == 0:
            print("no positive or negative label exist; reset !!!")
            self._reset_memory()
            return False

        return True

    def _reset_memory(self):
        """Safely reset memory bank"""
        # Save original size
        mem_size = self.ppi_bank.size(0)
        hidden_size = self.ppi_bank.size(1)
    
        # Create new memory bank
        new_bank = torch.randn(mem_size, hidden_size)
        nn.init.xavier_normal_(new_bank)
        new_bank = F.normalize(new_bank, p=2, dim=-1)
    
        # Create new label bank (balanced distribution)
        new_labels = torch.zeros(mem_size, dtype=torch.long)
        new_labels[:mem_size//2] = 1  # 50% positive samples
        new_labels[mem_size//2:] = 0  # 50% negative samples
    
        # Reset pointer and counters
        self.ppi_bank.copy_(new_bank)
        self.label_bank.copy_(new_labels)
        self.ptr.zero_()
        self.ppi_count.fill_(mem_size//2)
        self.non_ppi_count.fill_(mem_size//2)

        print("memory reset")

    def forward(self, pair_features, labels):
        """Compute PPI-specific loss"""
        if not self.check_memory_health():
            return torch.tensor(0.0, device=pair_features.device)
        # 
        self.update_memory(pair_features.detach(), labels.detach()) 
        # 
        query = F.normalize(pair_features, p=2, dim=-1)
        # 
        memory = self.ppi_bank.to(query.device)
        mem_labels = self.label_bank.to(query.device)
        #
        sim_matrix = query @ memory.T
        # 
        sim_std = sim_matrix.std().detach()
        adaptive_temp = self.temp * (1 + 0.5 * torch.sigmoid(sim_std - 1.0))
        ppi_count = max(1, (mem_labels == 1).sum().item())
        non_ppi_count = max(1, (mem_labels == 0).sum().item())
        balance_factor = torch.clamp(torch.tensor(non_ppi_count / ppi_count), 0.33, 3.0)
        
        same_label = labels.unsqueeze(1) == mem_labels.unsqueeze(0)        
        all_logits = torch.logsumexp(sim_matrix / adaptive_temp, dim=1)
        
        ppi_mask = same_label & (mem_labels == 1).unsqueeze(0)
        ppi_sim = torch.where(ppi_mask, sim_matrix, -1e8)  # -inf
        ppi_logits = torch.logsumexp(ppi_sim / adaptive_temp, dim=1)
        ppi_logits = all_logits - ppi_logits

        valid_mask = ~torch.isnan(ppi_logits) & ~torch.isinf(ppi_logits)
        if valid_mask.any():
            ppi_loss = ppi_logits[valid_mask].mean() * balance_factor
        else:
            ppi_loss = torch.tensor(0.0, device=sim_matrix.device)

        total_loss = ppi_loss
        total_loss = torch.clamp(total_loss, min=0.0)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"error ppi_loss={ppi_loss.item()}")
            return torch.tensor(0.0, device=pair_features.device)

        return total_loss


#################################################################################################

class SupervisedContrastiveLoss(nn.Module):
    """ """
    def __init__(self, margin=1.0, gamma=1.0):
        """
        Args:
            margin:
            gamma: gamma=0
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]
        Returns:
            torch.Tensor:
        """
        # 1.
        features_norm = F.normalize(features, p=2, dim=-1)
        distance_matrix = 1.0 - torch.mm(features_norm, features_norm.T)  # [0,2]
        # 2.
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        # 3.
        eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        same_class = same_class & ~eye
        pos_mask = (labels.unsqueeze(1) == 1) & (labels.unsqueeze(0) == 1)
        pos_mask.fill_diagonal_(False)
        neg_mask = (labels.unsqueeze(1) == 0) & (labels.unsqueeze(0) == 0)
        neg_mask.fill_diagonal_(False)
        # 4. 
        pos_distance = distance_matrix[pos_mask]
        neg_distance = distance_matrix[neg_mask]
        diff_distance = distance_matrix[~same_class & ~eye]
        # 5.
        if len(pos_distance) == 0:
            pos_loss = torch.tensor(0.0)
        else:
            # 
            pos_weights = torch.exp(-self.gamma * pos_distance) if self.gamma > 0 else 1.0
            pos_loss = torch.mean(pos_weights * pos_distance.pow(2))
        # 5.
        if len(neg_distance) == 0:
            neg_loss = torch.tensor(0.0)
        else:
            # 
            neg_weights = torch.exp(-self.gamma * neg_distance) if self.gamma > 0 else 1.0
            neg_loss = torch.mean(neg_weights * neg_distance.pow(2))        
        
        if len(diff_distance) == 0:
            diff_loss = torch.tensor(0.0)
        else:
            # Negative pair loss: distance between different classes should be at least margin
            margin_penalty = F.relu(self.margin - diff_distance)
            diff_weights = torch.exp(self.gamma * (diff_distance - self.margin)) if self.gamma > 0 else 1.0
            diff_loss = torch.mean(diff_weights * margin_penalty.pow(2))

        return (pos_loss + neg_loss + diff_loss) * 10

###################################################################################

class SupervisedTripletLoss(nn.Module):
    """ """
    def __init__(self, margin=0.5, strategy='adaptive', temperature=0.1, 
                 min_pos_count=3, min_neg_count=3):
        """
        Args:
            margin:
            strategy: 'adaptive'/'batch_hard'
            temperature:
            min_pos_count:
            min_neg_count:
        """
        super().__init__()
        self.margin = margin
        self.strategy = strategy
        self.temperature = temperature
        self.min_pos_count = min_pos_count
        self.min_neg_count = min_neg_count
        self.eps = 1e-6
        self.register_buffer("avg_pos_dist", torch.tensor(0.5))
        self.register_buffer("avg_neg_dist", torch.tensor(1.5))
        self.momentum = 0.1
        
    def update_distance_stats(self, dist_matrix, pos_mask, neg_mask):
        """ """
        with torch.no_grad():
            pos_dist = dist_matrix[pos_mask]
            if pos_dist.numel() > 0:
                self.avg_pos_dist = (1 - self.momentum) * self.avg_pos_dist + self.momentum * pos_dist.mean()
            neg_dist = dist_matrix[neg_mask]
            if neg_dist.numel() > 0:
                self.avg_neg_dist = (1 - self.momentum) * self.avg_neg_dist + self.momentum * neg_dist.mean()
    
    def adaptive_margin(self):
        """ """
        base_margin = self.margin
        distance_diff = self.avg_neg_dist - self.avg_pos_dist
        dynamic_margin = torch.sigmoid((distance_diff - 1.0) * self.temperature) * 0.5
        margin = torch.clamp(base_margin + dynamic_margin, 0.2, 1.0)

        print(f"Margin: {margin}, avg_pos_dist: {self.avg_pos_dist}, avg_neg_dist: {self.avg_neg_dist}") 
        return margin
    
    def forward(self, features, labels):
        """ """
        if len(labels) < self.min_pos_count + self.min_neg_count:
            return torch.tensor(0.0, device=features.device)
        #
        features = F.normalize(features, p=2, dim=-1) 
        dist_matrix = torch.cdist(features, features, p=2)
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=features.device)
        same_class = same_class & ~eye  # exclude self
        ppi_mask = same_class & (labels == 1).unsqueeze(1)
        ppi_mask = ppi_mask & (labels == 1).unsqueeze(0)
        non_ppi_mask = same_class & (labels == 0).unsqueeze(1)
        non_ppi_mask = non_ppi_mask & (labels == 0).unsqueeze(0)
        diff_mask = ~same_class & ~eye

        self.update_distance_stats(dist_matrix, ppi_mask, diff_mask)

        if self.strategy == 'adaptive' or self.strategy == 'batch_hard':
            posi_loss = self.batch_hard_triplets(dist_matrix, ppi_mask, non_ppi_mask)
            #nega_loss = self.batch_hard_triplets(dist_matrix, non_ppi_mask, diff_mask)
        else:
            posi_loss = self.batch_all_triplets(dist_matrix, ppi_mask, diff_mask)
            nega_loss = self.batch_all_triplets(dist_matrix, non_ppi_mask, diff_mask)
        # 
        return torch.clamp(posi_loss, min=0, max=3.0)
    
    def batch_all_triplets(self, dist_matrix, pos_mask, diff_mask):
        """ """
        # max(d(a,p) - d(a,n) + margin, 0)
        ap_dist = dist_matrix.unsqueeze(2)  # [B, B, 1] # positive
        an_dist = dist_matrix.unsqueeze(1)  # [B, 1, B] # negative
        
        margin = self.adaptive_margin()
        triplet_loss = F.relu(ap_dist - an_dist + margin)
        
        pos_valid = pos_mask.unsqueeze(2)
        neg_valid = diff_mask.unsqueeze(1)
        valid_mask = (pos_valid & neg_valid) & (triplet_loss > self.eps)
        
        if valid_mask.any():
            return triplet_loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=dist_matrix.device)
    
    def batch_hard_triplets(self, dist_matrix, pos_mask, diff_mask):
        """ """
        margin = self.adaptive_margin()
        
        ap_dist = dist_matrix.clone()
        ap_dist[~pos_mask] = -1  #
        
        #
        hardest_pos_idx = torch.argmax(ap_dist, dim=1)
        hardest_pos_dist = torch.gather(dist_matrix, 1, hardest_pos_idx.unsqueeze(1)).squeeze(1)
        
        an_dist = dist_matrix.clone()
        an_dist[~diff_mask] = 10
        
        hardest_neg_idx = torch.argmin(an_dist, dim=1)
        hardest_neg_dist = torch.gather(dist_matrix, 1, hardest_neg_idx.unsqueeze(1)).squeeze(1)
        
        losses = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        
        valid_pos = hardest_pos_dist > -1
        valid_neg = hardest_neg_dist < 5  #
        valid_mask = valid_pos & valid_neg
        
        if valid_mask.any():
            return losses[valid_mask].mean()
        return torch.tensor(0.0, device=dist_matrix.device)

    def reset_stats(self):
        """ """
        self.avg_pos_dist.fill_(0.5)
        self.avg_neg_dist.fill_(1.5)
    

####################################################################################

def get_alpha(current_epoch, total_epochs=200, max_alpha=0.8, min_alpha=0.2, scheduler_type="cosine"):
    """Cosine/linear scheduler"""
    progress = current_epoch / total_epochs

    if scheduler_type == 'linear':
        return max_alpha - (max_alpha - min_alpha) * progress

    elif scheduler_type == 'cosine':
        if current_epoch > total_epochs:
            return min_alpha
        else:
            alpha_range = max_alpha - min_alpha
            return min_alpha + 0.5 * alpha_range * (1 + math.cos(progress * math.pi))

    elif scheduler_type == 'exponential':
        decay_rate = 0.95
        return min_alpha + (max_alpha - min_alpha) * (decay_rate ** (10 * progress))

    else:  
        if progress < 0.3:
            return max_alpha
        elif progress < 0.7:
            return 0.5
        else:
            return min_alpha
