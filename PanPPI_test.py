import torch
import sys
import os
from model.class_wrapper import load_config
from model.class_ESM2 import ESM2
from collections import OrderedDict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,roundup_power2_divisions:4,garbage_collection_threshold:0.95"
#torch.set_float32_matmul_precision('medium')
os.environ["WANDB_MODE"] = "offline"

pt_path = sys.argv[1]
test_path = sys.argv[2]
seed = int(sys.argv[3])
device = sys.argv[4]
output_path = sys.argv[5]

cfg = load_config("raw_MINT_param/esm2_t33_650M_UR50D.json") # model config
model = ESM2(num_layers=cfg.encoder_layers, embed_dim=cfg.encoder_embed_dim, attention_heads=cfg.encoder_attention_heads, token_dropout=cfg.token_dropout, use_multimer=True)

###################################################################

checkpoint = torch.load(pt_path, map_location="cpu")
new_checkpoint = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items())
model.load_state_dict(new_checkpoint, strict=False)

#model.save_pretrained("650M_freeze_classify/model_bin")  # config.json + pytorch_model.bin
###############################################################################################################

import argparse
import json
import yaml
import re
import numpy as np
import random
import pandas as pd
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from model.class_dataset import Class_CSVDataset, Class_CollateFn, Seqs_CSVDataset, Alphabet, Only_Seqs_CSVDataset
from model.class_wrapper import ESMWrapper,upgrade_state_dict,ESM_Inference_Wrapper
from transformers import EsmForSequenceClassification
from model.class_loss import PPIMemoryInfoNCE, SimularityLoss, SupervisedContrastiveLoss, NewPPIMemoryInfoNCE, SupervisedTripletLoss,get_alpha
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,precision_recall_curve,classification_report, auc, mean_squared_error
torch.multiprocessing.set_sharing_strategy('file_system')

###############################################################################################################################################

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

cfg = argparse.Namespace()
with open("raw_MINT_param/esm2_t33_650M_UR50D.json") as f:
    cfg.__dict__.update(json.load(f))

args = argparse.Namespace()
with open("args_parameter_PanPPI.yaml") as f:
    args.__dict__.update(yaml.safe_load(f))

test_dataset = Only_Seqs_CSVDataset(test_path,\
                          'source','target', 'species', 'class')
test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=4, batch_size=16, collate_fn=Class_CollateFn(512), shuffle=False)

def classification_metrics(targets, predictions, threshold=0.5):
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(targets, binary_predictions)
    f1 = f1_score(targets, binary_predictions)
    precision = precision_score(targets, binary_predictions, pos_label=1)
    recall = recall_score(targets, binary_predictions, pos_label=1)
    auc_score = roc_auc_score(targets, predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    auprc = auc(recall_vals, precision_vals)

    return {
        "Accuracy": accuracy,
        "AUPRC": auprc,
        "F1 Score": f1,
        "Precise":precision,
        "Recall":recall,
        "AUROC": auc_score,
        'y_probs': predictions,
        'y_pred': binary_predictions,
        'y_true': targets
    }

def test_atten_glr_contrastive_model(model,test_dataloader,device,threshold=0.5):

    model.eval()
    model = model.to(device)
    loss_fct = torch.nn.CrossEntropyLoss()
    constract_loss = SupervisedTripletLoss(margin=0.2, strategy='batch_all_triplets')

    loss_fct = loss_fct.to(device)
    constract_loss = constract_loss.to(device)

    logits_list = []
    preds = []
    targets = []
    y_pred = []
    total_loss = 0

    test_metrics = {
                    'entro_loss': 0.0, 
                    'cons_loss': 0.0,
                    'samples': 0
                    }
 
    with torch.no_grad():

        for tokens, chains_id, class_id in test_dataloader:

            mask = ((~tokens.eq(model.cls_idx)) \
                  & (~tokens.eq(model.eos_idx)) \
                  & (~tokens.eq(model.padding_idx)))

            mask = mask.to(device)
            tokens = tokens.to(device)
            chains_id = chains_id.to(device)
            class_id = class_id.to(device)
    
            mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask
            inp = torch.where(mask, model.mask_idx, tokens)
            output = model(inp, chains_id)

            logits = output['logits']
            embedding = output['linear_feature']

            entro_loss = loss_fct(logits.view(-1, 2), class_id[:,0].view(-1))
            cons_loss = constract_loss(embedding, class_id[:,0])

            logits_list.append(logits.cpu().detach().numpy())
            prob = torch.nn.functional.softmax(logits, dim=1)
            preds.append(prob[:,1].cpu().detach().numpy())
            targets.append(class_id[:,0].cpu().numpy())
            class1_probs = prob[:,1]
            predicted = torch.where(class1_probs >= threshold,
                            torch.ones_like(class1_probs),
                            torch.zeros_like(class1_probs))
            y_pred.append(predicted.cpu().numpy())

        logits_list = np.concatenate(logits_list)
        y_predict = np.hstack(y_pred)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

    metrics = classification_metrics(targets, preds, threshold)
    metrics['logits'] = logits_list
    metrics['predict'] = y_predict
    metrics['prob'] = preds
    metrics['targets'] = targets

    print(classification_report(targets, y_predict, target_names=[f'class_0(neg)', f'class_1(posi)']))
     
    return metrics

metrics = test_atten_glr_contrastive_model(model, test_loader, device, threshold=0.5)

df = pd.read_csv(test_path, names=['source', 'target', 'species', 'class'])
df = df.dropna()

data = {'gene': list(df.iloc[:, 2]),
        'predict': metrics['predict'],
        'true':metrics['targets'],
        'softmax':metrics['prob']
       }

df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
print(metrics)
