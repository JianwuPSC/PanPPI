
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import sys
import random
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

fasta_csv_path = sys.argv[1]
output_path = sys.argv[2]
device = sys.argv[3] # cuda:0


##################################

def ESM2_representation(model,alphabet,fasta_seq,device):
    
    """
    model_path : /home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
    fasta_seq : [('ID','MDAFREA')]
    """
    #model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available():
        model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    data_batch = batch_converter(fasta_seq)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]
    
    label_list = []
    represent_list = []
    
    with torch.no_grad():
        
        labels, strs, toks = data_batch
        
        if torch.cuda.is_available():
            toks = toks.to(device=device, non_blocking=True)
            
        print(f"Device: {toks.device}")
        model.eval()
        out = model(toks, repr_layers=repr_layers, return_contacts=False)

        logits = out["logits"].to(device="cpu")
        representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
        for i, label in enumerate(labels):
                
            label_list.append(label)
            truncate_len = min(1280, len(strs[i]))
            represent_list.append((representations[repr_layers[0]])[i, 1 : truncate_len + 1].mean(0).clone())

        result = {'label':label_list, 'representation':represent_list[0].detach().numpy()}
         
    return result

#######################################

##### esm_embedding
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
seq_model, seq_alphabet = pretrained.load_model_and_alphabet('/home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
#seq_model.eval()

name_list=[]
represent_list=[]
class_list=[]

esm_represent_source_list=[]
esm_represent_target_list=[]

df = pd.read_csv(fasta_csv_path, names=['col1', 'col2', 'species', 'classify'])
for index,row in df.iterrows():
    source = row['col1'] # traget P53691
    target = row['col2'] # source P02829
    species = row['species']
    ppi_class = row['classify']

    if len(source) > 1024:
        start = random.randint(0, len(source) - 1024 + 1)
        source = source[start : start + 1024]
    if len(target) > 1024:
        start = random.randint(0, len(target) - 1024 + 1)
        target = target[start : start + 1024]

    source_seq_input = [(species,source)]
    target_seq_input = [(species,target)]
    
    source_seq_repre = ESM2_representation(seq_model,seq_alphabet,source_seq_input,device)
    target_seq_repre = ESM2_representation(seq_model,seq_alphabet,target_seq_input,device)
    
    name_list.append(species)
    class_list.append(ppi_class)
    
    esm_represent_source_list.append(source_seq_repre['representation'])
    esm_represent_target_list.append(target_seq_repre['representation'])

    del source_seq_repre; del target_seq_repre
    torch.cuda.empty_cache()

    print(index)

################################################################

from sklearn.model_selection import train_test_split

name_df = pd.DataFrame(name_list)
name_df = name_df.rename(columns={0: 'name'})

esm_represent_source_df = pd.DataFrame(esm_represent_source_list)
esm_represent_target_df = pd.DataFrame(esm_represent_target_list)

class_df = pd.DataFrame(class_list)
class_df = class_df.rename(columns={0: 'class'})

esm_represent_soucer_df = pd.concat([name_df,esm_represent_source_df,esm_represent_target_df,class_df],axis=1)
esm_represent_soucer_df.to_csv(output_path+'/'+'esm_represent_df.csv', index=False,header=False)

