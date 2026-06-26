
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
import sys
import torch
import random
from model.class_dataset import Alphabet
sys.path.append("/path/mint/") # your mint path
from mint.helpers.extract import load_config, MINTWrapper

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

fasta_csv_path = sys.argv[1]
output_path = sys.argv[2]
device = sys.argv[3] # cuda:0


def Mint_embedding(wrapper,alphabet,source,target,device):
    
    source = alphabet.encode("<cls>" + source.replace("J", "L") + "<eos>")
    target = alphabet.encode("<cls>" + target.replace("J", "L") + "<eos>")    
    if len(source) > 512:
        start = random.randint(0, len(source) - 512 + 1)
        source = source[start : start + 512]
    if len(target) > 512:
        start = random.randint(0, len(target) - 512 + 1)
        target = target[start : start + 512]

    source_ids = torch.zeros(len(source),dtype=torch.int32).unsqueeze(0)
    target_ids = torch.ones(len(target),dtype=torch.int32).unsqueeze(0)
    chain_ids = torch.cat([source_ids,target_ids], dim = -1)
    chain_ids = chain_ids.to(device)
    
    tokens = torch.cat([torch.tensor(source).unsqueeze(0),torch.tensor(target).unsqueeze(0)], dim = -1)
    tokens = tokens.to(device)
    
    embeddings = wrapper(tokens, chain_ids)
    
    return embeddings[0].cpu().detach().numpy()

##################################################################################

cfg = load_config("/path/mint/data/esm2_t33_650M_UR50D.json") # model config
checkpoint_path = '/path/mint/model/mint.ckpt'

wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)
alphabet = Alphabet.from_architecture('ESM-1b')

name_list = []
class_list = []
esm_represent_source_list = []

df = pd.read_csv(fasta_csv_path, names=['col1', 'col2', 'species', 'classify'])
for index,row in df.iterrows():

    source = row['col1'] # traget P53691
    target =  row['col2'] # source P02829
    species = row['species']
    ppi_class = row['classify']

    source_seq_repre = Mint_embedding(wrapper,alphabet,source,target,device)
    name_list.append(species)
    class_list.append(ppi_class)
    esm_represent_source_list.append(source_seq_repre)

    del source_seq_repre
    torch.cuda.empty_cache()

    print(index)

##### Mnit_embedding

esm_represent_source_df = pd.DataFrame(esm_represent_source_list)
class_df = pd.DataFrame(class_list)
class_df = class_df.rename(columns={0: 'class'})
name_df = pd.DataFrame(name_list)
name_df = name_df.rename(columns={0: 'name'})

esm_represent_soucer_df = pd.concat([name_df,esm_represent_source_df,class_df],axis=1)
esm_represent_soucer_df.to_csv(output_path+'/'+'mint_represent_df.csv', index=False,header=False)
