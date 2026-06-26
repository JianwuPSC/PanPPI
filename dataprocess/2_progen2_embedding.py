
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

fasta_csv_path = sys.argv[1]
output_path = sys.argv[2]
device = sys.argv[3] # cuda:0

##################################

import os
import time
import random
import argparse
import sys
sys.path.append('/path/progen2/')

import torch
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM, ProGenModel


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def ProGen_embedding(model,tokenizer,tokens,device):
    with torch.no_grad():
        model =  model.to(device)
        target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
        output = model(target)
    embedding = output['last_hidden_state'].mean(dim=0).cpu().detach().numpy()

    del output; del model
    torch.cuda.empty_cache()
    return embedding

ckpt = '/path/progen2/weight/progen2-large'
model = ProGenModel.from_pretrained(ckpt)
tokenizer = create_tokenizer_custom(file='/path/progen2/tokenizer.json')

##### progen_embedding

name_list=[]
represent_list=[]
class_list=[]

esm_represent_source_list=[]
esm_represent_target_list=[]

df = pd.read_csv(fasta_csv_path, names=['col1', 'col2', 'species', 'classify'])
for index,row in df.iterrows():

    source = row['col1'] # traget P53691
    target =  row['col2'] # source P02829
    species = row['species']
    ppi_class = row['classify']

    #source_seq_input = [(species,source)]
    #target_seq_input = [(species,target)]
    if len(source) > 1024:
        start = random.randint(0, len(source) - 1024 + 1)
        source = source[start : start + 1024]

    if len(target) > 1024:
        start = random.randint(0, len(target) - 1024 + 1)
        target = target[start : start + 1024]
    #source_seq_repre = ESM2_representation(seq_model,seq_alphabet,source_seq_input,device)
    #target_seq_repre = ESM2_representation(seq_model,seq_alphabet,target_seq_input,device)
    source_seq_repre = ProGen_embedding(model,tokenizer,source,device)
    target_seq_repre = ProGen_embedding(model,tokenizer,target,device)
                   
    name_list.append(species)
    class_list.append(ppi_class)
    
    esm_represent_source_list.append(source_seq_repre)
    esm_represent_target_list.append(target_seq_repre)

    del source_seq_repre; del target_seq_repre
    torch.cuda.empty_cache()

    print(index)

################################################################

from sklearn.model_selection import train_test_split

name_df = pd.DataFrame(name_list)
name_df = name_df.rename(columns={0: 'name'})
represent_df = pd.DataFrame(represent_list)

esm_represent_source_df = pd.DataFrame(esm_represent_source_list)
esm_represent_target_df = pd.DataFrame(esm_represent_target_list)

class_df = pd.DataFrame(class_list)
class_df = class_df.rename(columns={0: 'class'})

esm_represent_soucer_df = pd.concat([name_df,esm_represent_source_df,esm_represent_target_df,class_df],axis=1)
esm_represent_soucer_df.to_csv(output_path+'/'+'progen_represent_df.csv', index=False,header=False)

